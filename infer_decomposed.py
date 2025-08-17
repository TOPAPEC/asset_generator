# wan_runner.py
import os, sys, math, torch, imageio
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    GGUFQuantizationConfig,
)
from diffusers.utils import load_image, export_to_video
from diffusers.hooks import apply_group_offloading
import diffusers

from lib.WanVideoToVideo import WanVideoToVideo

diffusers.logging.set_verbosity_info()

_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*a, **k):
    k.pop("enable_gqa", None)
    return _orig_sdpa(*a, **k)
F.scaled_dot_product_attention = _patched_sdpa

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
WORKSPACE_DIR = os.environ.get("HF_HOME", "/workspace/hf_cache")

WAN_BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_TRANSFORMER_SUBFOLDER = "transformer"
WAN_T1_URL = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf"
WAN_T2_URL = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf"

LORA_REPO = "Kijai/WanVideo_comfy"
LORA_LIGHTX2V = "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
# (Kept your Lightning Hi/Low files in case you want to load non-diffusers-style manually later)
LORA_LIGHTNING_HIGH = "Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors"
LORA_LIGHTNING_LOW  = "Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors"

DEFAULT_PROMPT = (
    "A 360 degrees view of the character (camera MUST circle around the character), "
    "character begins to run forward and is running until the end. "
    "Camera is moving fast and is able to capture full front view, side view and full back view. "
    "All animations are smooth and detailed"
)
DEFAULT_NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def to_pil(im):
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, np.ndarray):
        arr = im
        if arr.dtype != np.uint8:
            mn = float(arr.min())
            mx = float(arr.max())
            if mx <= 1.0 and mn >= 0.0:
                arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, "RGB")
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, "RGBA").convert("RGB")
        if arr.ndim == 2:
            return Image.fromarray(arr, "L").convert("RGB")
    raise TypeError(f"Unsupported frame type {type(im)}")

def _ensure_dir(d: str):
    if d and len(d) > 0:
        os.makedirs(d, exist_ok=True)

def _save_frames(frames: List[Image.Image], out_frames_dir: str, prefix: str = "frame"):
    _ensure_dir(out_frames_dir)
    for i, im in enumerate(frames):
        to_pil(im).save(os.path.join(out_frames_dir, f"{prefix}_{i:03d}.png"))

def _export_mp4(frames: List[Image.Image], out_mp4_path: str, fps: int = 16):
    if not out_mp4_path:
        return
    _ensure_dir(os.path.dirname(out_mp4_path) if os.path.dirname(out_mp4_path) else ".")
    export_to_video(frames, out_mp4_path, fps=fps)

@dataclass(frozen=True)
class NoLoraWarmup:
    steps: int
    cfg: float

class WanVideo:
    """
    Static global class managing a singleton Wan I2V pipeline and Lightning LoRA toggling.
    """
    _pipe: Optional[WanImageToVideoPipeline] = None
    _lora_loaded: bool = False
    _lora_enabled: bool = False
    _adapter_names: Tuple[str, str] = ("lightning", "lightning_2")
    _adapter_weights: Tuple[float, float] = (3.0, 1.5)
    _device: str = "cuda"
    _height_default: int = 768
    _width_default: int = 768

    @classmethod
    def init(
        cls,
        device: str = "cuda",
        max_memory: Optional[Dict[int, str]] = None,
        height: int = 768,
        width: int = 768,
        offload_device: str = "cpu",
        onload_device: str = "cuda",
        quant_compute_dtype: torch.dtype = torch.bfloat16,
        pipe_dtype: torch.dtype = torch.bfloat16,
    ):
        if cls._pipe is not None:
            return  # already initialized

        print("Loading GGUF transformers...")
        t_hi = WanTransformer3DModel.from_single_file(
            WAN_T1_URL,
            quantization_config=GGUFQuantizationConfig(compute_dtype=quant_compute_dtype),
            torch_dtype=pipe_dtype,
            config=WAN_BASE_REPO,
            subfolder=WAN_TRANSFORMER_SUBFOLDER,
        )
        t_lo = WanTransformer3DModel.from_single_file(
            WAN_T2_URL,
            quantization_config=GGUFQuantizationConfig(compute_dtype=quant_compute_dtype),
            torch_dtype=pipe_dtype,
            config=WAN_BASE_REPO,
            subfolder=WAN_TRANSFORMER_SUBFOLDER,
        )
        print("Loaded GGUF models")

        if max_memory is None:
            max_memory = {0: "48GB"}

        print("Initializing WanImageToVideoPipeline...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_BASE_REPO,
            torch_dtype=pipe_dtype,
            transformer=t_hi,
            transformer_2=t_lo,
            max_memory=max_memory,
        )

        # Group offloading configuration (keep your original structure)
        pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        # transformer_2 offloading is commented intentionally, like your original
        # pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=2)

        pipe.to(device)
        cls._pipe = pipe
        cls._device = device
        cls._height_default = height
        cls._width_default = width

    @classmethod
    def _require_pipe(cls):
        if cls._pipe is None:
            raise RuntimeError("WanVideo.init() must be called before using generation APIs.")

    @classmethod
    def load_lora(cls, adapter_weights: Tuple[float, float] = (3.0, 1.5)):
        cls._require_pipe()
        if cls._lora_loaded:
            # Update weights if needed
            cls._pipe.set_adapters(list(cls._adapter_names), adapter_weights=list(adapter_weights))
            cls._adapter_weights = adapter_weights
            return

        print("Loading Lightning LoRA adapters...")
        cls._pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_LIGHTX2V,
            adapter_name=cls._adapter_names[0],
        )
        kwargs = {"load_into_transformer_2": True}
        cls._pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_LIGHTX2V,
            adapter_name=cls._adapter_names[1],
            **kwargs,
        )
        cls._pipe.set_adapters(list(cls._adapter_names), adapter_weights=list(adapter_weights))
        cls._lora_loaded = True
        cls._adapter_weights = adapter_weights
        print("LoRA loaded.")

    @classmethod
    def enable_lora(cls):
        cls._require_pipe()
        if not cls._lora_loaded:
            cls.load_lora(cls._adapter_weights)
        cls._lora_enabled = True
        print("LoRA enabled.")

    @classmethod
    def disable_lora(cls):
        cls._require_pipe()
        cls._pipe.unload_lora_weights()
        cls._lora_enabled = False
        cls._lora_loaded = False
        print("LoRA disabled (unloaded).")

    @classmethod
    def _run_pass(
        cls,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
    ) -> List[Image.Image]:
        if num_steps <= 0:
            return []
        result = cls._pipe(
            image=load_image(image).convert("RGB"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        return [to_pil(f) for f in result.frames[0]]

    @classmethod
    def _phase_split(cls, total_steps: int, warmup: Optional[NoLoraWarmup]) -> Tuple[int, int]:
        s1 = warmup.steps if warmup else 0
        s1 = max(0, min(total_steps, s1))
        s2 = total_steps - s1
        return s1, s2

    @classmethod
    def i2v(
        cls,
        image_path: str,
        num_steps: int,
        num_frames: int,
        out_frames_dir: str,
        out_mp4_path: str,
        no_lora: Optional[Dict[str, float]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEG,
        guidance_scale_after: float = 1.0,
        fps: int = 16,
    ) -> Tuple[List[Image.Image], str]:

        cls._require_pipe()
        height = height or cls._height_default
        width = width or cls._width_default

        warm = None
        if no_lora is not None:
            warm = NoLoraWarmup(steps=int(no_lora.get("steps", 0)), cfg=float(no_lora.get("cfg", 1.0)))

        s1, s2 = cls._phase_split(num_steps, warm)

        # Load + enable LoRA for potential second phase (we may disable first)
        cls.load_lora(cls._adapter_weights)

        frames_all: List[Image.Image] = []
        init_image = Image.open(image_path).convert("RGB")

        # Phase 1 (no LoRA)
        if s1 > 0:
            cls.disable_lora()
            print(f"[i2v] Warmup without LoRA: steps={s1}, cfg={warm.cfg}, frames={num_frames}")
            frames_warm = cls._run_pass(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=s1,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=warm.cfg,
            )
            frames_all = frames_warm
            # Last frame becomes conditioning for LoRA phase
            # init_image = frames_warm[-1] if len(frames_warm) > 0 else init_image

        # Phase 2 (with LoRA)
        if s2 > 0:
            cls.enable_lora()
            print(f"[i2v] LoRA phase: steps={s2}, cfg={guidance_scale_after}, frames={num_frames}")
            frames_final = cls._run_pass(
                image=frames_all,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=s2,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale_after,
            )
            frames_all = frames_final

        # Output
        _ensure_dir(out_frames_dir)
        _save_frames(frames_all, out_frames_dir, prefix="frame")
        _export_mp4(frames_all, out_mp4_path, fps=fps)
        return frames_all, out_mp4_path

    @classmethod
    def flf2v(
        cls,
        first_image_path: str,
        last_image_path: str,
        num_steps: int,
        num_frames: int,
        out_frames_dir: str,
        out_mp4_path: str,
        no_lora: Optional[Dict[str, float]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        prompt_first: str = DEFAULT_PROMPT,
        prompt_last: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEG,
        guidance_scale_after: float = 1.0,
        fps: int = 16,
        bridge_frames: int = 8,
    ) -> Tuple[List[Image.Image], str]:
        """
        First-and-Last-image-to-Video: generate two conditioned segments and connect them with a crossfade bridge.
        - Applies no_lora warmup only to the first segment (logical entry).
        """
        cls._require_pipe()
        height = height or cls._height_default
        width = width or cls._width_default
        n1 = num_frames // 2
        n2 = num_frames - n1

        # Segment A: from first image with optional warmup
        frames_A, _ = cls.i2v(
            image_path=first_image_path,
            num_steps=num_steps,
            num_frames=max(n1, 1),
            out_frames_dir=os.path.join(out_frames_dir, "segA"),
            out_mp4_path=os.path.join(os.path.dirname(out_mp4_path), "segA.mp4"),
            no_lora=no_lora,
            height=height,
            width=width,
            prompt=prompt_first,
            negative_prompt=negative_prompt,
            guidance_scale_after=guidance_scale_after,
            fps=fps,
        )

        # Segment B: from last image; keep LoRA ON (resumes stylization)
        cls.enable_lora()
        frames_B = cls._run_pass(
            image=Image.open(last_image_path).convert("RGB"),
            prompt=prompt_last,
            negative_prompt=negative_prompt,
            num_steps=max(num_steps, 1),
            num_frames=max(n2, 1),
            height=height,
            width=width,
            guidance_scale=guidance_scale_after,
        )

        # Crossfade bridge (explicit simple implementation)
        bridge: List[Image.Image] = []
        if bridge_frames > 0 and len(frames_A) > 0 and len(frames_B) > 0:
            a_end = to_pil(frames_A[-1]).resize((width, height))
            b_start = to_pil(frames_B[0]).resize((width, height))
            for i in range(1, bridge_frames + 1):
                alpha = i / (bridge_frames + 1)
                bridge.append(Image.blend(a_end, b_start, alpha))

        frames_all = frames_A + bridge + frames_B

        # Output
        _ensure_dir(out_frames_dir)
        _save_frames(frames_all, out_frames_dir, prefix="frame")
        _export_mp4(frames_all, out_mp4_path, fps=fps)
        return frames_all, out_mp4_path


if __name__ == "__main__":
    WanVideo.init(
        device="cuda",
        height=768,
        width=768,
        offload_device="cpu",
        onload_device="cuda",
    )

    frames, mp4_path = WanVideo.i2v(
        image_path="image.png",
        num_steps=6,                 # total steps
        num_frames=48,
        out_frames_dir="raw_frames",
        out_mp4_path="output.mp4",
        no_lora={"steps": 5, "cfg": 7.0},   # warmup S=2, cfg=1.0; then 2 steps with LoRA
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEG,
        guidance_scale_after=1.0,
        fps=16,
    )

    # 3) FLF2V example (optional)
    # WanVideo.flf2v(
    #     first_image_path="first.png",
    #     last_image_path="last.png",
    #     num_steps=4,
    #     num_frames=81,
    #     out_frames_dir="raw_frames_flf2v",
    #     out_mp4_path="output_flf2v.mp4",
    #     no_lora={"steps": 2, "cfg": 1.0},
    #     guidance_scale_after=1.0,
    # )
