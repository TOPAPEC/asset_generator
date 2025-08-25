import os, sys, torch, imageio
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    GGUFQuantizationConfig,
    LCMScheduler, UniPCMultistepScheduler,
)
from diffusers.utils import load_image, export_to_video
from diffusers.hooks import apply_group_offloading
import diffusers
diffusers.logging.set_verbosity_info()
from PIL import Image
import safetensors 
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*a, **k):
    k.pop("enable_gqa", None)
    return _orig_sdpa(*a, **k)
F.scaled_dot_product_attention = _patched_sdpa

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers

import os, glob

def get_first_and_last_frame(folder_path):
    frames = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if not frames:
        raise FileNotFoundError
    return load_image(Image.open(frames[0])).convert("RGB"), load_image(Image.open(frames[-1])).convert("RGB")

import os, glob, re

def get_last_frame_number(folder_path):
    frames = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if not frames:
        raise FileNotFoundError
    return int(re.search(r'\d+', os.path.basename(frames[-1])).group())

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

workspace_dir = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_cache"

lora_path = hf_hub_download(
    repo_id="Kijai/WanVideo_comfy",
    filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    # cache_dir=workspace_dir,
)

print("Loading models")
t_hi = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    subfolder="transformer",
    # cache_dir=workspace_dir
)

t_lo = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    subfolder="transformer",
    # cache_dir=workspace_dir,
)
print("Loaded models")


max_memory={0: "48GB"}
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
    torch_dtype=torch.bfloat16, 
    # cache_dir=workspace_dir, 
    transformer=t_hi,
    transformer_2=t_lo,
    max_memory=max_memory,
)

offload_device = "cpu"
onload_device = "cuda"
# pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=4)
lora_path = hf_hub_download(
    repo_id="Kijai/WanVideo_comfy",
    filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    # cache_dir=workspace_dir
)

pipe.to("cuda")

pipe.load_lora_weights(
   "Kijai/WanVideo_comfy", 
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
    adapter_name="lightning"
)
kwargs = {}
kwargs["load_into_transformer_2"] = True
pipe.load_lora_weights(
  "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", 
    adapter_name="lightning_2", **kwargs
)
pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[3., 1.5])

# fimg, limg = get_first_and_last_frame("raw_frames/")

fimg = load_image(Image.open("image.png")).convert("RGB")
out_dir = "raw_frames"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("videos/", exist_ok=True)

NEGATIVE = "bright colors, overexposed, motion blur, static, blurred details, subtitles, watermark, style, artwork, painting, picture, still, worst quality, low quality, jpeg artifacts, ugly, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, cluttered background, many people in background, walking backwards, distorted face, distorted features, blurry eyes"

def generate_sequence(
    pipe,
    image,
    prompt,
    negative_prompt=NEGATIVE,
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
    out_dir="frames",
    video_path="videos/output.mp4",
    frame_offset=0,
    fps=16
):
    print(f"Starting generation: {prompt[:50]}...")  # short preview of prompt

    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        height=height,
        width=width,
    ).frames[0]

    for i, im in enumerate(frames):
        to_pil(im).save(os.path.join(out_dir, f"frame_{frame_offset + i + 1:03d}.png"))

    export_to_video(frames, video_path, fps=fps)
    print(f"Video saved: {video_path}")

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Photoreal shoreline at golden hour. Shot: WS→MWS, eye-level, 35–50mm. Camera: steady tracking forward behind her with a subtle push-in ~20%. Motion: she starts running from frame one; hair/cloth react; waves move; horizon steady. Lighting: warm rim-light, light haze. Stability: face and outfit unchanged. Duration ~5 s.",
    frame_offset=81*1,
    out_dir=out_dir,
    video_path="videos/output1.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Small-town brick wall alley. Shot: MS side profile, eye-level, 50mm. Camera: constant lateral track matching her pace (left→right), no extra moves. Motion: she breaks into a run to screen-left from the start; ponytail swings; mild background parallax. Light: soft overcast. Stability: identity locked. Duration ~5 s.",
    frame_offset=81*2,
    out_dir=out_dir,
    video_path="videos/output2.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Cyclorama boxing gym. Shot: MS→MCU, eye-level, 50mm, shallow DoF. Camera: gentle dolly-in ~25%, no pan/tilt. Motion: she sets stance then throws jab–cross–jab with correct wrist alignment and guard; chalk dust motes drift. Light: soft top + rim. Stability: face/outfit consistent. Duration ~5 s.",
    frame_offset=81*3,
    out_dir=out_dir,
    video_path="videos/output3.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Brick wall portrait. Shot: MS→CU centered, eye-level, 85mm tele. Camera: slow dolly-in ~30% to a clean face/neck/upper chest CU; no zoom artifacts. Motion: subtle breathing and natural eye saccades; hair micro-movement. Light: soft overcast with warm skin tone. Stability: facial landmarks stable. Duration ~5 s.",
    frame_offset=81*4,
    out_dir=out_dir,
    video_path="videos/output4.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Forest clearing with soft canopy light. Shot: starts MS full-body profile, 50mm. Camera: gentle dolly-in ~25% ending on MCU framing face, neck, shoulders. Motion: she talks and laughs, then shifts to stern expression; subtle head/shoulder movement. Light: cool ambient with warm rim. Stability: outfit/face constant. Duration ~5 s.",
    frame_offset=81*5,
    out_dir=out_dir,
    video_path="videos/output5.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Cozy town street. Shot: starts MWS (upper body), 35mm. Camera: subtle push-in ~20% as she performs smooth 180° turn, ending on MCU of head/neck/upper back. Motion: hair sways as she turns from face-on to back view. Light: warm/cool afternoon. Stability: proportions and facial features consistent. Duration ~5 s.",
    frame_offset=81*6,
    out_dir=out_dir,
    video_path="videos/output6.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Open town plaza. Shot: WS→MS, 35mm, centered. Camera: steady orbit arc ~120° clockwise at constant radius. Motion: she drops any objects, takes fighting stance, executes clean roundhouse kicks with hip rotation and guard up; ground scuffs appear. Light: crisp daylight. Stability: limbs well-formed; identity locked. Duration ~5 s.",
    frame_offset=81*7,
    out_dir=out_dir,
    video_path="videos/output7.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Town scene with chair. Shot: starts WS full height, 35mm. Camera: smooth orbit arc ~100° around her as she sits, combined with gentle push-in so mid-body and face become clearer during orbit. Motion: she takes chair, sits gracefully, adjusts posture; cloth settles. Light: warm street practicals. Stability: face/outfit consistent. Duration ~5 s.",
    frame_offset=81*8,
    out_dir=out_dir,
    video_path="videos/output8.mp4"
)

