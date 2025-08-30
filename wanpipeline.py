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
pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
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

fimg = load_image(Image.open("input_pics/soldier.png")).convert("RGB")
out_dir = "raw_frames"

os.makedirs(out_dir, exist_ok=True)
os.makedirs("videos/", exist_ok=True)

NEGATIVE = "bright colors, overexposed, motion blur, static, blurred details, subtitles, watermark, style, artwork, painting, picture, still, worst quality, low quality, jpeg artifacts, ugly, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, cluttered background, many people in background, walking backwards, distorted face, distorted features, blurry eyes"

def generate_sequence(
    pipe,
    image,
    prompt,
    negative_prompt=NEGATIVE,
    num_inference_steps=8,
    guidance_scale=1.0,
    num_frames=96,
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
    prompt="""
    Robo-soldier in dark neon-lit alley at night, armored plating wet from rain, glowing orange visor; cinematic photoreal style.

Beat 1 (0–24f): medium-wide, 35 mm, eye-level; steady dolly-in ~20%; robo-soldier walks forward, heavy steps, arms swaying; neon signs flicker; horizon steady.  
Beat 2 (24–48f): camera cranes upward fast ~2 m, tilt down; leap moment, soldier crouches and jumps; frame holds his midair posture, legs bent, arms extended slightly.  
Beat 3 (48–72f): camera arcs overhead ~120°, top-down angle; captures body at jump apex, glowing visor visible, then falling back down; shallow depth of field.  
Beat 4 (72–96f): camera drops behind soldier, medium shot, 50 mm; tracks backward as soldier lands hard, knees bend, dust splash; then resumes forward walk, seen from back.  

Lighting: wet alley reflections, cold blue fill + orange visor glow.  
Color grade: teal–orange cinematic HDR.  
Stability: armor intact, visor glow constant, background fixed.  
""",
    frame_offset=96*1,
    out_dir=out_dir,
    video_path="videos/output1.mp4"
)


generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="""
Robo-soldier in neon-lit futuristic street, metallic armor glistening with moisture, glowing orange visor. Lighting slightly brighter than dark alley baseline, with more visible rim and fill light.

Beat 1 (0–24f): medium-wide, 35 mm, side angle (profile view); camera pans right slowly ~20%; robo-soldier begins moving forward on his arms, mechanical limbs pressing into wet pavement, sparks from joints.  
Beat 2 (24–48f): camera holds side profile, dolly-in ~15%; soldier continues arm-walk, then shifts balance preparing for roll; mechanical body tenses, arms bend.  
Beat 3 (48–72f): camera tracks side view; soldier performs forward roll (quilt-like tumble), motion blur avoided; armor plates glint under brighter streetlight; background stays stable.  
Beat 4 (72–96f): camera still side-on, medium shot, 50 mm; soldier rises into aggressive fighting stance, feet planted, arms raised in combat guard; glowing visor intensifies slightly; reflections on wet ground enhance silhouette.

Lighting: slightly brighter, with cold blue key and orange visor glow as contrast.  
Color grade: cinematic teal-orange, HDR sharp detail.  
Stability: armor intact, visor glow steady, background consistent. """,
    frame_offset=96*2,
    out_dir=out_dir,
    video_path="videos/output2.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="""
Robo-soldier close-up, framed from torso up, glowing orange visor, metallic armor with rain droplets. Cinematic photoreal style.

Beat 1 (0–24f): medium close-up, 50 mm, eye-level; camera steady, slight dolly-in ~10%; robot faces front, armor wet, visor glowing steadily; subtle breathing movement.  
Beat 2 (24–48f): same shot size; robot nods head slightly down then up, showing visor under different light angles; reflections flicker; background neon blur steady.  
Beat 3 (48–72f): camera arcs left ~45° around torso; robot slowly turns his head, then begins rotating shoulders.  
Beat 4 (72–96f): close-up (85 mm), shallow DoF; camera settles behind robot; full reveal of armored spine and back of head, glowing elements at neck and shoulder joints visible; horizon fixed.

Lighting: moody neon, slightly brighter on visor front, rim-light along shoulders, soft reflections on wet plating.  
Color grade: teal–orange, cinematic HDR.  
Stability: armor consistent, visor glow steady, face/spine intact. 
""",
    frame_offset=96*3,
    out_dir=out_dir,
    video_path="videos/output3.mp4"
)