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
pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=4)
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

fimg, limg = get_first_and_last_frame("raw_frames/")
out_dir = "raw_frames"

print("Starting generation")
frames = pipe(
    image=fimg,
    last_image=limg,
    prompt="Character is standing in front of ocean shoreline. The character runs forward and jumps in the middle of the video. Camera is smoothly moving after him and is able to capture him fully all the time. Each frame is an animation masterpiece",
    negative_prompt="",
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
).frames[0]

os.makedirs("./videos/", exist_ok=True)

export_to_video(frames, "videos/output1.mp4", fps=16)

last_im_number = 81 * 1

for i, im in enumerate(frames):
    to_pil(im).save(os.path.join(out_dir, f"frame_{last_im_number + 1 + i:03d}.png"))

print("Starting generation")
frames = pipe(
    image=fimg,
    last_image=limg,
    prompt="Character is standing in front of cyclorama on the tallest building in megapolis. The character sits down and begin crouching forward slowly. Camera is slowly moving towards the character and stops to move when he fills the frame fully",
    negative_prompt="",
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
).frames[0]

last_im_number = 81 * 2

for i, im in enumerate(frames):
    to_pil(im).save(os.path.join(out_dir, f"frame_{last_im_number + 1 + i:03d}.png"))

export_to_video(frames, "videos/output2.mp4", fps=16)


print("Starting generation")
frames = pipe(
    image=fimg,
    # last_image=limg,
    prompt="Character is standing in front of cyclorama at a boxing club. The character moves into a boxing stance and starts punching with his hands. Camera is focused on the character and slowly and smoothly moves towards and around it to capture his closes side view",
    negative_prompt="",
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
).frames[0]

last_im_number = 81 * 3

for i, im in enumerate(frames):
    to_pil(im).save(os.path.join(out_dir, f"frame_{last_im_number + 1 + i:03d}.png"))

export_to_video(frames, "videos/output3.mp4", fps=16)

print("Starting generation")
frames = pipe(
    image=fimg,
    prompt="Character is standing in front of cyclorama. The camera slowly and smoothly zooms in on the persona's face (no abrupt zoom), then in full-face so that the face, neck and chest are visible",
    negative_prompt="",
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
).frames[0]

last_im_number = 81 * 4

for i, im in enumerate(frames):
    to_pil(im).save(os.path.join(out_dir, f"frame_{last_im_number + 1 + i:03d}.png"))

export_to_video(frames, "videos/output4.mp4", fps=16)

print("Starting generation")
frames = pipe(
    image=fimg,
    prompt="Character stands in nature setting. The background is initially flashed by camera light, then it become clear and vibrant forest becomes visible. The camera slowly and smoothly zooms in on the persona's face, neck and chest are visible",
    negative_prompt="",
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=81,
    height=768,
    width=768,
).frames[0]

last_im_number = 81 * 5

for i, im in enumerate(frames):
    to_pil(im).save(os.path.join(out_dir, f"frame_{last_im_number + 1 + i:03d}.png"))

export_to_video(frames, "videos/output5.mp4", fps=16)