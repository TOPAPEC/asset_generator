import os, sys, torch, imageio
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    GGUFQuantizationConfig,
    LCMScheduler,
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


workspace_dir = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_cache"

lora_path = hf_hub_download(
    repo_id="Kijai/WanVideo_comfy",
    filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    cache_dir=workspace_dir,
)

print("Loading models")
t_hi = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    subfolder="transformer",
    cache_dir=workspace_dir
)

t_lo = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/blob/main/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    subfolder="transformer",
    cache_dir=workspace_dir,
)
print("Loaded models")


max_memory={0: "24GB"}
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers", 
    torch_dtype=torch.bfloat16, 
    cache_dir=workspace_dir, 
    transformer=t_hi,
    transformer_2=t_lo,
    max_memory=max_memory,
)

offload_device = "cpu"
onload_device = "cuda"
pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=2)
pipe.load_lora_weights(lora_path)
org_state_dict = safetensors.torch.load_file(lora_path)
converted_state_dict = _convert_non_diffusers_wan_lora_to_diffusers(org_state_dict)
pipe.transformer_2.load_lora_adapter(converted_state_dict)

pipe.scheduler.config["prediction_type"] = "epsilon"
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

img = load_image(Image.open("image.jpg")).convert("RGB")
prompt = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "A neon-lit cyber-samurai walking through rainy Tokyo at night"
)
print("Starting generation")
frames = pipe(
    image=img,
    prompt=prompt,
    num_inference_steps=6,
    guidance_scale=1.0,
    num_frames=17,
    height=720,
    width=1280,
).frames[0]

export_to_video(frames, "output.mp4", fps=16)
