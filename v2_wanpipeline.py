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

print("Loading models")
t_hi = WanTransformer3DModel.from_single_file(
    "https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/main/model.pt",
    config="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    subfolder="transformer",
    low_cpu_mem_usage=False
)


max_memory={0: "48GB"}
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers", 
    torch_dtype=torch.bfloat16, 
    # cache_dir=workspace_dir, 
    transformer=t_hi,
    max_memory=max_memory,
)

cfg = dict(pipe.scheduler.config)
cfg.update({
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "flow_shift": 5.0,
    "solver_type": "bh2",
    "solver_order": 2,
    "final_sigmas_type": "zero",
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "lower_order_final": True,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0,
})
pipe.scheduler = UniPCMultistepScheduler.from_config(cfg)


offload_device = "cpu"
onload_device = "cuda"
# pipe.vae.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# pipe.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# pipe.transformer_2.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
# apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=4)

pipe.to("cuda")

# fimg, limg = get_first_and_last_frame("raw_frames/")

fimg = load_image(Image.open("image.png")).convert("RGB")
out_dir = "raw_frames"

os.makedirs(out_dir, exist_ok=True)

def generate_sequence(
    pipe,
    image,
    prompt,
    negative_prompt="",
    num_inference_steps=4,
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
    prompt="Character is preparing to run in front of ocean shoreline. The character begin to run forward from the very begining of the video. The background at the starting moment flashed by camera light, but it becomes clear very fast. Camera is smoothly moving after him and is able to capture him fully all the time. Each frame is an animation masterpiece",
    negative_prompt="motion blur, blurry eyes, distorted face, distorted character features",
    frame_offset=81*1,
    out_dir=out_dir,
    video_path="videos/output1.mp4"
)

# generate_sequence(
#     pipe=pipe,
#     image=fimg,
#     prompt="The character is standing in front brick wall in a small town. The character begins to run to the left. The background at the starting moment flashed by camera light, but it becomes clear very fast and brick wall with the town become visible. Camera is slowly moving with the character capturing its sideview",
#     frame_offset=81*2,
#     out_dir=out_dir,
#     video_path="videos/output2.mp4"
# )

# generate_sequence(
#     pipe=pipe,
#     image=fimg,
#     prompt="Character is standing in front of cyclorama at a boxing club. The character moves into a boxing stance and starts punching with his hands. Camera is focused on the character and slowly and smoothly moves towards and around it to capture his closes side view",
#     frame_offset=81*3,
#     out_dir=out_dir,
#     video_path="videos/output3.mp4"
# )


# generate_sequence(
#     pipe=pipe,
#     image=fimg,
#     prompt="The character is standing in front brick wall in a small town. The camera slowly and smoothly zooms in on the persona's face (no abrupt zoom), then in full-face so that the face, neck and chest are visible",
#     frame_offset=81*4,
#     out_dir=out_dir,
#     video_path="videos/output4.mp4"
# )


# generate_sequence(
#     pipe=pipe,
#     image=fimg,
#     prompt="Character stands in nature setting. It is talking and laughing and then becomes angry. The background is initially flashed by camera light, then it become clear and vibrant forest becomes visible. The camera slowly and smoothly zooms in on the persona's face from a side, neck and shoulder are visible and the camera stops on the character side view",
#     frame_offset=81*5,
#     out_dir=out_dir,
#     video_path="videos/output5.mp4"
# )


# generate_sequence(
#     pipe=pipe,
#     image=fimg,
#     prompt="Character stands in town setting. Character begins to turn around from the start of the video. The background is initially flashed by camera light, then it become clear and vibrant and cozy town becomes visible. The camera slowly and smoothly zooms in on the persona's face (capturing back of his head with neck and shoulders with upper back at the end)",
#     frame_offset=81*6,
#     out_dir=out_dir,
#     video_path="videos/output6.mp4"
# )


generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Character is training in town setting. The character drops any items from his hands, jumps into fighting stance and begins to land professional roundhouse kicks. The background is initially flashed by camera light, then it become clear and vibrant and cozy town becomes visible. The camera slowly and smoothly goes around the character capturing first frontview of it fighting then sideview and backview. If some background objects are constricting camera movements, camera uses noclip to fly through them.",
    frame_offset=81*7,
    out_dir=out_dir,
    video_path="videos/output7.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Character stands in town setting. Character takes a chair and sits on it. The background is initially flashed by camera light, then it become clear and vibrant and cozy town becomes visible. The camera slowly and smoothly goes around the character capturing first frontview of it sitting and then sideview and then back view always capturing the character full height. If some background objects are constricting camera movements, camera uses noclip to fly through them",
    frame_offset=81*8,
    out_dir=out_dir,
    video_path="videos/output8.mp4"
)

generate_sequence(
    pipe=pipe,
    image=fimg,
    prompt="Dynamic 3d anime style animation. Character is training in nature near a very high (around 50 meters tall) a waterfall cliff. The character drops any items from his hands and jumps off the cliff and free fall then lends right into the water near the waterfall. The background is initially flashed by camera light, then it become clear and vibrant and nature setting with waterfall appear clear and beautiful. The camera is ALWAYS keeping character in frame following the character even while it falling from the cliff and even when it dives underwater.",
    frame_offset=81*9,
    out_dir=out_dir,
    video_path="videos/output9.mp4"
)
