import os
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UNet2DConditionModel, StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

BASE_MODEL = "Bercraft/Illustrious-XL-v2.0-FP16-Diffusers"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "pose_lora"))

PROMPT = """vast deep space filled with countless twinkling stars and swirling galaxies, in the distance a strange and mysterious floating monster, resembling a zerg or nekromorph creature, haunting silhouette, alien horror design, twisted organic forms, tendrils floating in void, cosmic horror atmosphere, dark sci-fi fantasy art, cinematic space illustration, glowing nebula colors, ominous mood, digital painting, concept art, artstation, pixiv"""
NEGATIVE = """photorealistic, 3d render, lowres, blurry, distorted, extra limbs, bad anatomy, cartoonish, flat shading, text, signature, watermark, jpeg artifacts, noisy background, bright cheerful style, chibi, low contrast"""
GUIDANCE = 8.0
STEPS = 30
LORA_SCALE = 1.0
SEED = 42
TARGET_LONG = 1024


workspace_dir = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_cache"

os.makedirs(OUT_DIR, exist_ok=True)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def make_grid_2x2(images, tile_w, tile_h):
    if len(images) != 4:
        raise ValueError("make_grid_2x2 expects exactly 4 images.")
    fixed = []
    for im in images:
        if im.size != (tile_w, tile_h):
            im = im.resize((tile_w, tile_h), Image.BICUBIC)
        fixed.append(im)

    grid = Image.new("RGB", (tile_w * 2, tile_h * 2))
    grid.paste(fixed[0], (0, 0))
    grid.paste(fixed[1], (tile_w, 0))
    grid.paste(fixed[2], (0, tile_h))
    grid.paste(fixed[3], (tile_w, tile_h))
    return grid

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=dtype, 
)
pipe.to("cuda")


device_str = "cuda" if torch.cuda.is_available() else "cpu"
gens = [
    torch.Generator(device=device_str).manual_seed(SEED + i)
    for i in range(4)
]


valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}



result = pipe(
    prompt=PROMPT, 
    negative_prompt=NEGATIVE,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    generator=gens,
    num_images_per_prompt=4,
    width=TARGET_LONG,
    height=TARGET_LONG
)
imgs = result.images

grid = make_grid_2x2(imgs, TARGET_LONG, TARGET_LONG)

out_p = os.path.join(OUT_DIR, "_gen_2x2.png")
grid.save(out_p)