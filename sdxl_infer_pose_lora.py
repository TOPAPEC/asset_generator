import os
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

BASE_MODEL = "Bercraft/Illustrious-XL-v2.0-FP16-Diffusers"
CONTROLNET_ID = "xinsir/controlnet-openpose-sdxl-1.0"
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "out_lora", "char_lora.safetensors"))
POSES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "poses"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "pose_lora"))
CHAR_DETAILS = str(open("character_details.txt", "r"))
print(f"CHARACTER DETS: {CHAR_DETAILS}")

PROMPT = f"Ohwjfdk, girl, short bob hair, pastel rainbow hair, blue eyes, pale skin, slim build"
NEGATIVE = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, (((poorly drawn face))), mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,"
GUIDANCE = 8.0
STEPS = 30
LORA_SCALE = 1.0
SEED = 42
TARGET_LONG = 1024

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.safetensors"
REF_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "image.png"))

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

controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    BASE_MODEL, controlnet=controlnet, torch_dtype=dtype, 
)
pipe.to("cuda")

pipe.load_lora_weights(LORA_PATH)
if hasattr(pipe, "fuse_lora"):
    pipe.fuse_lora(lora_scale=LORA_SCALE)

pipe.safety_checker = None

device_str = "cuda" if torch.cuda.is_available() else "cpu"
gens = [
    torch.Generator(device=device_str).manual_seed(SEED + i)
    for i in range(4)
]

ref_image = load_image(REF_IMAGE_PATH).convert("RGB")

valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
files = [f for f in sorted(os.listdir(POSES_DIR)) if os.path.splitext(f)[1].lower() in valid_exts]

for name in files:
    p = os.path.join(POSES_DIR, name)
    pose = Image.open(p).convert("RGB")
    w, h = pose.size

    s = TARGET_LONG / max(w, h)
    nw = int((w * s) // 8 * 8)
    nh = int((h * s) // 8 * 8)
    nw = max(nw, 64)
    nh = max(nh, 64)

    pose_resized = pose.resize((nw, nh), Image.BICUBIC)

    result = pipe(
        prompt=PROMPT + " " + name.split(".")[0].replace("_", " "),
        negative_prompt=NEGATIVE,
        image=pose_resized,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        controlnet_conditioning_scale=1.0,
        generator=gens,
        num_images_per_prompt=4,
        width=nw,
        height=nh
    )
    imgs = result.images

    grid = make_grid_2x2(imgs, nw, nh)

    out_p = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_gen_2x2.png")
    grid.save(out_p)