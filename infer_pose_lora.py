import os
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline

BASE_MODEL = "Lykon/dreamshaper-8"
CONTROLNET_ID = "lllyasviel/sd-controlnet-openpose"
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sd-scripts", "out", "char_lora.safetensors"))
POSES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "poses"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "pose_lora"))
PROMPT = "super_mecha_robotrigger, rating:general, 1girl, solo, red eyes, standing, full body, no humans, shadow, robot, mecha, science fiction, looking ahead, robot joints, open hands, humanoid robot"
NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, missing fingers, deformed, detailed background, multiple characters"
GUIDANCE = 7.0
STEPS = 20
LORA_SCALE = 0.8
SEED = 42
TARGET_LONG = 768

os.makedirs(OUT_DIR, exist_ok=True)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL, 
    controlnet=controlnet, 
    torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if torch.cuda.is_available():
    pipe.to("cuda")
pipe.load_lora_weights(LORA_PATH)
if hasattr(pipe, "fuse_lora"):
    pipe.fuse_lora(lora_scale=LORA_SCALE)
pipe.safety_checker = None
gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)

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
    img = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=pose_resized,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        controlnet_conditioning_scale=1.0,
        generator=gen,
        width=nw,
        height=nh
    ).images[0]
    out_p = os.path.join(OUT_DIR, os.path.splitext(name)[0] + "_gen.png")
    img.save(out_p)