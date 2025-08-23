import os
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

BASE_MODEL = "stablediffusionapi/anything-v5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_openpose"
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "out_lora", "char_lora.safetensors"))
POSES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "poses"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "pose_lora"))
PROMPT = "Ohwjfdk, man, short black hair, blue eyes, muscular build, samurai armor, running, absurdres, masterpiece, illustration anime art, correct anatomy, simple background"
NEGATIVE = "worst quality, low quality, bad anatomy, bad hands, bad body, missing fingers, extra digit, three legs, three arms, fewer digits, blurry, text, watermark, lowres, bad anatomy, bad hands, extra fingers, missing fingers, deformed, detailed background, multiple characters"
GUIDANCE = 4.0
STEPS = 30
LORA_SCALE = 1.0
SEED = 42
TARGET_LONG = 512

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.safetensors"  # SD1.5 adapter
REF_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "image.png"))

workspace_dir = "/workspace/hf_cache"
os.environ["HF_HOME"] = "/workspace/hf_cache"

os.makedirs(OUT_DIR, exist_ok=True)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL, controlnet=controlnet, torch_dtype=dtype
)
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if torch.cuda.is_available():
    pipe.to("cuda")
pipe.load_lora_weights(LORA_PATH)
if hasattr(pipe, "fuse_lora"):
    pipe.fuse_lora(lora_scale=LORA_SCALE)

pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder="models",
    weight_name=IP_ADAPTER_WEIGHT,
)
# pipe.load_ip_adapter(
#     "h94/IP-Adapter-FaceID", 
#     subfolder="", 
#     weight_name="ip-adapter-faceid_sd15.bin"
# )

# pipe.load_ip_adapter(
#     "TheDenk/InstantID-SD1.5",
#     subfolder="",
#     weight_name="ip-adapter.bin",
    
# )

pipe.set_ip_adapter_scale(0.5)


pipe.safety_checker = None

gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)

ref_image = load_image(REF_IMAGE_PATH).convert("RGB")

ip_embeds = pipe.prepare_ip_adapter_image_embeds(
    ip_adapter_image=ref_image,
    ip_adapter_image_embeds=None,
    device=pipe.device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

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
        ip_adapter_image_embeds=ip_embeds,
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