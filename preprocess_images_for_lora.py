#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, json, shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageFilter
import torch
import argparse
import asyncio
import aiohttp
import base64

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers import AutoProcessor, AutoModel

# =========================================================
#                 Utilities & Determinism
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
#                 pHash (DCT-II) + Hamming
# =========================================================

def _dct_1d(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape(-1, 1)
    mat = np.cos(np.pi * (n + 0.5) * k / N)
    return mat @ x

def _dct_2d(a: np.ndarray) -> np.ndarray:
    return _dct_1d(_dct_1d(a.T).T)

def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    size = hash_size * highfreq_factor
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    dct = _dct_2d(a)
    dct_low = dct[:hash_size, :hash_size]
    flat = dct_low.flatten()
    med = np.median(flat[1:]) if flat.size > 1 else flat[0]
    bits = 0
    for i, v in enumerate(dct_low.flatten()):
        if v > med:
            bits |= (1 << i)
    return bits

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

@torch.no_grad()
def filter_by_pickscore(image_paths: List[str], keep_ratio: float = 0.8, batch_size: int = 8) -> List[str]:
    if len(image_paths) <= 1:
        return image_paths
    
    device = device_str()
    processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
    
    scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        
        inputs = processor(
            images=images,
            text=[""] * len(images),
            return_tensors="pt",
            padding=True
        )
        
        image_inputs = {k: v.to(device) for k, v in inputs.items() if k in ['pixel_values']}
        text_inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        
        image_embs = model.get_image_features(**image_inputs)
        text_embs = model.get_text_features(**text_inputs)
        
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        
        logits_per_image = model.logit_scale.exp() * (image_embs @ text_embs.t())
        probs = torch.softmax(logits_per_image, dim=-1)
        batch_scores = probs.diag().cpu().tolist()
        scores.extend(batch_scores)
    
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    num_keep = max(1, int(len(image_paths) * keep_ratio))
    indexed_scores = [(score, i, path) for i, (score, path) in enumerate(zip(scores, image_paths))]
    indexed_scores.sort(reverse=True)
    
    return [path for _, _, path in indexed_scores[:num_keep]]

@torch.no_grad()
def filter_by_siglip_negative_prompt(
    image_paths: List[str], 
    negative_prompt: str = "blurry face, distorted face, bad quality image, bad anatomy, deformed, ugly, low resolution, pixelated, artifacts",
    keep_ratio: float = 0.7,
    batch_size: int = 16
) -> List[str]:
    if len(image_paths) <= 1:
        return image_paths
    
    model, proc, dev = _siglip2_model_device()
    
    text_inputs = proc(text=[negative_prompt], return_tensors="pt").to(dev)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)
    
    image_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as img:
                batch_images.append(img.convert("RGB"))
        
        inputs = proc(images=batch_images, return_tensors="pt").to(dev)
        img_emb = model.get_image_features(**inputs)
        img_emb = torch.nn.functional.normalize(img_emb.float(), dim=-1)
        image_embs.append(img_emb.cpu())
    
    all_image_embs = torch.cat(image_embs, dim=0)
    similarities = (all_image_embs @ text_emb.cpu().T).squeeze(1)
    
    num_keep = max(1, int(len(image_paths) * keep_ratio))
    indexed_sims = [(sim.item(), i, path) for i, (sim, path) in enumerate(zip(similarities, image_paths))]
    indexed_sims.sort()
    
    return [path for _, _, path in indexed_sims[:num_keep]]

# =========================================================
#      Stage 2: SigLIP2 semantic diversity (FPS-k)
# =========================================================

def _siglip2_model_device():
    dev = device_str()
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").eval().to(dev)
    proc  = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    return model, proc, dev

@torch.no_grad()
def _siglip2_embeddings(paths: List[str], batch_size: int = 16) -> torch.Tensor:
    model, proc, dev = _siglip2_model_device()
    embs = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            with Image.open(p) as im:
                batch.append(im.convert("RGB"))
        inputs = proc(images=batch, return_tensors="pt").to(dev)
        e = model.get_image_features(**inputs)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0, 1)

def semantic_select_k_fps(paths: List[str], k: int = 20, batch_size: int = 16) -> List[str]:
    n = len(paths)
    if n <= k:
        return paths
    E = _siglip2_embeddings(paths, batch_size=batch_size)  # [N,D], L2 normed
    centroid = torch.nn.functional.normalize(E.mean(dim=0, keepdim=True), dim=-1)
    sim_to_centroid = (E @ centroid.T).squeeze(1)
    first = int(torch.argmin(sim_to_centroid).item())
    selected = [first]
    min_dists = 1.0 - (E @ E[first].unsqueeze(1)).squeeze(1)
    min_dists[first] = -1.0
    while len(selected) < k:
        nxt = int(torch.argmax(min_dists).item())
        selected.append(nxt)
        d_new = 1.0 - (E @ E[nxt].unsqueeze(1)).squeeze(1)
        min_dists = torch.minimum(min_dists, d_new)
        min_dists[nxt] = -1.0
    selected = sorted(set(selected))
    return [paths[i] for i in selected]

# =========================================================
#        Combined two‑phase selection into dst_dir
# =========================================================

def select_frames_two_phase(
    src_dir: str,
    dst_dir: str,
    keep_every: int = 1,
    phash_hamming_thresh: int = 10,
    target_k: int = 20,
    siglip_batch: int = 16,
    pickscore_ratio: float = 0.8,
) -> List[str]:
    os.makedirs(dst_dir, exist_ok=True)
    # stride pre-sampling
    candidates: List[Tuple[int,str]] = []
    i = -1
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith((".png",".jpg",".jpeg",".webp")):
            continue
        i += 1
        if i % keep_every != 0:
            continue
        candidates.append((i, os.path.join(src_dir, name)))
    if not candidates:
        return []
    
    candidate_paths = [sp for idx, sp in candidates]
    
    # PickScore filtering (top 80%)
    filtered_paths = filter_by_pickscore(candidate_paths, keep_ratio=pickscore_ratio)
    # SigLIP negative prompt filtering (keep best 30%)
    filtered_candidates = [(idx, sp) for idx, sp in candidates if sp in set(filtered_paths)]


    # Stage 1: pHash near-dup prune while copying
    stage1_paths, stage1_hashes = [], []
    for idx, sp in filtered_candidates:
        with Image.open(sp) as img:
            hh = phash(img)
            if all(hamming(hh, prev) >= phash_hamming_thresh for prev in stage1_hashes):
                out = os.path.join(dst_dir, f"img_{idx:03d}.png")
                img.convert("RGB").save(out)
                stage1_paths.append(out)
                stage1_hashes.append(hh)
    if not stage1_paths:
        return []

    # Stage 2: FPS-k to exact target_k
    keep = semantic_select_k_fps(stage1_paths, k=target_k, batch_size=siglip_batch)

    # delete dropped
    keep_set = set(os.path.abspath(p) for p in keep)
    for p in stage1_paths:
        if os.path.abspath(p) not in keep_set:
            try: os.remove(p)
            except OSError: pass
            base, _ = os.path.splitext(p)
            txtp = base + ".txt"
            if os.path.isfile(txtp):
                try: os.remove(txtp)
                except OSError: pass
    return keep

# --- YOLOv8-Seg character cut-out + random background compositing ---
import os, random, math, numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageFilter
import torch


def _gaussian_feather(mask: Image.Image, radius: int = 2) -> Image.Image:
    return mask if radius <= 0 else mask.filter(ImageFilter.GaussianBlur(radius))

def _centrality_score(box: Tuple[float,float,float,float], W: int, H: int) -> float:
    x0, y0, x1, y1 = box
    cx = 0.5*(x0+x1); cy = 0.5*(y0+y1)
    dx = (cx - W/2) / (W/2); dy = (cy - H/2) / (H/2)
    return 1.0 - min(1.0, (dx*dx + dy*dy)**0.5)

def _collect_backgrounds(backgrounds_dir: str) -> List[str]:
    exts = (".png",".jpg",".jpeg",".webp")
    return [os.path.join(backgrounds_dir,f) for f in os.listdir(backgrounds_dir)
            if f.lower().endswith(exts)]

def _composite_on_random_bg(img: Image.Image, mask_L: Image.Image, bg_paths: List[str], feather_px: int = 2) -> Image.Image:
    if not bg_paths:
        raise RuntimeError("No backgrounds found in backgrounds/ directory")
    bg_path = random.choice(bg_paths)
    with Image.open(bg_path) as bg:
        bg = bg.convert("RGB").resize(img.size, Image.BICUBIC)
    mask_L = _gaussian_feather(mask_L, radius=feather_px)
    fg = img.convert("RGBA")
    fg.putalpha(mask_L)
    out = Image.alpha_composite(bg.convert("RGBA"), fg)
    return out.convert("RGB")

@torch.no_grad()
def find_closest_images_to_text(image_paths: List[str], text_query: str, top_k: int = 3, batch_size: int = 16) -> List[str]:
    if len(image_paths) <= top_k:
        return image_paths
    
    model, proc, dev = _siglip2_model_device()
    
    text_inputs = proc(text=[text_query], return_tensors="pt").to(dev)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)
    
    image_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as img:
                batch_images.append(img.convert("RGB"))
        
        inputs = proc(images=batch_images, return_tensors="pt").to(dev)
        img_emb = model.get_image_features(**inputs)
        img_emb = torch.nn.functional.normalize(img_emb.float(), dim=-1)
        image_embs.append(img_emb.cpu())
    
    all_image_embs = torch.cat(image_embs, dim=0)
    similarities = (all_image_embs @ text_emb.cpu().T).squeeze(1)
    
    indexed_sims = [(sim.item(), i, path) for i, (sim, path) in enumerate(zip(similarities, image_paths))]
    indexed_sims.sort(reverse=True)
    
    return [path for _, _, path in indexed_sims[:top_k]]

async def generate_character_traits(
    closeup_paths: List[str],
    fullbody_paths: List[str], 
    api_key: str,
    model: str = "baidu/ernie-4.5-vl-28b-a3b"
) -> str:
    
    async with aiohttp.ClientSession() as session:
        images_data = []
        for path in closeup_paths + fullbody_paths:
            with open(path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode()
            images_data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        
        print(f"Selected images for character analysis:")
        for path in closeup_paths:
            print(f"  Closeup: {path}")
        for path in fullbody_paths:
            print(f"  Fullbody: {path}")
        
        prompt = """Analyze these 6 images of the same character and identify the most consistent, defining traits that should always be present in a LoRA training dataset.

Focus on STABLE IDENTITY TRAITS that never change:
- Hair color and style (e.g., "long blonde hair", "short black hair with bangs")
- Eye color (e.g., "blue eyes", "brown eyes")
- Permanent body features (scars, tattoos, body type, skin tone)
- Permanent modifications (mechanical limbs, prosthetics, etc.)
- Basic character class (girl, woman, man, boy, character)

DO NOT include:
- Clothing/accessories that vary between images
- Expressions or poses
- Backgrounds or scenes
- Temporary items

Return ONLY a comma-separated list of the core character traits, make it as consice as possible like:
"girl, long blonde hair, blue eyes, pale skin, slim build"

Character traits:"""

        content = [{"type": "text", "text": prompt}] + images_data
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                              json=payload, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API error: {resp.status}")
                return "character"

async def generate_variable_details(
    image_path: str,
    api_key: str,
    model: str = "baidu/ernie-4.5-vl-28b-a3b"
) -> str:
    
    async with aiohttp.ClientSession() as session:
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        img_b64 = base64.b64encode(img_data).decode()
        
        prompt = """Describe the VARIABLE details in this image that can change between training images:

Include only:
- Clothing/outfit details (but only if clearly visible and significant)
- Accessories (glasses, hats, jewelry - mark as optional)
- Background/scene (indoors, outdoors, specific location)
- Notable pose or expression (if very distinctive)

Keep it as concise as possible to fit into limited clip model context window. Use natural language. If nothing distinctive is visible, return empty string.
Return one list of tags separated by commas, be as consice as possible. Like:
"heavy blue plated samurai armor, katana, blurry background, ready to fight, fists clenched, leaning"
Variable details:"""

        payload = {
            "model": model,
            "messages": [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                              json=payload, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return ""

async def generate_prompts_two_stage(
    image_paths: List[str],
    trigger_word: str,
    api_key: str,
    model: str = "google/gemini-2.5-flash"
) -> Dict[str, str]:
    
    closeup_paths = find_closest_images_to_text(image_paths, "high quality close up shot", top_k=3)
    fullbody_paths = find_closest_images_to_text(image_paths, "fullbody high quality shot", top_k=3)
    
    character_traits = await generate_character_traits(closeup_paths, fullbody_paths, api_key, model)

    with open("character_details.txt", "w") as file: 
        file.write(character_traits)
    
    semaphore = asyncio.Semaphore(8)
    
    async def process_image(image_path: str) -> Tuple[str, str]:
        async with semaphore:
            variable_details = await generate_variable_details(image_path, api_key, model)
            
            prompt_parts = [trigger_word, character_traits]
            if variable_details.strip():
                prompt_parts.append(variable_details)
            
            final_prompt = ", ".join(part.strip() for part in prompt_parts if part.strip())
            return image_path, final_prompt
    
    tasks = [process_image(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    
    return dict(results)


def save_refined_tags(images: List[str], refined_tags_map: Dict[str, str], trigger: str) -> None:
    for p in images:
        base = os.path.splitext(os.path.basename(p))[0]
        txtp = os.path.join(os.path.dirname(p), base + ".txt")
        tags = refined_tags_map.get(p, "")
        if trigger and not tags.startswith(trigger):
            tags = f"{trigger}, {tags}" if tags else trigger
        with open(txtp, "w", encoding="utf-8") as f:
            f.write(tags)

# =========================================================
#                         IO
# =========================================================

def save_tags(images: List[str], tags_map: Dict[str, str], trigger: str) -> None:
    for p in images:
        base = os.path.splitext(os.path.basename(p))[0]
        txtp = os.path.join(os.path.dirname(p), base + ".txt")
        tags = str(list(dict(tags_map[p]).keys())).strip("[]").replace('\'', "")
        if trigger and not tags.startswith(trigger):
            tags = f"{trigger}, {tags}" if tags else trigger
        with open(txtp, "w", encoding="utf-8") as f:
            f.write(tags)

# =========================================================
#                         MAIN
# =========================================================

async def main():
    ap = argparse.ArgumentParser(description="LoRA Preprocess: select diverse frames, cut with SAM2, replace backgrounds, tag with WD EVA02 v3, refine with GLM-4.5v.")
    ap.add_argument("--src_frames_dir", type=str, default="raw_frames")
    ap.add_argument("--sampled_dir", type=str, default="30_images")
    ap.add_argument("--with_bg_dir", type=str, default="with_bg")
    ap.add_argument("--backgrounds_dir", type=str, default="backgrounds")
    ap.add_argument("--keep_every", type=int, default=1)
    ap.add_argument("--phash_hamming", type=int, default=10)
    ap.add_argument("--target_k", type=int, default=60)
    ap.add_argument("--siglip_batch", type=int, default=16)
    ap.add_argument("--feather_px", type=int, default=2)
    ap.add_argument("--replace_original_bg", action="store_true")
    ap.add_argument("--sam2_ckpt", type=str, default="checkpoints/sam2_hiera_base_plus.pt")
    ap.add_argument("--sam1_type", type=str, default="vit_h")
    ap.add_argument("--sam1_ckpt", type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--wd_general_thr", type=float, default=0.35)
    ap.add_argument("--wd_character_thr", type=float, default=0.85)
    ap.add_argument("--max_tags", type=int, default=64)
    ap.add_argument("--trigger", type=str, default="Ohwjfdk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # 0) Clean sample dir
    shutil.rmtree(args.sampled_dir, ignore_errors=True)
    os.makedirs(args.sampled_dir, exist_ok=True)

    # 1) Two‑phase selection to exactly k images
    sampled = select_frames_two_phase(
        src_dir=args.src_frames_dir,
        dst_dir=args.sampled_dir,
        keep_every=args.keep_every,
        phash_hamming_thresh=args.phash_hamming,
        target_k=args.target_k,
        siglip_batch=args.siglip_batch
    )
    if not sampled:
        print("No sampled images produced.")
        return
    print(f"[Select] kept {len(sampled)} images")


    # 3) WD EVA02 v3 base tags (auto image size)
    
    prompts_map = await generate_prompts_two_stage(
        sampled,
        trigger_word=args.trigger,
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    
    for image_path, prompt in prompts_map.items():
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)


if __name__ == "__main__":
    asyncio.run(main())
