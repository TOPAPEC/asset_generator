#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, json, shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageFilter
import torch
import argparse

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

from ultralytics import YOLO  # pip install ultralytics

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

def isolate_with_yolo_and_replace_bg(
    image_paths: List[str],
    backgrounds_dir: str = "backgrounds",
    out_dir: str = "with_bg",
    yolo_weights: str = "yolov8n-seg.pt",   # n/m/l/x variants; custom .pt also works
    conf: float = 0.25,                     # lower if missed
    feather_px: int = 2,
    prefer_person_class: bool = True,       # prioritize person-like classes if present
    replace_original: bool = False
) -> List[str]:
    """
    For each image: run YOLOv8-Seg, pick best instance (size * centrality * conf),
    composite onto a random background, and save.
    """
    os.makedirs(out_dir, exist_ok=True)
    bg_paths = _collect_backgrounds(backgrounds_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(yolo_weights)  # downloads if needed

    outputs = []
    for p in image_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            W, H = im.size

        # Ultralytics API: returns list of Results; take first
        results = model.predict(source=p, imgsz=max(640, max(W, H)), conf=conf, verbose=False, device=0 if device=="cuda" else None)
        if not results:
            comp = im
        else:
            r = results[0]
            comp = im
            best_idx = None
            best_score = -1e9

            # If no masks, keep original
            if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
                # masks.data: [N, Hm, Wm] (downsampled), boxes.xyxy: [N,4], probs/conf: r.boxes.conf
                masks = r.masks.data.cpu().numpy()  # float [0..1]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((masks.shape[0], 4), dtype=np.float32)
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones((masks.shape[0],), dtype=np.float32)
                clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.cls is not None else np.zeros((masks.shape[0],), dtype=int)

                # Upsample masks to image size
                # r.masks.data is aligned to r.masks.orig_shape; Ultralytics also has r.masks.xy if polygon desired
                for i in range(masks.shape[0]):
                    m = masks[i]
                    # Resize mask to (H, W)
                    m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize((W, H), Image.BICUBIC)
                    area = float(np.array(m_img).sum()) / (255.0 * W * H + 1e-9)  # 0..1
                    if area < 0.02 or area > 0.95:
                        continue  # skip tiny or huge

                    box = boxes[i] if i < len(boxes) else np.array([0,0,W,H], dtype=np.float32)
                    central = _centrality_score(tuple(map(float, box)), W, H)
                    cls_bonus = 0.15 if (prefer_person_class and (clses[i] == 0)) else 0.0  # COCO class 0 = person
                    score = (math.log(area + 1e-6) * 0.8) + (central * 1.0) + (float(confs[i]) * 0.8) + cls_bonus
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_mask_img = m_img

                if best_idx is not None:
                    comp = _composite_on_random_bg(im, best_mask_img, bg_paths, feather_px=feather_px)

        if replace_original:
            out_path = p
        else:
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_dir, f"{base}.jpg")
        comp.save(out_path, quality=95, subsampling=1)
        outputs.append(out_path)
    return outputs


# =========================================================
#          WD‑EVA02‑Large v3 tagger + size handling
# =========================================================

def load_wd_eva02_v3(dev: Optional[str] = None):
    d = dev or device_str()
    dtype = torch.float16 if d == "cuda" else torch.float32
    proc = AutoImageProcessor.from_pretrained("p1atdev/wd-swinv2-tagger-v3-hf", trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained("p1atdev/wd-swinv2-tagger-v3-hf").to(d, dtype=dtype).eval()
    return model, proc, d, dtype


GENERIC_DROP = {
    "person","people","simple_background",
    "artist_name","copyright_name",
    "rating:safe","rating:questionable","rating:explicit",
    "photo"
}

from typing import List, Tuple, Dict
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

@torch.no_grad()
def wd_tags_for_images(
    image_paths: List[str],
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    device: str,
    torch_dtype: torch.dtype,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    max_tags: int = 64
) -> Dict[str, List[Tuple[str, float]]]:
    out: Dict[str, List[Tuple[str, float]]] = {}
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inputs = processor.preprocess(img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)
        probs = torch.sigmoid(model(**inputs).logits[0].float())
        pairs: List[Tuple[str, float]] = []
        for i, s in enumerate(probs):
            lab = model.config.id2label[i]
            if lab.startswith("rating:"): continue
            thr = character_threshold if lab.startswith("character:") else general_threshold
            sc = float(s.item())
            if sc >= thr:
                tag = lab.split(":",1)[1] if ":" in lab else lab
                tag = tag.strip().lower().replace(" ", "_")
                pairs.append((tag, sc))
        pairs.sort(key=lambda x: x[1], reverse=True)
        out[p] = pairs[:max_tags]
    return out

# =========================================================
#          GLM‑4.5v VLM tag refinement (multimodal)
# =========================================================

REFINEMENT_SYSTEM_PROMPT = (
    "You are a tag refiner for dataset labeling. You will receive (a) an image and (b) "
    "a comma-separated list of preliminary tags. Tasks: "
    "1) Remove generic, non-descriptive tags (e.g., solo, 1girl, people, simple_background, artist_name, rating:*). "
    "2) Keep concise, image-specific tags visible in the image (objects, clothing, colors, textures, actions, scene details, lighting). "
    "3) Enrich with specific visible details (fine-grained clothing names, materials, colors, scene elements, weather/lighting, camera angle if evident). "
    "Prefer lowercase with underscores; no medium words like painting/drawing/cartoon; output ONLY a comma-separated list.\n"
    "Enrichment examples:\n"
    "Base: woman, outdoors, jacket, car, street, night -> "
    "woman, red_leather_biker_jacket, rainy_street, night_city_lights, reflections_on_asphalt, parked_taxi, short_black_hair, bokeh\n"
    "Base: man, suit, office, laptop, window -> "
    "man, navy_suit, slim_black_tie, open_laptop, code_editor_screen, glass_office, skyline_reflection, side_lighting\n"
    "Base: cat, sitting, window, room -> "
    "tabby_cat, windowsill, warm_sunlight, dust_particles_in_light, lace_curtains, wooden_frame"
)

def load_glm_vlm(dev: Optional[str] = None):
    d = dev or device_str()
    dtype = torch.float16 if d == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        "z-ai/glm-4.5v",
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(d)
    proc = AutoProcessor.from_pretrained("z-ai/glm-4.5v", trust_remote_code=True)
    return model, proc, d, dtype

@torch.no_grad()
def refine_tags_with_glm(
    image_path: str,
    base_tags: List[Tuple[str, float]],
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    device: str,
    torch_dtype: torch.dtype,
    system_prompt: str = REFINEMENT_SYSTEM_PROMPT,
    max_new_tokens: int = 160
) -> str:
    base_list = [t for (t, _) in sorted(base_tags, key=lambda x: x[1], reverse=True)]
    base_csv = ", ".join(base_list)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Base tags: {base_csv}\nRefine these tags based on the attached image."}
    ]
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        inputs = processor(
            images=[im],
            text=processor.apply_chat_template(messages, add_generation_prompt=True),
            return_tensors="pt"
        ).to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    out_text = out_text.replace("\n", " ").replace("，", ",")
    if ":" in out_text and out_text.lower().startswith("refined"):
        out_text = out_text.split(":", 1)[1].strip()
    pieces = [_clean_tag(x) for x in out_text.split(",") if x.strip()]
    final, seen = [], set()
    for t in pieces:
        if not t or t in GENERIC_DROP: continue
        if t not in seen:
            seen.add(t); final.append(t)
    return ", ".join(final)

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

def main():
    ap = argparse.ArgumentParser(description="LoRA Preprocess: select diverse frames, cut with SAM2, replace backgrounds, tag with WD EVA02 v3, refine with GLM-4.5v.")
    ap.add_argument("--src_frames_dir", type=str, default="raw_frames")
    ap.add_argument("--sampled_dir", type=str, default="30_images")
    ap.add_argument("--with_bg_dir", type=str, default="with_bg")
    ap.add_argument("--backgrounds_dir", type=str, default="backgrounds")
    ap.add_argument("--keep_every", type=int, default=1)
    ap.add_argument("--phash_hamming", type=int, default=10)
    ap.add_argument("--target_k", type=int, default=20)
    ap.add_argument("--siglip_batch", type=int, default=16)
    ap.add_argument("--feather_px", type=int, default=2)
    ap.add_argument("--replace_original_bg", action="store_true")
    ap.add_argument("--sam2_ckpt", type=str, default="checkpoints/sam2_hiera_base_plus.pt")
    ap.add_argument("--sam1_type", type=str, default="vit_h")
    ap.add_argument("--sam1_ckpt", type=str, default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--wd_general_thr", type=float, default=0.35)
    ap.add_argument("--wd_character_thr", type=float, default=0.85)
    ap.add_argument("--max_tags", type=int, default=64)
    ap.add_argument("--trigger", type=str, default="M1N2N3Z4_K")
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
    wd_model, wd_proc, dev, dtype = load_wd_eva02_v3()

    base_tags_map = wd_tags_for_images(
        sampled, wd_model, wd_proc, dev, dtype,
        general_threshold=args.wd_general_thr,
        character_threshold=args.wd_character_thr,
        max_tags=args.max_tags
    )
    print("[WD] base tags generated")

    # 5) Save sidecar .txt files with trigger prefix
    save_tags(sampled, base_tags_map, args.trigger)
    print("[Save] tags saved")

if __name__ == "__main__":
    main()
