import os, numpy as np
from PIL import Image
def dhash(img, hash_size=8):
    g = img.convert("L").resize((hash_size+1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)
    d = a[:, 1:] > a[:, :-1]
    bits = 0
    idx = 0
    for v in d.flatten():
        if v:
            bits |= 1 << idx
        idx += 1
    return bits
def hamming(a, b):
    return (a ^ b).bit_count()
src = "raw_frames"
dst = "30_images"
os.makedirs(dst, exist_ok=True)
prev_hash = None
keep_every = 2
for i, name in enumerate(sorted(os.listdir(src))):
    if not name.lower().endswith((".png",".jpg",".jpeg",".webp")):
        continue
    p = os.path.join(src, name)
    img = Image.open(p)
    if i % keep_every != 0:
        continue
    h = dhash(img)
    if prev_hash is None or hamming(h, prev_hash) >= 6:
        img.convert("RGB").save(os.path.join(dst, f"img_{i:03d}.png"))
        prev_hash = h

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch, os

model_id = "p1atdev/wd-swinv2-tagger-v3-hf"
processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(model_id)

images_dir = dst
thresh = 0.35

for name in sorted(os.listdir(images_dir)):
    if not name.lower().endswith((".png",".jpg",".jpeg",".webp")):
        continue
    p = os.path.join(images_dir, name)
    img = Image.open(p).convert("RGB")
    inputs = processor.preprocess(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device, model.dtype) for k, v in inputs.items()})
        probs = torch.sigmoid(outputs.logits[0])
    labels = [model.config.id2label[i] for i, s in enumerate(probs) if float(s) > thresh]
    with open(os.path.join(images_dir, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(labels))


import os
root = images_dir
trigger = "super_mecha_robotrigger"
for name in os.listdir(root):
    if not name.endswith(".txt"):
        continue
    p = os.path.join(root, name)
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt.startswith(trigger):
        txt = f"{trigger}, {txt}"
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
