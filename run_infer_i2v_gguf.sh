#!/usr/bin/env bash
set -e
git config --global user.email "danbugrienko@yandex.ru"
git config --global user.name "TOPAPEC"
# sudo apt update
# sudo apt install software-properties-common
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.11 python3.11-venv python3.11-dev
if [ ! -d .venv ]; then
  python -m venv .venv
fi
. .venv/bin/activate
export HF_HOME="/workspace/hf_cache/"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers/"
pip install -r requirements.txt
# pip install "git+https://github.com/facebookresearch/segment-anything-2"

cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
SD_SCRIPTS_DIR="$PARENT_DIR/sd-scripts"

# python wanpipeline.py
# python preprocess_images_for_lora.py

if [ ! -d "$SD_SCRIPTS_DIR" ]; then
  git clone https://github.com/kohya-ss/sd-scripts.git "$SD_SCRIPTS_DIR"
fi

cd "$SD_SCRIPTS_DIR"
if [ ! -d .venv ]; then
  python -m venv .venv
fi
. .venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt diffusers transformers accelerate safetensors torchvision
pip install lycoris-lora
# pip install -U "diffusers>=0.26.0" transformers accelerate safetensors huggingface_hub bitsandbytes==0.45.2
pip install "numpy<2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
accelerate launch --num_processes=1 sdxl_train_network.py \
  --pretrained_model_name_or_path Bercraft/Illustrious-XL-v2.0-FP16-Diffusers \
  --output_dir ../asset_generator/out_lora \
  --output_name char_lora \
  --save_model_as safetensors \
  --sdpa \
  --sample_every_n_steps 250 \
  --sample_prompts ../asset_generator/sample_prompts.json \
  --network_module lycoris.kohya \
  --network_args "algo=lokr" \
  --network_dim 32 \
  --network_alpha 16 \
  --train_data_dir ../asset_generator \
  --caption_extension .txt \
  --resolution 768 \
  --enable_bucket \
  --train_batch_size 2 \
  --max_data_loader_n_workers 0 \
  --max_train_steps 3000 \
  --no_half_vae \
  --clip_skip 2 \
  --cache_latents \



deactivate
cd "$PROJECT_DIR"
. .venv/bin/activate
python sdxl_infer_pose_lora.py
