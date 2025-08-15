#!/usr/bin/env bash
set -e
git config --global user.email "danbugrienko@yandex.ru"
git config --global user.name "TOPAPEC"
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
python -m venv .venv
. .venv/bin/activate
export HF_HOME="./hf_cache/"
export TRANSFORMERS_CACHE="./hf_cache/transformers/"
pip install -r requirements.txt

cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
SD_SCRIPTS_DIR="$PARENT_DIR/sd-scripts"

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python infer_i2v_gguf.py
# python preprocess_images_for_lora.py

if [ ! -d "$SD_SCRIPTS_DIR" ]; then
  git clone https://github.com/kohya-ss/sd-scripts.git "$SD_SCRIPTS_DIR"
fi

cd "$SD_SCRIPTS_DIR"
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt diffusers transformers accelerate safetensors torchvision
pip install "numpy<2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
accelerate launch --num_processes=1 train_network.py \
  --pretrained_model_name_or_path Lykon/dreamshaper-8 \
  --output_dir ./out \
  --output_name char_lora \
  --save_model_as safetensors \
  --mixed_precision fp16 \
  --sdpa \
  --gradient_checkpointing \
  --network_module networks.lora \
  --network_dim 32 \
  --network_alpha 16 \
  --train_data_dir ../asset_generator \
  --caption_extension .txt \
  --resolution 512 \
  --enable_bucket \
  --train_batch_size 2 \
  --max_data_loader_n_workers 0 \
  --max_train_steps 3000

deactivate
cd "$PROJECT_DIR"
. .venv/bin/activate
python infer_pose_lora.py