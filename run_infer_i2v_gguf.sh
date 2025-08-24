#!/usr/bin/env bash
set -e
git config --global user.email "danbugrienko@yandex.ru"
git config --global user.name "TOPAPEC"
sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.10 python3.10-venv python3.10-dev
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

cd ..
git clone https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo.git
cd Wan2.2-TI2V-5B-Turbo/
python3.10 -m venv .venv
. .venv/bin/activate
pip install wheel setuptools
pip install pandas decord
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop[]
# python wanpipeline.py
# python v2_wanpipeline.py
python preprocess_images_for_lora.py

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
pip install "numpy<2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
accelerate launch --num_processes=1 train_network.py \
  --pretrained_model_name_or_path stablediffusionapi/anything-v5 \
  --output_dir ../asset_generator/out_lora \
  --output_name char_lora \
  --save_model_as safetensors \
  --mixed_precision fp16 \
  --sdpa \
  --sample_every_n_steps 250 \
  --sample_prompts ../asset_generator/sample_prompts.json \
  --network_module lycoris.kohya \
  --network_args "algo=locon" "conv_dim=8" "conv_alpha=4" \
  --network_dim 32 \
  --network_alpha 16 \
  --train_data_dir ../asset_generator \
  --caption_extension .txt \
  --resolution 512 \
  --enable_bucket \
  --train_batch_size 2 \
  --max_data_loader_n_workers 0 \
  --max_train_steps 5000 \
  --sample_sampler "dpmsolver++" \
  --sample_steps 32 \
  --sample_width 512 --sample_height 512 \
  --sample_cfg_scale 7.0 \
  --sample_n_samples 3 --sample_n_rows 1

deactivate
cd "$PROJECT_DIR"
. .venv/bin/activate
python infer_pose_lora.py