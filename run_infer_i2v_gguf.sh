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
pip install lycoris-lora

cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
SD_SCRIPTS_DIR="$PARENT_DIR/sd-scripts"

# python wanpipeline_searching.py
# python wanpipeline.py
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

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export TORCH_NCCL_BLOCKING_WAIT=1

# accelerate launch --num_processes=2 --num_machines=1 sdxl_train_network.py \
#   --pretrained_model_name_or_path Bercraft/Illustrious-XL-v2.0-FP16-Diffusers \
#   --output_dir ../asset_generator/out_lora \
#   --output_name char_lora \
#   --save_model_as safetensors \
#   --mixed_precision bf16 \
#   --sdpa \
#   --train_data_dir ../asset_generator \
#   --caption_extension .txt \
#   --resolution 1024 \
#   --enable_bucket --min_bucket_reso 256 --max_bucket_reso 1024 \
#   --train_batch_size 2 \
#   --max_data_loader_n_workers 0 \
#   --max_train_steps 100 \
#   --network_module lycoris.kohya \
#   --network_args "algo=locon" "conv_dim=8" "conv_alpha=4" \
#   --network_dim 32 \
#   --network_alpha 16 \
#   --clip_skip 2 \
#   --cache_latents --vae_batch_size 1 \
#   --ddp_gradient_as_bucket_view \
#   --ddp_static_graph \
#   --sample_every_n_steps 100 \
#   --sample_prompts ../asset_generator/sample_prompts.json \
#   --sample_sampler "dpmsolver++"

python ../asset_generator/sdxl_train_from_json.py ../asset_generator/sdxl_train_conf.json


deactivate
cd "$PROJECT_DIR"
. .venv/bin/activate
python sdxl_infer_pose_lora.py
