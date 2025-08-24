CUDA_VISIBLE_DEVICES=0 python wan2.2_fewstep.py --config_path configs/inference/wan22.yaml \
    --checkpoint_folder wan_models/Wan2.2-TI2V-5B-Turbo \
    --seed $SEED \
    --prompt "Character is preparing to run in front of ocean shoreline. The character begin to run forward from the very begining of the video. The background at the starting moment flashed by camera light, but it becomes clear very fast. Camera is smoothly moving after him and is able to capture him fully all the time. Each frame is an animation masterpiece" \
    --image $IMAGE \
    --h 768 --w 768 \
    --output_path $OUTPATH