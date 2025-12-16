#!/bin/bash
set -x
cd /home/fsq/gui_agent/GUI-R1-Evol-2/scripts/

SAVE_NAME=mind2web_gt_grpo_qwen2_5_vl_3b_105

# export TORCH_COMPILE_CACHE=/mnt/Shared_06_disk1/fsq/vllm_cache

# 遍历ckpt编号，从1到10为例
ckpt_numbers=(570)
for ckpt_num in "${ckpt_numbers[@]}"; do
    echo "Processing ckpt number: $ckpt_num"

    LOCAL_DIR=/mnt/Shared_05_disk/fsq/gui_agent_exp/gui-r1/$SAVE_NAME/global_step_$ckpt_num/actor
    LOCAL_HF_DIR=$LOCAL_DIR/huggingface
    OUTPUT_DIR=/home/fsq/gui_agent/GUI-R1-Evol-2/guir1/outputs

    # merge model if not exist
    if ! ls "$LOCAL_HF_DIR"/*.safetensors 1> /dev/null 2>&1; then
        echo "No .safetensors files found in $LOCAL_HF_DIR, running convert.py..."
        python model_merger.py --local_dir $LOCAL_DIR
    fi
    
    cd ..
    cd guir1

    # rm -rf ~/.cache/torch/inductor
    # rm -rf .torch_inductor

    python inference/inference_vllm_mind2web.py \
        --model_path $LOCAL_HF_DIR \
        --data_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_full.json \
        --image_dir /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/images \
        --output_name ${SAVE_NAME}_global_step_$ckpt_num \
        --num_actor 2
    python eval/eval_mind2web_point.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_full.json
    python eval/eval_mind2web_reformat.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_full.json
    cd ..
    cd scripts  
done

