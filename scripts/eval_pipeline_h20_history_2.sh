#!/bin/bash
set -x
cd /apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/scripts/

EXP_DIR=/root/datasets/fsq/gui_r1_exp
DATA_DIR=/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b
SAVE_NAME=mind2web_phase3_from_mind2web_phase1_r1gui_org_grpo_qwen2_5_vl_3b_h20
OUTPUT_DIR=/apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/guir1/outputs

export TORCH_COMPILE_CACHE=/root/datasets/fsq/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/root/datasets/fsq/torchinductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR
chmod -R 777 $TORCHINDUCTOR_CACHE_DIR

export ray_init_num_cpus=32
export CUDA_VISIBLE_DEVICES=2,3

# 遍历ckpt编号，从1到10为例
ckpt_numbers=(650)
for ckpt_num in "${ckpt_numbers[@]}"; do
    echo "Processing ckpt number: $ckpt_num"
   
    LOCAL_DIR=${EXP_DIR}/${SAVE_NAME}/global_step_$ckpt_num/actor
    LOCAL_HF_DIR=$LOCAL_DIR/huggingface

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
        --data_path ${DATA_DIR}/metadata/hf_test_full.json \
        --image_dir ${DATA_DIR}/images \
        --output_name ${SAVE_NAME}_global_step_$ckpt_num \
        --num_actor 2 \
        --history_num 4 \
        --use_history
    python eval/eval_mind2web_point.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path ${DATA_DIR}/metadata/hf_test_full.json
    python eval/eval_mind2web_reformat.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path ${DATA_DIR}/metadata/hf_test_full.json
    cd ..
    cd scripts   
done

