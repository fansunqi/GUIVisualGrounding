#!/bin/bash
set -x
cd /apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/scripts/

EXP_DIR=/root/datasets/fsq/gui_r1_exp
DATA_DIR=/root/cache/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52
SAVE_NAME=combine_phase3_from_mind2web_phase1_r1gui_org_grpo_qwen2_5_vl_3b_h20
OUTPUT_DIR=/apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/guir1/outputs

export TORCH_COMPILE_CACHE=/root/datasets/fsq/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/root/datasets/fsq/torchinductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR
chmod -R 777 $TORCHINDUCTOR_CACHE_DIR

export ray_init_num_cpus=32
export CUDA_VISIBLE_DEVICES=2,3

# 遍历ckpt编号，从1到10为例
ckpt_numbers=(475 500 525 550 575)
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

    python inference/inference_vllm_screenspot.py \
        --model_path $LOCAL_HF_DIR \
        --data_path ${DATA_DIR}/screenspot_test.parquet \
        --output_name ${SAVE_NAME}_global_step_$ckpt_num \
        --num_actor 2
    python eval/eval_screenspot.py \
        --model_id ${SAVE_NAME}_global_step_$ckpt_num  \
        --prediction_file_path ${OUTPUT_DIR}/${SAVE_NAME}_global_step_${ckpt_num}/screenspot_test.json
    # python inference/inference_vllm_screenspot.py \
    #     --model_path $LOCAL_HF_DIR \
    #     --data_path ${DATA_DIR}/screenspot_pro_test.parquet \
    #     --output_name ${SAVE_NAME}_global_step_$ckpt_num \
    #     --num_actor 2
    # python eval/eval_screenspot.py \
    #     --model_id ${SAVE_NAME}_global_step_$ckpt_num  \
    #     --prediction_file_path ${OUTPUT_DIR}/${SAVE_NAME}_global_step_${ckpt_num}/screenspot_pro_test.json
    cd ..
    cd scripts   
done

