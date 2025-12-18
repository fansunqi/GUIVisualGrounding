#!/bin/bash
set -x
cd /home/fsq/gui_agent/GUI-R1/scripts/

EXP_DIR=/data/fsq/GUI-R1_exp
DATA_DIR=/data/fsq/gui_agent_data/Mind2Web
SAVE_NAME=mind2web_gt_history_fix_norm_grpo_qwen2_5_vl_3b

export TORCH_COMPILE_CACHE=/home/fsq/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/home/fsq/torchinductor_cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR
chmod -R 777 $TORCHINDUCTOR_CACHE_DIR

# 遍历ckpt编号，从1到10为例
ckpt_numbers=(650)
for ckpt_num in "${ckpt_numbers[@]}"; do
    echo "Processing ckpt number: $ckpt_num"
   
    LOCAL_DIR=${EXP_DIR}/${SAVE_NAME}/global_step_$ckpt_num/actor
    LOCAL_HF_DIR=$LOCAL_DIR/huggingface
    OUTPUT_DIR=/home/fsq/gui_agent/GUI-R1/guir1/outputs

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
        --use_history \
        --history_num 4
    python eval/eval_mind2web_point.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path ${DATA_DIR}/metadata/hf_test_full.json
    python eval/eval_mind2web_reformat.py \
        --pred_path $OUTPUT_DIR/${SAVE_NAME}_global_step_$ckpt_num/hf_test_full.json \
        --gt_path ${DATA_DIR}/metadata/hf_test_full.json
    cd ..
    cd scripts   
done

