#!/bin/bash
set -x
cd /home/fsq/gui_agent/GUI-R1/scripts/

# 遍历ckpt编号，从1到10为例
ckpt_numbers=(250 300 350 400 450 500 550 600)
for ckpt_num in "${ckpt_numbers[@]}"; do
    echo "Processing ckpt number: $ckpt_num"
   
    LOCAL_DIR=/data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_$ckpt_num/actor
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
        --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_$ckpt_num/actor/huggingface \
        --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
        --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num
    python eval/eval_mind2web.py \
        --pred_path /home/fsq/gui_agent/GUI-R1/guir1/outputs/mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num/hf_test_task.json
    python inference/inference_vllm_mind2web.py \
        --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_$ckpt_num/actor/huggingface \
        --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_website.json \
        --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num
    python eval/eval_mind2web.py \
        --pred_path /home/fsq/gui_agent/GUI-R1/guir1/outputs/mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num/hf_test_website.json
    python inference/inference_vllm_mind2web.py \
        --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_$ckpt_num/actor/huggingface \
        --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
        --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num
    python eval/eval_mind2web.py \
        --pred_path /home/fsq/gui_agent/GUI-R1/guir1/outputs/mind2web_ws_grpo_qwen2_5_vl_3b_global_step_$ckpt_num/hf_test_domain.json
    cd ..
    cd scripts   
done

