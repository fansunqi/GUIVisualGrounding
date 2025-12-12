# python inference/inference_vllm_mind2web.py \
#     --model_path /data/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
#     --output_name qwen_2_5_vl_3b

# export TORCH_COMPILE_CACHE=/data/fsq/vllm_cache
# python inference/inference_vllm_mind2web.py \
#     --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_250/actor/huggingface \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
#     --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_250 \
#     --num_actor 2

# accelerate launch inference/inference_vllm_mind2web_accelerate.py \
#     --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_50/actor/huggingface \
#     --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
#     --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_50


python inference/inference_vllm_mind2web.py \
    --model_path /mnt/Shared_05_disk/fsq/hf_home/hub/models--ByteDance-Seed--UI-TARS-2B-SFT/snapshots/f366a1db3e7f29635f5b236d6a71dea367a0a700 \
    --data_path /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_test_task.json \
    --image_dir /mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/images \
    --output_name ui_tars_2b_sft \
    --num_actor 2