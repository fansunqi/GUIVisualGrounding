python inference/inference_vllm_mind2web.py \
    --model_path /data/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
    --output_name qwen_2_5_vl_3b


python inference/inference_vllm_mind2web.py \
    --model_path /data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b/global_step_570/actor/huggingface \
    --data_path /data/fsq/gui_agent_data/Mind2Web/metadata/hf_test_domain.json \
    --output_name mind2web_ws_grpo_qwen2_5_vl_3b_global_step_570