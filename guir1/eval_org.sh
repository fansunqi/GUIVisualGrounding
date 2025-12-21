
# MODEL_NAME=mind2web_ws_grpo_qwen2_5_vl_3b_global_step_575
MODEL_NAME=gui_phase3_from_mind2web_phase1_r1gui_org_grpo_qwen2_5_vl_3b_h20_global_step_180
DATA_DIR=./outputs/${MODEL_NAME}


# python evaluation/eval_omni.py --model_id ${MODEL_NAME} --prediction_file_path  ${DATA_DIR}/androidcontrol_high_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/androidcontrol_low_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiact_web_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiodyssey_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_desktop_test.json
# python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_web_test.json
# python eval/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test.json
python eval/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_test.json