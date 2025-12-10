MODEL_PATH=/data/fsq/GUI-R1_exp/ckpt/global_step_90/actor/huggingface
DATA_DIR=/data/fsq/hf_home/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52

# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet
# python inference/inference_vllm_guiact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiact_web_test.parquet
# python inference/inference_vllm_guiodyssey.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiodyssey_test.parquet
# python inference/inference_vllm_omniact_desktop.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_desktop_test.parquet
# python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_web_test.parquet
python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_test.parquet
# python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet