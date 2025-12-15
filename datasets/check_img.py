import os
import json

train_data_path = "/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/metadata/hf_train_ws_sim_0.7.json"
img_dir = "/mnt/Shared_05_disk/fsq/gui_agent_data/Mind2Web/images"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
train_data = load_json(train_data_path)

for data_item in train_data:
    img_url = data_item["img_url"]
    img_path = os.path.join(img_dir, img_url)
    
    if not os.path.exists(img_path):
        print(img_url)