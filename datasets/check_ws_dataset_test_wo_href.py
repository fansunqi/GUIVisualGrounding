import json
import os
import pdb

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)    

img_dir = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/images_mind2web_test_png"
train_filepath = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_test_full_ws_pretrain_png_wo_href.json"
save_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_test_full_ws_pretrain_png_wo_href_fixed.json"


if __name__ == "__main__":
    data = load_json(train_filepath)
    print(f"Loaded {len(data)} samples from {train_filepath}")

    # Check and fix each sample
    new_data = []
    for idx, sample in enumerate(data):
        img_url = sample["img_url"]
        img_path = os.path.join(img_dir, img_url)
        if not os.path.exists(img_path):
            print(f"Image not found for sample {idx}: {img_url}")
        else:
            new_data.append(sample)
        
    print(f"After fixing, {len(new_data)} samples remain.")
    save_json(new_data, save_path)
    print(f"Saved fixed dataset to {save_path}")
    