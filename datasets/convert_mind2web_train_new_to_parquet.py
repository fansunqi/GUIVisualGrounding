import re
import os
import json
import pandas as pd
import pdb
from tqdm import tqdm

data_dir = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b"

# mind2web_train_new_data_path = os.path.join(data_dir, "metadata/hf_train_new.json")
# save_file = os.path.join(data_dir, "metadata/hf_train_new.parquet")

# mind2web_train_new_data_path = os.path.join(data_dir, "metadata/hf_test_full.json")
# save_file = os.path.join(data_dir, "metadata/hf_test_full.parquet")

mind2web_train_new_data_path = os.path.join(data_dir, "metadata/hf_test_task.json")
save_file = os.path.join(data_dir, "metadata/hf_test_task.parquet")

image_dir = os.path.join(data_dir, "images")
history_num = 4
interleaved_history = "tttt"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
 
def get_image_size(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)   


def get_value(step_repr):
    pattern = r'\]\s+(.*?)\s+->'
    match = re.search(pattern, step_repr)
    if match:
        return match.group(1)
    else:
        return None

def get_answer(sample, step, step_repr):
    image = sample['img_url']
    image_size = sample['img_size']
    task = sample['task']

    action_type = step['operation']['op']
    if action_type != 'TYPE':
        element = get_value(step_repr)
    else:
        element = step['operation']['value']
    bbox = step['bbox']
    point_x = bbox["x"] + (bbox["width"] / 2)
    point_y = bbox["y"] + (bbox["height"] / 2)
    # click_point = [point_x / image_size[0], point_y / image_size[1]]
    # click_point = [round(item, 2) for item in click_point]
    click_point = [point_x, point_y]
    click_point = [int(item) for item in click_point]
    
    action_type = action_type.lower() # 转换为小写
    
    # answer = {'action': action_type, 'value': element, 'position': click_point}
    answer = {'action': action_type, 'point': click_point, 'input_text': element,}
    return answer


def get_history_qwen(image_list, sample, num_history, interleaved_history='tttt', decay_factor=1):
    
    # dummy
    min_pixels = 0
    max_pixels = 0
    
    # last one is the current image, past are the history
    # curr_image = image_list[-1]
    # curr_dict = [{'type': 'image', 'image': curr_image, 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}]
    if num_history == 0 or sample['step_history'] == []:
        # assert len(image_list) == 1
        # return curr_dict
        return []
    
    step_history = sample['step_history']
    repr_history = sample['repr_history']
    
    action_history = []
    action_prefix = []
    for i, (step, step_repr) in enumerate(zip(step_history[-num_history:], repr_history[-num_history:])):
        
        action = get_answer(sample, step, step_repr)
        max_pixels = max(min_pixels, max_pixels * decay_factor ** (num_history - i))
        
        # 注意，下面的 action_prefix 和 action_history 是不同的
        if interleaved_history == 'vvtt':
            action_prefix.append({"type": "image", "image": image_list[i+1], "min_pixels": min_pixels, "max_pixels": max_pixels})
        elif interleaved_history == 'ttvv':
            action_prefix.append({"type": "text", "text": f'{action}'})

        if interleaved_history in ['tttt', 'vvtt']:
            action_history.append({"type": "text", "text": f'{action}'})
        elif interleaved_history in ['vvvv', 'ttvv']:
            action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": min_pixels, "max_pixels": max_pixels})
        elif interleaved_history == 'vtvt':
            action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": min_pixels, "max_pixels": max_pixels})
            action_history.append({"type": "text", "text": f'{action}'})
        elif interleaved_history == 'tvtv':
            action_history.append({"type": "text", "text": f'{action}'})
            action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": min_pixels, "max_pixels": max_pixels})
    # tmp = action_prefix + action_history + curr_dict
    tmp = action_prefix + action_history
    return tmp
    
    
if __name__ == "__main__":
    data = load_json(mind2web_train_new_data_path)
    print(f"Loaded {len(data)} records from Mind2Web R1 train new dataset.")
    
    new_data = []
    for data_item in tqdm(data):
        
        image_url = data_item["img_url"]
        image_path = os.path.join(image_dir, image_url)
        (img_width, img_height) = get_image_size(image_path)
        
        gt_op = data_item['step']['operation']['op']
        gt_op = gt_op.lower() # 转换为小写
        
        instruction = data_item['task']
        
        # gt_box
        # [x, y, x + width, y + height]
        # [0-1 之间的]
        bbox = data_item["step"]["bbox"]
        x = bbox["x"] / img_width 
        y = bbox["y"] / img_height
        width = bbox["width"] / img_width
        height = bbox["height"] / img_height
        gt_bbox = [x, y, x + width, y + height]
        
        gt_input_text = data_item['step']['operation']['value']
        
        
        # history
        history = get_history_qwen(
            image_list = [], 
            sample = data_item, 
            num_history = history_num, 
            interleaved_history=interleaved_history)
        
        # 将 history 从 list 变成 str
        history_str = ""
        if history is None or len(history) == 0:
            history_str = "no history"
        else:
            for item in history:
                if item['type'] == 'text':
                    history_str += item['text'] + " "
                if item['type'] == 'image':
                    history_str += "<image>" + " "
        
        row_dict = {
            "image": image_path,
            "gt_bbox": gt_bbox,
            "instruction": instruction,
            "id": data_item["id"],# 这个暂时不清楚是什么作用
            "gt_action": gt_op,
            "gt_input_text": gt_input_text,
            "history": history_str,
            "task_type":"high"      
        }
        new_data.append(row_dict)
        
        # pdb.set_trace()
        
        
    df = pd.DataFrame(new_data)
    df.to_parquet(save_file, index=False)
    print(f"Saved converted dataset to {save_file}")

