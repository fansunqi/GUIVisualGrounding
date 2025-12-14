import json
import os

# 将 bbox 替代为 clickable
# 要考虑 match results

train_data_path = "/home/fsq/gui_agent/GUI-agent-data-process/hf_train_new.json"
match_results_path = "/home/fsq/gui_agent/GUI-agent-data-process/screenshot_match_results_train_all_new.json"
interact_results_dir = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/interact_results/clickables/data"
image_dir = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/images"

new_data_path = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/metadata/hf_train_sim_0.7.json"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_to_dict(match_list):
    match_dict = {}
    for match_item in match_list:
        match_dict[match_item["action_uid"]] = match_item["pos"]
    return match_dict


def check_contain(small_box, big_box):
    '''
    检查 small_box 是否被完全包含在 big_box 中
    small_box, big_box 是一个 dict, e.g.
    {
        "x": 193.375,
        "y": 0,
        "width": 128.40625,
        "height": 65
    }
    '''
    sx, sy, sw, sh = small_box["x"], small_box["y"], small_box["width"], small_box["height"]
    bx, by, bw, bh = big_box["x"], big_box["y"], big_box["width"], big_box["height"]

    if (sx >= bx and sy >= by and
        sx + sw <= bx + bw and
        sy + sh <= by + bh):
        return True
    else:
        return False


def compute_similarity(box1, box2):
    '''
    计算 box1 和 box2 的相似/重合度
    box1, box2 是一个 dict, e.g.
    {
        "x": 193.375,
        "y": 0,
        "width": 128.40625,
        "height": 65
    }
    '''
    # runqi
    eps = 6
    if box1["x"] > box2["x"] - eps and box1["x"] + box1["width"] < box2["x"] + box2["width"] + eps and \
       box1["y"] > box2["y"] - eps and box1["y"] + box1["height"] < box2["y"] + box2["height"] + eps:
        return 1.0
    
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]

    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area



skip_action_uid_list = [
    "a6f86a41-b433-478a-b445-563cafaebe34"
]

if __name__ == "__main__":
    train_data = load_json(train_data_path)
    
    
    
    match_results = load_json(match_results_path)
    match_results_reformat = list_to_dict(match_results)
    
    train_data_new = []
    for train_item in train_data:
        # 替换 train_item["step"]["bbox"]
        
        # 读取 interact_results
        id = train_item["id"]
        action_uid = train_item["action_uid"]
        
        if action_uid in skip_action_uid_list:
            continue
        
        clickables_path = os.path.join(interact_results_dir, f"{id}_{action_uid}.json")
        clickables = load_json(clickables_path)
        
        # 读取相对应的 match_results:
        pos = match_results_reformat[action_uid]
        pos = pos.strip("()")
        x_str, y_str = pos.split(",")
        x_off = int(x_str.strip())
        y_off = int(y_str.strip())
        
        # 读取图片的尺寸
        image_size = train_item["img_size"]
        crop_box = {
            "x": x_off,
            "y": y_off,
            "width": image_size[0],
            "height": image_size[1]
        }
        
        # 逐个看框框是否在截取的图片中
        if clickables == None:
            continue
        
        gt_bbox = train_item["step"]["bbox"]
        
        hit_bboxes_list = []
        highest_sim_score = 0.0
        for clickable_bbox in clickables:
            if check_contain(clickable_bbox, crop_box):
                
                # clickable_bbox 进行偏移
                clickable_bbox["x"] -= x_off
                clickable_bbox["y"] -= y_off
                
                sim_score = compute_similarity(gt_bbox, clickable_bbox)
                if sim_score > highest_sim_score:
                    highest_sim_score = sim_score
                
                hit_bboxes_list.append(clickable_bbox)
                
        if highest_sim_score < 0.7:   
            continue
                
        print(f"hit_bboxes num: {str(len(hit_bboxes_list))}; highest_sim_score: {str(highest_sim_score)}")
        
        train_item["step"]["bbox"] = hit_bboxes_list
        train_data_new.append(train_item)
        
    print(len(train_data_new))
    print(len(train_data))
    with open(new_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data_new, f, ensure_ascii=False, indent=2)
    

