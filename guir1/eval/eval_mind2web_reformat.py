import re
import os
import ast
import json
import argparse
import numpy as np


IMG_DIR = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/images"
action2id = {'CLICK': 4, 'SELECT': 2, 'TYPE': 3}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(save_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


def find_item_by_action_uid(data_list, action_uid):
    for item in data_list:
        if item["action_uid"] == action_uid:
            return item
    return None


def extract_trailing_number_rpartition(s: str):
    s = s.strip()
    _, sep, tail = s.rpartition('_')
    if sep and tail.isdigit():
        return int(tail)
    return None


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
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 2) for item in click_point]
    answer = {'action': action_type, 'value': element, 'position': click_point}
    return answer


def get_bbox(meta):
    image_size = meta['img_size']
    action = meta['step']

    bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
            action["bbox"]["y"] + action["bbox"]["height"]]
    # bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
    # bbox = [round(item, 3) for item in bbox]
    return bbox


def match_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_mind2web_metrics(results):
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {'CLICK': [], 'TYPE': [], 'SELECT': []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0

    for i, (annot_id, item) in enumerate(results.items()):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True
        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1
    
    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])
    macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])

    print("[Operation F1]: " + str(marco_op_f1))
    print("[Element Acc]: " + str(num_ele / num_step))
    print("[Step Success]: " + str(num_step_success / num_step))
    print("[Episode Success]: " + str(num_episode_success / num_episode))
    print("[Operation F1 cate]: " + str([np.mean(x) for x in op_f1.values()]))

    print("[Macro Ele Acc]: " + str(macro_ele_acc))
    print("[Macro Op F1]: " + str(macro_action_f1))
    print("[Macro Step SR]: " + str(macro_step_acc))

    metrics = {
        "Operation F1": marco_op_f1,
        "Element Accuracy": num_ele / num_step,
        "Step Success": num_step_success / num_step,
        "Episode Success": num_episode_success / num_episode,
        "Operation F1 categories": [np.mean(x) for x in op_f1.values()],
        "Macro Element Accuracy": macro_ele_acc,
        "Macro Operation F1": macro_action_f1,
        "Macro Step Success Rate": macro_step_acc
    }
    return metrics

def main(args):    
    # 读取 gt
    gt = load_json(args.gt_path)
    
    results = {}
     
    with open(args.pred_path) as file:
        for line in file:
            
            line_result = json.loads(line)
            action_uid = line_result["action_id"]
            
            # 从 gt 中，根据 action_uid 找到对应的 gt_item
            gt_item = find_item_by_action_uid(gt, action_uid)
            split_i = gt_item["split"]
            if split_i not in results:
                results[split_i] = {}
                
            # id -> anno_id
            id = gt_item["id"]
            anno_id = extract_trailing_number_rpartition(id)
            
            if anno_id not in results[split_i]:
                results[split_i][anno_id] = []
            
            meta = gt_item
            if 'img_url' in gt_item.keys():
                image_path = os.path.join(IMG_DIR, gt_item["img_url"])
            else:
                image_path = ""
            meta["img_url_abs"] = image_path
            meta["anno_id"] = anno_id
            answer = get_answer(gt_item, gt_item['step'], gt_item['step_repr'])
            meta["answer"] = answer
            
            step_result = {
                "split": split_i,
                "anno_id": anno_id, 
                "img_path": meta['img_url_abs'], 
                "instruction": meta['task'], 
                # "sentence": generated_texts,
                "Op_match": False, 
                "Ele_match": False, 
                "Op_F1": [0, meta['answer']["action"]],
                "meta": meta
            }
            
            try:
                pred = line_result["pred"]
                pred_answer = match_answer(pred)
                action_pred = ast.literal_eval(pred_answer)[0]
                action_pred["action"] = action_pred["action"].upper()
                
                if action_pred["action"] == answer["action"]:
                    step_result["Op_match"] = True 
                    
                # 判断点击位置是否在 bbox 内
                click_point = action_pred["point"] 
                bbox_ref = get_bbox(meta)
                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True
                    
                
                action_pred_idx = action2id[action_pred["action"]]
                pred_str = str(action_pred_idx)
                if action_pred["action"] in ['TYPE', 'SELECT']:
                    pred_str += ' '
                    pred_str += action_pred["input_text"].lower()
                    
                
                action_ref_idx = action2id[answer["action"]]
                ref_str = str(action_ref_idx)
                if answer["action"] in ['TYPE', 'SELECT']:
                    ref_str += ' '
                    ref_str += answer["value"].lower()
                    
                op_f1 = calculate_f1(pred_str, ref_str)
                step_result["Op_F1"][0] = op_f1
            
            except Exception as e:
                print(e)
                print(f"format wrong with {anno_id}'s prediction: {pred}")
                
            results[split_i][anno_id].append(step_result)
    
    
    eval_dict = {}
    for split in results.keys():
        print("==="*10)
        print(f"{split}")
        print("==="*10)
        eval_dict[split] = calculate_mind2web_metrics(results[split])
        
    metric = sum([x["Macro Step Success Rate"] for x in eval_dict.values()]) / len(eval_dict)

    # save_json(results, os.path.join(args.output_path, f'mind2web_tmp_dict.json'))
    save_json(eval_dict, os.path.join(args.output_path, f'reformat_eval.json'))
                
                
            
            
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, default="/mnt/Shared_06_disk1/fsq/data/Mind2Web/metadata/hf_test_full.json")
    parser.add_argument('--output_path', type=str, default="output_path")
    args = parser.parse_args()
    
    args.output_path = os.path.dirname(args.pred_path)
    main(args)