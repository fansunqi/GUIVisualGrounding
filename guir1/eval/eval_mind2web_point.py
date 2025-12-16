import pdb
import json
import argparse
from pprint import pprint


def check_contain(pred_coord, gt_coord):
    px,py=pred_coord
    gx,gy,gw,gh=gt_coord["x"],gt_coord["y"],gt_coord["width"],gt_coord["height"]
    if px>=gx and py>=gy and px<=gx+gw and py<=gy+gh:
        return True
    else:
        return False

def find_item_by_action_uid(data, action_uid):
    for item in data:
        if item["action_uid"] == action_uid:
            return item
    return None

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add1(dict_stat, data_type):
    if data_type in dict_stat:
        dict_stat[data_type] += 1
    else:
        dict_stat[data_type] = 1
  
def main(args):
    
    gt_data = load_json(args.gt_path)
    
    total = {}
    valid = {}
    correct = {}
    acc = {}
    with open(args.pred_path) as file:
        for line in file:
            line_result = json.loads(line)
            action_uid = line_result["action_id"]
            data_item = find_item_by_action_uid(gt_data, action_uid)
            split = data_item["split"]
            
            add1(total, split)
            if line_result["pred_coord"]==[0,0,0,0]:
                continue
            else:
                add1(valid, split)
                if check_contain(line_result["pred_coord"], line_result["gt_coord"]):
                    add1(correct, split)
    
    for data_type in total.keys():
        acc[data_type] = correct[data_type] / total[data_type] if total[data_type]>0 else 0
    
    total["all"] = sum(v for k, v in total.items() if k != "all")
    valid["all"] = sum(v for k, v in valid.items() if k != "all")
    correct["all"] = sum(v for k, v in correct.items() if k != "all")
    acc["all"] = correct["all"] / total["all"]
    
    all_results = {
        "total": total,
        "valid": valid,
        "correct": correct,
        "acc": acc,
    }
      
    pprint(all_results)
    
    save_path = args.pred_path.replace(".json", "_eval.json")  
    save_json(all_results, save_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='<pred_path>')
    parser.add_argument('--gt_path', type=str, default='<gt_path>')
    args = parser.parse_args()
    main(args)
    
