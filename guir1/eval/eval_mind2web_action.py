# adapted from https://github.com/showlab/ShowUI/blob/main/main/eval_mind2web.py
import re
import ast
import json
import argparse
import numpy as np


action2id = {'click': 4, 'select': 2, 'type': 3}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_item_by_action_uid(data_list, action_uid):
    for item in data_list:
        if item["action_uid"] == action_uid:
            return item
    return None

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
    op_f1 = {'click': [], 'type': [], 'select': []}
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

            # 没懂下面的是什么意思
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
            if gt_item is None:
                # print(f"Warning: action_uid {action_uid} not found in gt.")
                raise ValueError(f"action_uid {action_uid} not found in gt.")
            gt_answer = gt_item["step"]
            split = gt_item["split"]
            anno_id = gt_item["annot_id"]
            
            step_result = {}
            step_result["Op_match"] = False
            step_result["Ele_match"] = False
            step_result["Op_F1"] =  [0, gt_answer["operation"]["op"].lower()]
            
            try:
                
                pred = line_result["pred"]
                pred_answer = match_answer(pred)
                # try:
                action_pred = ast.literal_eval(pred_answer)[0]
                # except:
                #     # TODO
                #     raise ValueError(f"pred_answer {pred_answer} is not a valid dictionary string.")
                
                
                # 判断 op 是否一致
                if action_pred["action"] == gt_answer["operation"]["op"].lower():
                    step_result["Op_match"] = True
                    
                # 判断点击位置是否在 bbox 内
                click_point = action_pred["point"]
                bbox_ref = gt_answer["bbox"]
                if (bbox_ref["x"] <= click_point[0] <= bbox_ref["x"] + bbox_ref["width"]) and \
                (bbox_ref["y"] <= click_point[1] <= bbox_ref["y"] + bbox_ref["height"]):
                    step_result["Ele_match"] = True
                
                # 计算 op_f1
                action_pred_idx = action2id[action_pred["action"]]
                pred_str = str(action_pred_idx)
                if action_pred["action"] in ['type', 'select']:
                    pred_str += ' '
                    pred_str += action_pred["input_text"].lower()

                action_ref_idx = action2id[gt_answer["operation"]["op"].lower()]
                ref_str = str(action_ref_idx)
                if gt_answer["operation"]["op"].lower() in ['type', 'select']:
                    ref_str += ' '
                    ref_str += gt_answer["operation"]["value"].lower()
            
                op_f1 = calculate_f1(pred_str, ref_str)
                step_result["Op_F1"][0] = op_f1
            
            except Exception as e:
                print(e)
                print(f"format wrong with {anno_id}'s prediction: {line_result}")

            if split not in results:
                results[split] = {}
            if anno_id not in results[split]:
                results[split][anno_id] = []
            results[split][anno_id].append(step_result)

    eval_dict = {}
    for split in results.keys():
        print("==="*10)
        print(f"{split}")
        print("==="*10)
        eval_dict[split] = calculate_mind2web_metrics(results[split])
        
    output_file = args.pred_path.replace('.json', '_eval.json')
    with open(output_file, 'w') as f:
        json.dump(eval_dict, f, indent=4)
    print(f"Saved eval results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
    
    