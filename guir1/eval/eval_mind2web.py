import pdb
import json
import argparse


def check_contain(pred_coord, gt_coord):
    px,py=pred_coord
    gx,gy,gw,gh=gt_coord["x"],gt_coord["y"],gt_coord["width"],gt_coord["height"]
    if px>=gx and py>=gy and px<=gx+gw and py<=gy+gh:
        return True
    else:
        return False
    
def main(args):
    total = 0
    valid = 0
    correct = 0
    with open(args.pred_path) as file:
        for line in file:
            line_result = json.loads(line)
            total += 1
            if line_result["pred_coord"]==[0,0,0,0]:
                continue
            else:
                valid += 1
                if check_contain(line_result["pred_coord"], line_result["gt_coord"]):
                    correct += 1
    
    acc = correct / total if total>0 else 0
    print("total:", total, "valid:", valid, "correct:", correct)
    print("accuracy:", acc)
    
    save_path = args.pred_path.replace(".json", "_eval.txt")
    with open(save_path, "w") as f:
        f.write(f"total: {total}, valid: {valid}, correct: {correct}\n")
        f.write(f"accuracy: {acc}\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='<pred_path>')
    args = parser.parse_args()
    main(args)
    
