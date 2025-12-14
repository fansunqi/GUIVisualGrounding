import re
import pdb
import os
import csv


def read_txt_value(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        total, valid, correct = map(int, numbers[:3])
        accuracy = float(numbers[3])
        data = {
            "total": total,
            "valid": valid,
            "correct": correct,
            "accuracy": accuracy
        }
        # return data
        return accuracy




if __name__ == "__main__":
    txt_path = "/home/fsq/gui_agent/GUI-R1/guir1/outputs/mind2web_ws_grpo_qwen2_5_vl_3b_global_step_5/hf_test_domain_eval.txt"
    
    output_csv = "/home/fsq/gui_agent/GUI-R1/stat/mindweb_eval_summary.csv"
    tasks = ["task", "website", "domain"]
    ckpt_numbers = list(range(25, 625, 25))

    # Prepare a dictionary to store results
    results = {task: [] for task in tasks}

    for ckpt in ckpt_numbers:
        for task in tasks:
            txt_path = f"/home/fsq/gui_agent/GUI-R1/guir1/outputs/mind2web_ws_grpo_qwen2_5_vl_3b_global_step_{ckpt}/hf_test_{task}_eval.txt"
            if os.path.exists(txt_path):
                res = read_txt_value(txt_path)
                results[task].append(res)
            else:
                results[task].append("")

    # Write to CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["ckpt"] + tasks
        writer.writerow(header)
        for idx, ckpt in enumerate(ckpt_numbers):
            row = [str(ckpt)]
            for task in tasks:
                res = results[task][idx]
                row.append("" if res == "" else str(res))
            writer.writerow(row)
                
                
    
        