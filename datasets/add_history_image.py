import json
from tqdm import tqdm

train_data_path = "/data/fsq/gui_agent_data/Mind2Web/metadata/hf_train_new.json"
save_path = "/data/fsq/gui_agent_data/Mind2Web/metadata/hf_train_new_history_image.json"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
def find_item_by_action_uid(match_results, action_uid):
    for item in match_results:
        if item["action_uid"] == action_uid:
            return item
    # import pdb; pdb.set_trace()
    return None   


if __name__ == "__main__":
    train_data = load_json(train_data_path)
    new_train_data = []
    
    for train_item in tqdm(train_data):
        
        step_history = train_item["step_history"]
        new_step_history = []
        
        right_data = True
        
        for step in step_history:
            step_action_uid = step["action_uid"]
            match_train_item = find_item_by_action_uid(train_data, step_action_uid)
            
            if match_train_item == None:
                print(f"action_uid {step_action_uid} miss")
                right_data = False
                break
                
            img_url = match_train_item["img_url"]
            step["img_url"] = img_url
            new_step_history.append(step)
        
        if right_data == False:
            continue
        
        train_item["step_history"] = new_step_history
        new_train_data.append(train_item)
        
    save_json(save_path, new_train_data)
    
    