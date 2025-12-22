import os
import pdb
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

screenshot_rendered_test_mhtml_dir = "/apdcephfs_private/qy/projects/fsq/GUI-agent-data-process/screenshot_rendered_test_mhtml"
interact_results_dir = "/apdcephfs_private/qy/projects/fsq/GUI-agent-data-process/interact_results_test"
img_save_dir = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/images_mind2web_test_png"
hf_test_full_path = "/apdcephfs_private/qy/projects/fsq/GUI-agent-data-process/hf_test_full.json"
data_save_path = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b/metadata/hf_test_full_ws_pretrain_png.json"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
 

def save_json(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4, ensure_ascii=False)
        

def process_image(input_path, output_path="cropped_image.jpg"):
    """
    用PIL读取图片，输出尺寸，截取(0,0)开始的1280x720区域并保存
    
    参数:
        input_path: 输入图片的路径（相对路径或绝对路径）
        output_path: 输出截取后图片的路径（默认保存为 cropped_image.jpg）
    """
    try:
        # 1. 读取图片
        # 使用 Image.open() 打开图片，返回 Image 对象
        img = Image.open(input_path)
        
        # 2. 输出图片原始尺寸（宽 x 高）
        # width, height = img.size
        # print(f"原始图片尺寸：宽 {width} px × 高 {height} px")
        
        # pdb.set_trace()
        
        # 3. 定义截取区域（左、上、右、下）
        # 截取规则：从(0,0)开始，宽1280，高720 → 右边界=0+1280，下边界=0+720
        left = 0
        top = 0
        right = 0 + 1280
        bottom = 0 + 720
        crop_region = (left, top, right, bottom)
        
        # 4. 执行截取（crop()方法返回截取后的新Image对象）
        # 注意：如果原始图片尺寸小于1280x720，会自动截取到图片实际边界（不会报错）
        cropped_img = img.crop(crop_region)
        
        # 5. 保存截取后的图片
        cropped_img.save(output_path)
        # print(f"截取完成！已保存为：{os.path.abspath(output_path)}")
        # print(f"截取后图片尺寸：{cropped_img.size[0]} px × {cropped_img.size[1]} px")
        
        # 可选：关闭图片（释放资源，虽然Pillow会自动处理，但好习惯）
        img.close()
        cropped_img.close()
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}，请检查路径是否正确")
        return False
    except PermissionError:
        print(f"错误：没有权限读取/写入文件，请检查文件权限")
        return False
    except Exception as e:
        print(f"处理图片时发生错误：{str(e)}")
        return False

def find_item_by_action_uid(match_results, action_uid):
    for item in match_results:
        if item["action_uid"] == action_uid:
            return item
    return None

if __name__ == "__main__":
    
    # 首先获取所有截屏文件
    screenshot_files = os.listdir(screenshot_rendered_test_mhtml_dir)
    action_uids = [Path(f).stem for f in screenshot_files]
    
    hf_test_full = load_json(hf_test_full_path)
    
    data_list = []
    for action_uid in tqdm(action_uids):
        
        # 首先截取 [1280, 720] 大小的区域
        screenshot_path = os.path.join(screenshot_rendered_test_mhtml_dir, f"{action_uid}.png")
        # img_file = f"{action_uid}.jpg"
        img_file = f"{action_uid}.png"
        save_img_path = os.path.join(img_save_dir, img_file)
        flag = process_image(screenshot_path, output_path=save_img_path)
        
        if flag == False:
            continue
        
        test_item = find_item_by_action_uid(hf_test_full, action_uid)
        id = test_item["id"]
        
        
        # 读取 clickables
        clickable_data_path = os.path.join(interact_results_dir, "clickables/data", f"{id}_{action_uid}.json")
        clickables = load_json(clickable_data_path)
        bbox_list = []
        for bbox in clickables:
            if bbox["x"] + bbox["width"] <= 1280 and bbox["y"] + bbox["height"] <= 720:
                bbox_list.append(bbox)
        if len(bbox_list) > 0:
            data_item = {
                "split": "test_ws_pretrain",
                "id": id,
                "action_uid": action_uid,
                "task": "click any clickable area on the page, such as a button, but not a blank space",
                "img_size": [1280, 720],
                "img_url": img_file,
                "step": {
                    "action_uid": action_uid,
                    "operation": {
                        "value": "",
                        "op": "CLICK"
                    },
                    "bbox": bbox_list
                },
                "step_repr": "",
                "step_history": [],
                "repr_history": []
            }
            data_list.append(data_item)
        
        # 读取 typeables
        typeable_data_path = os.path.join(interact_results_dir, "typeables/data", f"{id}_{action_uid}.json")
        typeables = load_json(typeable_data_path)
        bbox_list = []
        for bbox in typeables:
            if bbox["x"] + bbox["width"] <= 1280 and bbox["y"] + bbox["height"] <= 720:
                bbox_list.append(bbox)
        if len(bbox_list) > 0:
            data_item = {
                "split": "test_ws_pretrain",
                "id": id,
                "action_uid": action_uid,
                "task": "type any input text into the input field on the page",
                "img_size": [1280, 720],
                "img_url": img_file,
                "step": {
                    "action_uid": action_uid,
                    "operation": {
                        "value": "",
                        "op": "TYPE"
                    },
                    "bbox": bbox_list
                },
                "step_repr": "",
                "step_history": [],
                "repr_history": []
            }
            data_list.append(data_item)
            
        # 读取 selectables
        selectable_data_path = os.path.join(interact_results_dir, "selectables/data", f"{id}_{action_uid}.json")
        selectables = load_json(selectable_data_path)
        bbox_list = []
        for bbox in selectables:
            if bbox["x"] + bbox["width"] <= 1280 and bbox["y"] + bbox["height"] <= 720:
                bbox_list.append(bbox)
        if len(bbox_list) > 0:
            data_item = {
                "split": "test_ws_pretrain",
                "id": id,
                "action_uid": action_uid,
                "task": "select any valid option from the dropdown menu on the page",
                "img_size": [1280, 720],
                "img_url": img_file,
                "step": {
                    "action_uid": action_uid,
                    "operation": {
                        "value": "",
                        "op": "SELECT"
                    },
                    "bbox": bbox_list
                },
                "step_repr": "",
                "step_history": [],
                "repr_history": []
            }
            data_list.append(data_item)
            
        
    save_json(data_list, data_save_path)
    print(f"数据保存完成，路径：{data_save_path}, 长度：{len(data_list)}")
        
        
        

            
    