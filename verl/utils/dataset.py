# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import json


import yaml
import pdb
from omegaconf import OmegaConf
from verl.utils.tokenizer import get_processor, get_tokenizer

import re


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, str):
        image = Image.open(image)
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


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

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
            # print(data_path)

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]

        # prompt_str: str = row_dict[self.prompt_key]
        text=row_dict['instruction']
        history=row_dict['history']
        task_type=row_dict['task_type']
        row_dict.pop('verify_bbox', None)
        row_dict.pop('success_rate', None)
        row_dict.pop('scale', None)
        images=[row_dict['image']]
      
        if task_type=='high':
            prompt_str=  (
                f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
                "Please provide the action to perform (enumerate from ['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
                "Note:\n specific input text (no default) is necessary for actions enum['type', 'select', 'scroll'] \n Example:\n"
                "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
                "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
                "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
                "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
            )
        else:
            prompt_str=(
                f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
                "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
                "Example:\n"
                "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            )
        messages = [{"role": "user", "content": prompt_str}]
        images=[process_image(image, self.max_pixels, self.min_pixels) for image in images]

        scalex,scaley=images[0].size
        gt_bbox=row_dict['gt_bbox']
        gt_bbox[0]*=scalex
        gt_bbox[1]*=scaley
        if len(gt_bbox)>2:
            gt_bbox[2]*=scalex
            gt_bbox[3]*=scaley

        gt={'action': row_dict['gt_action'],'gt_bbox': gt_bbox,'input_text': row_dict['gt_input_text']}
        # if self.system_prompt:
        #     messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # if self.image_key in row_dict:
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        row_dict["multi_modal_data"] = {
            "image": images
        }
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )  # (3, seq_length)
        # else:
        #     model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = json.dumps(gt)
        return row_dict
    


class Mind2WebDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        use_history: bool = False,
        img_dir: str = None,
        use_task: str = "gt",
        history_num: int = 0,
        interleaved_history: str = 'tttt',
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # 通过 load_json 加载 mind2web 文件
        self.dataset = load_json(data_path)
        self.image_dir = img_dir
        self.use_history = use_history
        self.use_task = use_task
        self.history_num = history_num
        self.interleaved_history = interleaved_history
        
    def __len__(self):
        return len(self.dataset)
    
    def append_history_image(self, sample, num_history, image_list, url_only=False):
        if num_history == 0:
            return image_list
        step_history = sample['step_history']
        for i, step in enumerate(step_history[-num_history:], start=1):
            image_path = os.path.join(self.image_dir, step["img_url"])
            if url_only:
                image_list.append(image_path)
            else:
                image_list.append(Image.open(image_path).convert("RGB"))
        return image_list
    
    def get_history_qwen(self, image_list, sample, num_history, interleaved_history='tttt', decay_factor=1):
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
            max_pixels = max(self.min_pixels, self.max_pixels * decay_factor ** (num_history - i))
            
            # 注意，下面的 action_prefix 和 action_history 是不同的
            if interleaved_history == 'vvtt':
                action_prefix.append({"type": "image", "image": image_list[i+1], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'ttvv':
                action_prefix.append({"type": "text", "text": f'{action}'})

            if interleaved_history in ['tttt', 'vvtt']:
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history in ['vvvv', 'ttvv']:
                action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
            elif interleaved_history == 'vtvt':
                action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
                action_history.append({"type": "text", "text": f'{action}'})
            elif interleaved_history == 'tvtv':
                action_history.append({"type": "text", "text": f'{action}'})
                action_history.append({"type": "image", "image": image_list[i+1], "min_pixels": self.min_pixels, "max_pixels": max_pixels})
        # tmp = action_prefix + action_history + curr_dict
        tmp = action_prefix + action_history
        return tmp
    
    def __getitem__(self, index):
        
        row_dict: dict = self.dataset[index]

        # prompt_str: str = row_dict[self.prompt_key]
        # text = row_dict['task']
        # history=row_dict['history']
        # task_type=row_dict['task_type']
        # row_dict.pop('verify_bbox', None)
        # row_dict.pop('success_rate', None)
        # row_dict.pop('scale', None)
        # images=[row_dict['image']]
        
        if self.use_task == "gt":
            text = row_dict['task']
        elif self.use_task == "dummy":
            text = "click any clickable area on the page, such as a button, but not a blank space"
        elif self.use_task == "meta":
            if row_dict['step']['operation']['op'].lower() == 'click':
                text = "click any clickable area on the page, such as a button, but not a blank space"
            elif row_dict['step']['operation']['op'].lower() == 'type':
                text = "type any input text into the input field on the page"  
            elif row_dict['step']['operation']['op'].lower() == 'select':
                text = "select any valid option from the dropdown menu on the page"
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        image_url = row_dict["img_url"]
        image_path = os.path.join(self.image_dir, image_url)
        image_list = [image_path]
          
        if self.history_num > 0 and self.use_history:
            
            # 首先需要得到一个 image_list
            if self.interleaved_history in ['vvvv', 'vvtt', 'ttvv', 'vtvt', 'tvtv']:
                image_list = self.append_history_image(row_dict, self.history_num, image_list, url_only=True)
            
            history = self.get_history_qwen(
                image_list = image_list, 
                sample = row_dict, 
                num_history = self.history_num, 
                interleaved_history=self.interleaved_history)
            
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
            
            # import pdb; pdb.set_trace()
            # print("history_str:", history_str)
              
            prompt_str = (
                f"You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history_str}'.\n"
                "Please provide the action to perform (enumerate from ['click', 'type', 'select']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>[{'action': enum[ 'click', 'type', 'select'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
                "Note:\n specific input text (no default) is necessary for actions enum['type', 'select'] \n Example:\n"
                "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
                "[{'action': enum['type', 'select'], 'point': [100, 100], 'input_text': 'shanghai shopping mall'}]\n"
            )
        else:
            prompt_str = (
                f"You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}' on the current screenshot.\n"
                "Please provide the action to perform (enumerate from ['click', 'type', 'select']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>[{'action': enum[ 'click', 'type', 'select'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
                "Note:\n specific input text (no default) is necessary for actions enum['type', 'select'] \n Example:\n"
                "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
                "[{'action': enum['type', 'select'], 'point': [100, 100], 'input_text': 'shanghai shopping mall'}]\n"
            )
        
        messages = [{"role": "user", "content": prompt_str}]
        images=[process_image(image_path, self.max_pixels, self.min_pixels) for image_path in image_list] 
        
        bbox = row_dict["step"]["bbox"]
        
        if isinstance(bbox, dict):
            
            x = bbox["x"]
            y = bbox["y"]
            width = bbox["width"]
            height = bbox["height"]
            
            gt_bbox = [x, y, x + width, y + height]
        elif isinstance(bbox, list):
            gt_bbox = []
            for single_bbox in bbox:
                x = single_bbox["x"]
                y = single_bbox["y"]
                width = single_bbox["width"]
                height = single_bbox["height"]
                
                single_gt_bbox = [x, y, x + width, y + height]
                gt_bbox.append(single_gt_bbox)
        else:
            raise NotImplementedError
                
        
        gt_op = row_dict['step']['operation']['op']
        gt_op = gt_op.lower() # 转换为小写
        gt={'action': gt_op,'gt_bbox': gt_bbox,'input_text': row_dict['step']['operation']['value']}
        # if self.system_prompt:
        #     messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # import pdb; pdb.set_trace()
        
        # if self.image_key in row_dict:
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        row_dict["multi_modal_data"] = {
            "image": images
        }
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )  # (3, seq_length)
        # else:
        #     model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = json.dumps(gt)
        return row_dict
        
        
        
        
        

    

if __name__ == "__main__": 
    
    # h100
    config_path =  "/home/fsq/gui_agent/GUI-R1/examples/config_mind2web.yaml"
    config = OmegaConf.load(config_path)
    config.worker.actor.model.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    config.data.system_prompt = """"""
    mind2web_train_path = "/data/fsq/gui_agent_data/Mind2Web/metadata/hf_train_history_image.json"
    mind2web_image_dir = "/data/fsq/gui_agent_data/Mind2Web/images/"
    
    # 103
    # config_path =  "/home/fsq/gui_agent/GUI-R1-Evol-2/examples/config_mind2web_4090.yaml"
    # config = OmegaConf.load(config_path)
    # config.worker.actor.model.model_path = "/mnt/Shared_06_disk1/fsq/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
    # config.data.system_prompt = """"""
    # mind2web_train_path = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/metadata/hf_train.json"
    # mind2web_image_dir = "/mnt/Shared_06_disk1/fsq/data/Mind2Web/images/"
    
    # h20
    # config_path =  "/apdcephfs_private/qy/projects/fsq/GUI-R1-Evol-2/examples/config_mind2web.yaml"
    # config = OmegaConf.load(config_path)
    # config.worker.actor.model.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    # config.data.system_prompt = """"""
    # data_dir = "/root/cache/hub/datasets--fansunqi--Mind2Web_R1/snapshots/70f9286e9c22b585b28c2fe6e766fd57977df18b"
    # mind2web_train_path = os.path.join(data_dir, "metadata/hf_train.json")
    # mind2web_image_dir = os.path.join(data_dir, "images")
    
    # instantiate tokenizer
    tokenizer = get_tokenizer(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    
    
    train_dataset = Mind2WebDataset(
        data_path=mind2web_train_path,
        img_dir=mind2web_image_dir,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        image_key=config.data.image_key,
        max_prompt_length=config.data.max_prompt_length,
        truncation="right",
        system_prompt=config.data.system_prompt,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        use_history=True,
        history_num=4,
        interleaved_history="tvtv"
    )
    
    for i in range(len(train_dataset)):
        data_item = train_dataset[i]
        import pdb; pdb.set_trace()
    
    
    '''
    # RLHFDataset
    config_path =  "/home/fsq/gui_agent/GUI-R1/examples/config.yaml"
    config = OmegaConf.load(config_path)
    config.worker.actor.model.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    config.data.system_prompt = """"""
    gui_r1_train_path = "/home/fsq/hf_home/hub/datasets--ritzzai--GUI-R1/snapshots/ca55ddaa180c5e8f8b27003221c391efa10a1f52/train.parquet"
    
    # instantiate tokenizer
    tokenizer = get_tokenizer(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    
    train_dataset = RLHFDataset(
        data_path=gui_r1_train_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        image_key=config.data.image_key,
        max_prompt_length=config.data.max_prompt_length,
        truncation="right",
        system_prompt=config.data.system_prompt,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
    )
    '''

