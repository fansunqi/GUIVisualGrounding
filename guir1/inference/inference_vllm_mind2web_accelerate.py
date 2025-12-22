import os
import re
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from datasets import Dataset as hf_dataset
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader

import math
import torch
from PIL import Image
from io import BytesIO
from PIL.Image import Image as ImageObject
from typing import Any, Dict, Union

from accelerate import Accelerator


# 推理参数
SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.001,
    "repetition_penalty": 1.05,
    "max_tokens": 1024,
}

# 微批大小
MICRO_BATCH = 6


def extract_coord(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    

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


class MultiModalDataset(Dataset):
    def __init__(self, data, processor, image_dir):
        self.data = data
        self.processor = processor
        self.processor.max_pixels = 2097152
        self.processor.min_pixels = 262144
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        task = sample['task']
        
        image_url = sample["img_url"]
        image_path = os.path.join(self.image_dir, image_url)
        image = process_image(image_path, self.processor.max_pixels, self.processor.min_pixels)

        text = (
            f"You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{task}' on the current screenshot.\n"
            "Please provide the action to perform (enumerate from ['click', 'type', 'select']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum[ 'click', 'type', 'select'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['type', 'select'] \n Example:\n"
            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['type', 'select'], 'point': [100, 100], 'input_text': 'shanghai shopping mall'}]\n"
        )
        text = '<image>\n' + text
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
              
        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"] = [scale_x.item(), scale_y.item()]
        sample["image_size"] = [origin_width, origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch


def process_data(model, processor, dataloader, sampling_params, accelerator):
    results = []
    model.eval()
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        prompts = batch["prompts"]
        multi_modal_data = batch["multi_modal_data"]
        original_samples = batch["original_samples"]

        inputs = processor(
            text=prompts,
            images=[mm_data.get("image") for mm_data in multi_modal_data],
            # videos=[mm_data.get("video") for mm_data in multi_modal_data],
            padding=True,
            return_tensors="pt",
        ).to(accelerator.device)

        with torch.no_grad():
            outputs = model.module.generate(
                **inputs,
                max_new_tokens=sampling_params["max_tokens"],
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                repetition_penalty=sampling_params["repetition_penalty"],
            )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

        for original_sample, generated_text in zip(original_samples, decoded):
            result_item = {
                "id": original_sample["id"],
                "action_id": original_sample["action_uid"],
                "pred": generated_text,
                "gt_coord": original_sample["step"]["bbox"],
                "gt_op": original_sample["step"]["operation"],
            }
            pred_coord, _ = extract_coord(generated_text)
            result_item["pred_coord"] = pred_coord
            results.append(result_item)

    return results


def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    # 加载数据
    if args.data_path.endswith('parquet'):
        data = load_dataset("parquet", data_files=args.data_path, split="train")
    else:
        data = [json.loads(s) for s in open(args.data_path, "r")] if args.data_path.endswith(".jsonl") else json.load(open(args.data_path,"r"))

    processor = AutoProcessor.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.float16
    # ).to(device)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    ).to(device)


    dataset = MultiModalDataset(data, processor, args.image_dir)
    dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    model, dataloader = accelerator.prepare(model, dataloader)

    results = process_data(model, processor, dataloader, SAMPLING_PARAMS, accelerator)

    if accelerator.is_main_process:
        OUTPUT_DIR = os.path.join(args.output_path, args.output_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        NEW_FILE = os.path.join(
            OUTPUT_DIR,
            args.data_path.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet','.json')
        )
        with open(NEW_FILE, "w") as ans_file:
            for sample in results:
                ans_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--image_dir', type=str, default="/data/fsq/gui_agent_data/Mind2Web/images")
    parser.add_argument('--output_name', type=str, default="<output_name>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    args = parser.parse_args()
    main(args)
