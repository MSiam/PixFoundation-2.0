import argparse
import json
import os
import sys
from tqdm import tqdm
from glob import glob
import pandas as pd
import torch.backends.cudnn as cudnn
import random
import shortuuid
sys.path.append(".")

from detectron2.engine import default_argument_parser
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from qwen_vl_utils import process_vision_info

from configs.cfg import setup_cfg
from datasets_.mevis_variants.mevis_dataset_mapper import MeViSDatasetMapper
from datasets_.mevis_variants.mevis_build import build_detection_test_loader, register_all_mevis, \
                                                _PREDEFINED_SPLITS_mevis

# from model.segment_anything.utils.transforms import ResizeLongestSide
# from model.qwen_2_5_vl import UniGRConfig, UniGRModel
from utils.utils import DirectResize
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import get_sparse_indices, dict_to_cuda, preprocess


def parse_args(args):
    #parser = argparse.ArgumentParser(description="Inference")
    parser = default_argument_parser()
    parser.add_argument("--dataset_root")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--num_frames_mllm", default=4, type=int)
    parser.add_argument("--max_pixels", default=384*28*28, type=int)
    parser.add_argument("--inference_mode", default="video")
    parser.add_argument("--postproc", default="simple")
    parser.add_argument('--dataset_type', type=str, default='mevis_variants')
    parser.add_argument('--image_prefix_root_flip', type=str, default='')
    parser.add_argument('--dataset_split', type=str, default='mevis_val', choices= ['mevis_train', 'mevis_val', 'mevis_test',
                                                                                    'mevis_val_mocentric_single',
                                                                                    'mevis_val_mocentric_tile_single',
                                                                                    'mevis_val_mocentric_reverse',
                                                                                    'mevis_val_mocentric_tile_reverse',
                                                                                    'mevis_val_mocentric_tile_black',
                                                                                    'refdavis_val'])

    parser.add_argument('--start_index', type=int, default=0)
    return parser.parse_args(args)

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def process(line, question_extension):
    qs = line["question"] + " Options:"
    options = line["options"].split('(b)')
    parts = [part.strip() for part in options]
    parts = [part.replace('(a)', 'A.').replace('(b)', 'B.') for part in parts]
    if len(parts) > 1:
        # parts[1] = "(b) " + parts[1]
        parts[1] = "B. " + parts[1]
    for part in parts:
        qs += f"\n{part}"
    qs += f"\n{question_extension}"
    return qs

def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    cfg = setup_cfg(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    if args.dataset_type == 'mevis_variants' or args.dataset_type == "refdavis":
        register_all_mevis(args.dataset_root)
        mapper = MeViSDatasetMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, args.dataset_split, mapper=mapper, num_workers=args.num_workers)
        image_prefix_root =  _PREDEFINED_SPLITS_mevis[args.dataset_split][0]
    else:
        raise NotImplementedError()


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # ---------------------------- prepare model ------------------------------------
    model_args = {
        "train_mask_decoder": False,
        "seg_token_idx": args.seg_token_idx,
    }
    config = UniGRConfig.from_pretrained(
        args.version,
        **model_args,
    )
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    else:
        raise NotImplementedError

    transform = DirectResize(args.image_size)

    model.eval()

    if args.output_dir != "":
        os.makedirs(args.output_dir, exist_ok=True)

    prompt = 'Locate the {}, output its bbox coordinates using JSON format.'

    for idx, inputs in enumerate(tqdm(data_loader)):
        if idx < args.start_index:
            continue

        ######## Setup Meta Information and Prompts
        video_name = inputs[0]['video_name']
        exp = inputs[0]['sentence']
        exp_id = inputs[0]['exp_id']
        frames = inputs[0]['file_names']
        video_len = inputs[0]['length']

        cur_prompt = prompt.format(exp)
        assert args.inference_mode == "video", "This is only meant to infer videos and evaluate their ability to capture motion"
        image_file_list = frames
        total_frames = len(image_file_list)
        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)

        # pre-process images
        frames_list, image_list_sam, image_list_np = [], [], []

        for frm_idx in sparse_idxs:
            image_path = image_file_list[frm_idx]
            image_pil = Image.open(image_path).convert("RGB")
            frames_list.append(image_pil)

        for frm_idx in range(total_frames):
            image_path = image_file_list[frm_idx]
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            image_list_sam.append(image)
            image_list_np.append(image_np)

        # prepare text query and prompt
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames_list, "max_pixels": args.max_pixels},
                {"type": "text", "text": cur_prompt}
            ]}
        ]

        messages += [{"role": "assistant", "content": [
            {"type": "text", "text": "Sure, [SEG]."}  # teacher forcing
            ]}
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = dict_to_cuda(inputs)
        input_ids = inputs['input_ids']

        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        pixel_values = inputs['pixel_values'].bfloat16() if 'pixel_values' in inputs else None
        pixel_values_videos = inputs['pixel_values_videos'].bfloat16() if 'pixel_values_videos' in inputs else None
        image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
        video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
        second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None

        # It only allows for generating segmentation with forced prompting of Sure, SEG. Cant be used with other options
        image_sam = torch.stack(image_list_sam, dim=1)

        output_ids, pred_masks = model.evaluate(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            image_sam,
            resize_list,
            original_size_list,
        )
        save_dir_vid_exp = os.path.join(args.output_dir, video_name, exp_id)
        os.makedirs(save_dir_vid_exp, exist_ok=True)
        for i, pred_mask_vid in enumerate(pred_masks):
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]

            for frame_idx in range(total_frames):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0

                if 'reverse' in args.dataset_split and 'tile' not in args.dataset_split:
                    current_index = len(image_file_list)-frame_idx-1
                else:
                    current_index = frame_idx

                save_path = "{}/{}.png".format(save_dir_vid_exp,
                                               os.path.basename(image_file_list[current_index]).split('.')[0])
                binary_mask = np.where(pred_mask > 0, 1, 0)
                cv2.imwrite(save_path, binary_mask * 255)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main(sys.argv[1:])
