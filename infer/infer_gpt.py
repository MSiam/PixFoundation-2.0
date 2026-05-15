import torch
import torchvision
from PIL import Image
import numpy as np
import os
import cv2
import argparse
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import copy
import time
import base64

from detectron2.engine import default_argument_parser

from configs.cfg import setup_cfg
from datasets_.mevis_variants.mevis_dataset_mapper import MeViSDatasetMapper
from datasets_.mevis_variants.mevis_build import build_detection_test_loader, register_all_mevis, \
                                                _PREDEFINED_SPLITS_mevis

from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor

from pycocotools import mask as cocomask
import json
import markdown
from bs4 import BeautifulSoup

import openai
from openai import OpenAI

client = OpenAI(api_key="")

#################################################### Main Functions ##############################
def parse_json(response):
    # Meta response for now is not being used, is used in InternVL Variant not Qwen
    try:
        html = markdown.markdown(response, extensions=['fenced_code'])
        soup = BeautifulSoup(html, 'html.parser')
        json_text = soup.find('code').text
        data = json.loads(json_text)
    except:
        data = json.loads(response)
    return data

def parse_response(response, orig_size, sent_size):
    try:
        bounding_boxes = parse_json(response)
        boxes = []
        for fr_bounding_box in bounding_boxes:
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(fr_bounding_box["bbox_2d"][1]/sent_size[1] * orig_size[1])
            abs_x1 = int(fr_bounding_box["bbox_2d"][0]/sent_size[0] * orig_size[0])
            abs_y2 = int(fr_bounding_box["bbox_2d"][3]/sent_size[1] * orig_size[1])
            abs_x2 = int(fr_bounding_box["bbox_2d"][2]/sent_size[0] * orig_size[0])

            if abs_x1 > abs_x2:
              abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
              abs_y1, abs_y2 = abs_y2, abs_y1

            boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
    except:
        # Occurs when the returned text has no json format bounding boxes at all
        boxes = None

    return boxes

def vis_boxes(final_response, img_path):
    if type(img_path) == str:
        img = np.array(Image.open(img_path))
    else:
        img = np.array(img_path)

    print('#################################', img_path)
    for response in final_response:
        start = response[:2]
        end = response[2:]

        img = cv2.rectangle(img, start, end, (255, 0, 0), 2)
    plt.imshow(img);plt.show()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

NUM_SECONDS_TO_SLEEP = 10
def prompt_gpt(prompt, frames):
    orig_size = None
    sent_size = None

    frames = frames[::2]
    images_base64_dict = []
    for frame in frames:
        img = Image.open(frame)
        w, h = img.size
        orig_size = (w, h)
        scale = 2
        res_img = cv2.resize(np.array(img)[:,:,::-1], (w//scale, h//scale))
        sent_size = (w//scale, h//scale)

        _, buffer_ = cv2.imencode(".jpg", np.array(res_img))
        base64_image = base64.b64encode(buffer_).decode("utf-8")
        images_base64_dict.append(
            {
                "type": "image_url",
                "image_url":  {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        )

    response = None
    ncalls = 0
    while ncalls < 2:
        ncalls += 1
        print('=====', ncalls)
        try:
            response = client.chat.completions.create(
            #response = openai.ChatCompletion.create(
                model='gpt-5',
                    messages=[{
                        'role': 'system',
                        'content': 'You are a helpful and precise assistant for following instructions precisely.'
                        }, {
                        'role': 'user',
                        'content':
                        [{
                            "type": "text",
                            "text": prompt,
                        }
                        ] + images_base64_dict,
                    }],
#                temperature=0.2,
            )
            break
        except openai.RateLimitError:
        #except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    if response is None:
        return '', orig_size, sent_size

    answer = response.choices[0].message.content
    return answer, orig_size, sent_size

def merge_preds(result, result_flip, flip_type, width):
    if result is None:
        return result_flip
    elif result_flip is None:
        return result

    for ridx, _ in enumerate(result_flip):
        x1, x2 = result_flip[ridx][::2]
        if flip_type == 'old':
            result_flip[ridx][0] += width//2
            result_flip[ridx][2] += width//2
        else:
            if x1 < width//2 and x2 < width//2: # First half of image
                result_flip[ridx][0] += width//2
                result_flip[ridx][2] += width//2
            elif x1 > width//2 and x2 > width//2: # second half of image
                result_flip[ridx][0] -= width//2
                result_flip[ridx][2] -= width//2

    merged_result = result + result_flip
    return merged_result

def prompt_sam(image, response, sam_predictor):
    pass

def prompt_sam2(video_dir, responses, sam2_predictor, init_index):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2_predictor.init_state(video_path=video_dir)

    set_track = False
    for oidx, response in enumerate(responses):
        # add new prompts and instantly get the output on the same frame
        input_box = np.array(response)
        sam2_predictor.add_new_points_or_box(state, frame_idx=init_index, obj_id=oidx, box=input_box)

    # propagate the prompts to get masklets throughout the video
    video_segments = [[] for i in range(len(responses))]
    for frame_idx, obj_ids, masks in sam2_predictor.propagate_in_video(state):
        for oid in obj_ids:
            video_segments[oid].append(
                (masks[oid, 0] > 0.0)
            )

    if init_index != 0:
        # Infer the reverse through the previous frames when init frame is not the first frame
        sam2_predictor.reset_state(state)

        for oidx, response in enumerate(responses):
            # add new prompts and instantly get the output on the same frame
            input_box = np.array(response)
            sam2_predictor.add_new_points_or_box(state, frame_idx=init_index, obj_id=oidx, box=input_box)

        # propagate the prompts to get masklets throughout the video
        rev_video_segments = [[] for i in range(len(responses))]
        first = True
        for frame_idx, obj_ids, masks in sam2_predictor.propagate_in_video(state, reverse=True):
            if first:
                # Skip first frame as it is init frame and is redundant
                first = False
                continue
            for oid in obj_ids:
                video_segments[oid].insert(0, (masks[oid, 0] > 0.0))

    return video_segments

def infer_full_video_pipeline(frames, frames_flip, prompt, sam_model=None,
                              sam_type='sam', flip_flag=False, flip_type='', dataset_split=''):

    response, orig_size, sent_size = prompt_gpt(prompt, frames)
    final_response = parse_response(response, orig_size, sent_size)
    original_response = response

    if flip_flag:
        response, orig_size, sent_size = prompt_gpt(prompt, frames_flip)
        original_response += '||' + response
        final_response_flip = parse_response(response, orig_size, sent_size)
        final_response = merge_preds(final_response, final_response_flip, flip_type, orig_size[0])

    try:
        # Infer Masks from SAM
        if sam_type == 'sam':
            # This Mode is only suitable if annotation is meant for Online Referring Expression Segmentation
            width, height = Image.open(frames[0]).size
            all_masks = np.zeros((len(frames), height, width, 3))
            for fridx, boxes in final_response.items():
                all_masks[fridx] = prompt_sam(frames[fridx], boxes, sam_model)

        elif sam_type == 'sam2':
            # The reason for SAM 2.0 is that MeVIS is annotated to track the object through whole video not online based on EXP
            frame = frames[0].split('/')[-1]
            video_dir = frames[0].replace(frame, '')

            init_index = 0
            if dataset_split=='mevis_val_mocentric_reverse':
                init_index = len(frames) - init_index - 1

            sam2_masks = prompt_sam2(video_dir, final_response, sam_model, init_index)
            sam2_masks = [torch.stack(masks) for masks in sam2_masks]
            all_masks = torch.stack(sam2_masks, dim=0)
    except:
        img_shape = Image.open(frames[0]).size
        all_masks = torch.zeros(1, len(frames), *img_shape[::-1])

    return all_masks, response

def inference(cfg, args):
    # Prepare Dataloader
    if args.dataset_type == 'mevis_variants' or args.dataset_type == "refdavis":
        register_all_mevis(args.dataset_root)
        mapper = MeViSDatasetMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, args.dataset_split, mapper=mapper, num_workers=args.num_workers)
        image_prefix_root =  _PREDEFINED_SPLITS_mevis[args.dataset_split][0]
    else:
        raise NotImplementedError()

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.sam_type == 'sam':
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
        sam_model= SamPredictor(sam)
    elif args.sam_type == 'sam2':
        sam_model = build_sam2_video_predictor("sam2_hiera_l.yaml", args.sam_ckpt).cuda()

    prompt = "Given the query {}, detect and localize the visual content described by this textual query and output the bounding box coordinates for all instances of that query in the first frame of the video in JSON format."
    prompt_ext = " Output Format: [{\"bbox_2d\": [x_min, y_min, x_max, y_max], ...}]"

    print("Start inference on {} batches".format(len(data_loader)))

    ans_file = open(os.path.join(args.output_dir, 'bounding_boxes.json'), "w")
    ans_file.write('[')

    for idx, inputs in enumerate(tqdm(data_loader)):
        if idx < args.start_index:
            continue

        ######## Setup Meta Information and Prompts
        video_name = inputs[0]['video_name']
        exp = inputs[0]['sentence']
        exp_id = inputs[0]['exp_id']
        frames = inputs[0]['file_names']
        video_len = inputs[0]['length']

        save_path = os.path.join(args.output_dir, video_name, exp_id)
        os.makedirs(save_path, exist_ok=True)

        if 'tile' in args.dataset_split:
            assert args.image_prefix_root_flip != '', "Image Prefix for the Flipped Tiling is missing"
            frames_flip = [frame.replace(image_prefix_root, args.image_prefix_root_flip) for frame in frames]
        else:
            frames_flip = None

        current_prompt = prompt.format(exp) + prompt_ext
        all_pred_masks, response = infer_full_video_pipeline(frames, frames_flip, current_prompt, sam_model=sam_model,
                                                             sam_type=args.sam_type, flip_flag='tile' in args.dataset_split,
                                                             flip_type=args.flip_type, dataset_split=args.dataset_split)
        print(response)

        ######## Process predicted masks for saving
        # all_pred_masks #Nobjects x #Nframes x H x W, #Nframes is always for the full video
        torch.cuda.empty_cache()
        if len(all_pred_masks) != 0:
            out_masks = all_pred_masks.max(dim=0)[0]
        else:
            img = Image.open(frames[0])
            h, w = np.array(img).shape[:2]
            out_masks = torch.zeros(video_len, h, w)

        for j in range(len(out_masks)):
            frame_name = frames[j]
            mask = out_masks[j].cpu().numpy().astype(np.float32)
            mask[mask > 0] = 1
            mask = Image.fromarray(mask * 255).convert('L')
            file_name = frame_name.split('/')[-1][:-4]
            save_file = os.path.join(save_path, file_name + ".png")
            mask.save(save_file)

        ans_file.write(json.dumps({"id":video_name+"_"+exp_id, "vid_id": video_name, "exp_id": exp_id,
                                   "prediction": response, "caption": exp}) + ",\n")
        ans_file.flush()

    ans_file.write('{}]')
    ans_file.close()

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--temperature', type=float, default=1e-10) # Currently disabled in the Code
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--image_prefix_root_flip', type=str, default='')
    parser.add_argument('--dataset_type', type=str, default='mevis_variants')
    parser.add_argument('--dataset_split', type=str, default='mevis_val', choices= ['mevis_train', 'mevis_val', 'mevis_test',
                                                                                    'mevis_val_mocentric_single',
                                                                                    'mevis_val_mocentric_tile_single',
                                                                                    'mevis_val_mocentric_reverse',
                                                                                    'mevis_val_mocentric_tile_reverse',
                                                                                    'refdavis_val'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--start_index', type=int, default=0)

    parser.add_argument('--sam_type', type=str, default='sam2', choices=['sam', 'sam2'])
    parser.add_argument('--sam_ckpt', type=str, default='')

    parser.add_argument('--flip_type', type=str, default='new')

    args = parser.parse_args()

    cfg = setup_cfg(args)
    inference(cfg, args)

