'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
Ref-Davis17 does not support visualize
'''
import argparse
import random
from pathlib import Path
import pandas as pd
import numpy as np
from detectron2.config import CfgNode as CN
import torch
import json

from datasets_.mevis_variants.mevis_dataset_mapper import MeViSDatasetMapper
from datasets_.mevis_variants.mevis_build import build_detection_test_loader, register_all_mevis
from configs.cfg import setup_cfg

from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)

import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw

from tqdm import tqdm

def main(args):
    cfg = setup_cfg(args)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    register_all_mevis(args.dataset_root)

    if args.frames_sel_file != '':
        # Load selected keyframes
        if 'csv' in args.frames_sel_file:
            selected_frames = pd.read_csv(args.frames_sel_file)
            selected_frames['exp_id_str'] = selected_frames['exp_id'].apply(str)
            selected_frames['new_index'] = selected_frames['video_name'].str.cat(selected_frames['exp_id_str'], sep='_')
            selected_frames = selected_frames.set_index('new_index')
            selected_frames = selected_frames.to_dict(orient='index')

        elif 'jsonl' in args.frames_sel_file:
            selected_frames = {}
            with open(args.frames_sel_file, 'r') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    video_exp = data['video_name'] + '_' + data['exp_id']
                    selected_frames[video_exp] = data
    else:
        selected_frames = None

    # Code supports multiprocessing, but temporarily I disabled MP
    sub_processor(0, cfg, args, selected_frames)

def sub_processor(pid, cfg, args, selected_frames):
    torch.cuda.set_device(pid)

    # start inference
    num_all_frames = 0
    mapper = MeViSDatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, args.dataset_split, mapper=mapper, num_workers=0)
    text = 'processor %d' % pid
    progress = tqdm(
        total=len(data_loader),
        position=pid,
        desc=text,
        ncols=0
    )

    videos_done = []
    if os.path.exists(args.output_dir):
        videos_done = os.listdir(args.output_dir)

    total = len(data_loader)  # inference data loader must have a fixed length
    for idx, inputs in enumerate(tqdm(data_loader)):
        video_len = inputs[0]["length"]

        all_pred_masks = []

        if args.reverse_flag or args.single_first_flag or args.single_last_flag or args.black_flag:
            # Only in the reverse flag or single first frame flag we don need separate video per expr
            save_path = os.path.join(args.output_dir, inputs[0]["video_name"])
            if inputs[0]["video_name"] in videos_done:
                continue
            videos_done.append(inputs[0]["video_name"])
        else:
            save_path = os.path.join(args.output_dir, inputs[0]["video_name"], inputs[0]["exp_id"])

        frames = inputs[0]['file_names']
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_name = inputs[0]["video_name"]
        exp = inputs[0]["sentence"]
        exp_id = inputs[0]["exp_id"]

        video_len = len(frames)

        all_pred_logits = []
        all_pred_masks = []


        if args.reverse_flag:
            reverse_frames = frames[::-1]
        elif args.single_first_flag:
            selected_image = np.array(Image.open(frames[0]))
        elif args.single_last_flag:
            selected_image = np.array(Image.open(frames[-1]))
        else:
            try:
                selected_frame = selected_frames[video_name+'_'+exp_id]
                assert selected_frame['video_name']==video_name, "wrong video"
                selected_frame['frame'] = os.path.join(args.dataset_root, selected_frame['frame'])
            except:
                # In case of failures in the keyframe selection itself, rare occurrence
                selected_frame = {'frame': frames[0]}

            selected_image = np.array(Image.open(selected_frame['frame']))

        for fidx, frame in tqdm(enumerate(frames)):
            image = np.array(Image.open(frame))

            if args.single_flag:
                augmented_image = selected_image
            elif args.black_flag:
                black_image = np.zeros(image.shape, dtype=np.uint8)
                if args.left_flag:
                    augmented_image = np.concatenate((black_image, image), axis=1)
                else:
                    augmented_image = np.concatenate((image, black_image), axis=1)
            elif args.reverse_flag:
                reverse_image = np.array(Image.open(reverse_frames[fidx]))

                if args.left_flag:
                    augmented_image = np.concatenate((reverse_image, image), axis=1)
                else:
                    augmented_image = np.concatenate((image, reverse_image), axis=1)
            else:
                if args.left_flag:
                    augmented_image = np.concatenate((selected_image, image), axis=1)
                elif args.up_flag:
                    augmented_image = np.concatenate((selected_image, image), axis=0)
                else:
                    augmented_image = np.concatenate((image, selected_image), axis=1)

            file_name = frame.split('/')[-1][:-4]
            save_file = os.path.join(save_path, file_name + ".jpg")
            Image.fromarray(augmented_image).save(save_file)

        progress.update(1)

    progress.close()

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--dataset_split', type=str, default='mevis_val')
    parser.add_argument('--frames_sel_file', type=str, default='')
    parser.add_argument('--left_flag', action="store_true") # Used for SingleFrame + Concat & Reverse + Concat
    parser.add_argument('--up_flag', action="store_true") # Used for SingleFrame + Concat & Reverse + Concat
    parser.add_argument('--single_flag', action="store_true") # Used for single static frame only without Conct
    parser.add_argument('--reverse_flag', action="store_true")# Switches to Reverse + Concat, default is Single Frame + Concat
    parser.add_argument('--black_flag', action="store_true")
    parser.add_argument('--single_first_flag', action="store_true")
    parser.add_argument('--single_last_flag', action="store_true")
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    main(args)

