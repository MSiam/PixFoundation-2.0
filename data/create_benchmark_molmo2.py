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
import torch
import json
from datasets import load_dataset
import glob

import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw

from tqdm import tqdm

def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.single_flag:
#    if not args.reverse_flag or args.single_flag:
        # Load selected keyframes only when single flag
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

    sub_processor(0, args, selected_frames)

def sub_processor(pid, args, selected_frames):
    torch.cuda.set_device(pid)

    # start inference
    num_all_frames = 0
    data_loader = load_dataset("allenai/Molmo2-VideoTrackEval", "animal", split="test")
    total = len(data_loader)  # inference data loader must have a fixed length
    for idx, inputs in enumerate(tqdm(data_loader)):
        video_len = inputs["n_frames"]

        all_pred_masks = []
        difficulty, vidname = inputs['video'].split('_')[:2]
        vid_rest_name= inputs['video'].replace(difficulty+'_'+vidname+'_', '')

        frames = sorted(glob.glob(os.path.join(args.dataset_root, "data", difficulty, vidname, vid_rest_name, "*.jpg")))

        save_path = os.path.join(args.output_dir, inputs["id"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        exp = inputs["exp"]

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
                selected_frame = selected_frames[inputs['id']]
            except:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create MoCentric-Bench on Molmo2TrackEval")
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--frames_sel_file', type=str, default='frames_sel.csv')
    parser.add_argument('--left_flag', action="store_true") # Used for SingleFrame + Concat & Reverse + Concat
    parser.add_argument('--up_flag', action="store_true") # Used for SingleFrame + Concat & Reverse + Concat
    parser.add_argument('--single_flag', action="store_true") # Only single flag no Concat
    parser.add_argument('--reverse_flag', action="store_true")# Switches to Reverse + Concat, default is Single Frame + Concat
    parser.add_argument('--black_flag', action="store_true")
    parser.add_argument('--single_first_flag', action="store_true")
    parser.add_argument('--single_last_flag', action="store_true")
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    main(args)

