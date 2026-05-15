###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import copy
import pandas as pd
import os
import time
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as cocomask
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
from tqdm import tqdm

def harmonic_mean(a, b):
    return 2.0/(1.0/(a+1e-10)+1.0/(b+1e-10))


def eval_queue(q, rank, out_dict, mevis_pred_path, selected_frames):


    text = 'processor %d' % 0
    progress = tqdm(
        total=q.qsize(),
        position=0,
        desc=text,
        ncols=0
    )


    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{mevis_pred_path}/{vid_name}/{exp}'):
            print(f'{vid_name}/{exp} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f'{mevis_pred_path}/{vid_name}/{exp}/00000.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])

        if args.tile_flag:
            if not args.up_flag:
                gt_masks = np.zeros((vid_len, h, w//2), dtype=np.uint8)
            else:
                gt_masks = np.zeros((vid_len, h//2, w), dtype=np.uint8)
        else:
            gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            if args.reverse_flag:
                temp_frame_idx = vid_len - frame_idx - 1
            else:
                temp_frame_idx = frame_idx

            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][temp_frame_idx]

                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_masks[frame_idx] = cv2.imread(f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)

        j_computed = False
        bg_eval= False # The standard is to evaluate the foreground except in the Single Frame evaluation its Bg
        # Tile + Reverse probing
        if args.reverse_flag and args.tile_flag:
                # In the reverse flag since we use reverse motion expression, mask aligns with left_flag
                empty_mask = np.zeros(gt_masks.shape[-2:], np.uint8)
                if args.left_flag:
                    gt_masks = np.array([np.concatenate((mask, empty_mask), axis=1) for mask in  gt_masks])
                elif args.up_flag:
                    gt_masks = np.array([np.concatenate((empty_mask, mask), axis=0) for mask in  gt_masks])
                else:
                    gt_masks = np.array([np.concatenate((empty_mask, mask), axis=1) for mask in  gt_masks])

        # Tile + Single Frame probing
        elif args.single_frame_flag and args.tile_flag:
            selected_mask = np.zeros(gt_masks.shape[-2:], np.uint8)

            # In the single frame flag since we use original motion expression, mask aligns with opposite of left_flag
            if args.left_flag:
                gt_masks = np.array([np.concatenate((selected_mask, mask), axis=1) for mask in  gt_masks])
            elif args.up_flag:
                gt_masks = np.array([np.concatenate((selected_mask, mask), axis=0) for mask in  gt_masks])
            else:
                gt_masks = np.array([np.concatenate((mask, selected_mask), axis=1) for mask in  gt_masks])

            if args.single_frame_static_side_only:
                half = gt_masks.shape[2] // 2

                if args.left_flag:
                    gt_masks = gt_masks[:, :, :half]
                    pred_masks = pred_masks[:, :, :half]
                else:
                    gt_masks = gt_masks[:, :, half:]
                    pred_masks = pred_masks[:, :, half:]

                pred_masks_temp = pred_masks
                pred_masks_temp[pred_masks_temp==255] = 1
                j = (pred_masks_temp.sum(axis=1).sum(axis=1) / (pred_masks_temp.shape[1] * pred_masks_temp.shape[2])).mean()
                j_computed = True

        # Single Frame only probing no tiling
        elif args.single_frame_flag and not args.tile_flag:
            if vid_name+'_'+exp in selected_frames:
                selected_frame = int(selected_frames[vid_name+'_'+exp]['frame'].split('/')[-1].split('.')[0])
            else:
                selected_frame = 0

            selected_mask = gt_masks[selected_frame]

            # Whatever is background in the groundtruth mask I can ignore in both
            selected_mask[selected_mask==0] = 255

            # Whatever is foreground in the groundtruth mask is whats important to be identied as Bg (Single Frame No motion-> Bg)
            # Models can be confused specifically by this area hence why focus the Bg evaluation only on this to avoid being overwhelmed by other Bg pixels
            selected_mask[selected_mask==1] = 0
            gt_masks = np.stack([selected_mask for mask in gt_masks], axis=0)
            bg_eval = True

            pred_masks_temp = copy.deepcopy(pred_masks)
            pred_masks_temp[pred_masks_temp==255] = 1
            fpr = (pred_masks_temp.sum(axis=1).sum(axis=1) / (pred_masks_temp.shape[1] * pred_masks_temp.shape[2])).mean()

            pred_masks_temp_2 = copy.deepcopy(pred_masks_temp)
            pred_masks_temp_2[gt_masks!=0] = 0
            fg_fpr = ((pred_masks_temp_2==1).sum(axis=1).sum(axis=1) / ((gt_masks==0).sum(axis=1).sum(axis=1) + 1e-10)).mean()
            f = [fpr, fg_fpr]

        if not j_computed:
            j = db_eval_iou(gt_masks, pred_masks, void_pixels=(gt_masks==255), bg_eval=bg_eval).mean()

        if args.single_frame_flag and not args.tile_flag:
            gt_masks_ = copy.deepcopy(gt_masks)
            gt_masks_[gt_masks_==255] = 0
            jbg_all = db_eval_iou(gt_masks_, pred_masks, void_pixels=(gt_masks_==255), bg_eval=True).mean()
            jbg_hmean_all = harmonic_mean(j, jbg_all)
            j = [j, jbg_hmean_all]
        else:
            f = db_eval_boundary(gt_masks, pred_masks, void_pixels=(gt_masks==255), bg_eval=bg_eval).mean()
        out_dict[exp_name] = [j, f]

        progress.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute Evaluation Metrics on MeVIS variants")
    parser.add_argument("--mevis_exp_path", type=str, default="datasets/mevis/valid_u/meta_expressions.json")
    parser.add_argument("--mevis_mask_path", type=str, default="datasets/mevis/valid_u/mask_dict.json")
    parser.add_argument("--mevis_pred_path", type=str, default="output/mevis/inference")

    parser.add_argument("--save_name", type=str, default="mevis_test.json")
    parser.add_argument("--frames_selection_file", type=str, default="keyframes.csv")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--left_flag", action="store_true")
    parser.add_argument("--single_frame_flag", action="store_true")
    parser.add_argument("--single_frame_static_side_only", action="store_true")
    parser.add_argument("--reverse_flag", action="store_true")
    parser.add_argument("--tile_flag", action="store_true")
    parser.add_argument("--up_flag", action="store_true")

    parser.add_argument("--parallel_threads", action="store_true")
    parser.add_argument("--filterout_static_exprs", action="store_true")
    args = parser.parse_args()

    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))['videos']
    mask_dict = json.load(open(args.mevis_mask_path))

    if args.parallel_threads:
        shared_exp_dict = mp.Manager().dict(exp_dict)
        shared_mask_dict = mp.Manager().dict(mask_dict)
        output_dict = mp.Manager().dict()
    else:
        shared_exp_dict = {}
        shared_mask_dict = {}
        output_dict = {}

    static_exprs = {}
    if args.filterout_static_exprs:
        with open('../data/static_expressions.json', 'r') as f:
            static_exprs = json.load(f)

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            if vid_name+'_'+exp not in static_exprs:
                queue.put([vid_name, exp])

    # Load keyframes in case I want to perform analysis on the amount of false positives
    if args.frames_selection_file != 'None':
        selected_frames = pd.read_csv(args.frames_selection_file)
        selected_frames['exp_id_str'] = selected_frames['exp_id'].apply(str)
        selected_frames['new_index'] = selected_frames['video_name'].str.cat(selected_frames['exp_id_str'], sep='_')
        selected_frames = selected_frames.set_index('new_index')
        selected_frames = selected_frames.to_dict(orient='index')
    else:
        selected_frames = None

    if args.parallel_threads:
        processes = []
        for rank in range(args.num_workers):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mevis_pred_path, selected_frames))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict, args.mevis_pred_path, selected_frames)

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    if args.single_frame_flag and not args.tile_flag:
        jbg = [output_dict[x][0][0] for x in output_dict]
        jbg_hmean_all = [output_dict[x][0][1] for x in output_dict]

        fpr = [output_dict[x][1][0] for x in output_dict]
        fg_fpr = [output_dict[x][1][1] for x in output_dict]

        print(f'FPR: {np.mean(fpr)}')
        print(f'FG FPR: {np.mean(fg_fpr)}')
        print(f'Jbg: {np.mean(jbg)}')
        print(f'Jbg Hmean all: {np.mean(jbg_hmean_all)}')
    else:
        j = [output_dict[x][0] for x in output_dict]
        f = [output_dict[x][1] for x in output_dict]

        print(f'J: {np.mean(j)}')
        print(f'F: {np.mean(f)}')
        print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

