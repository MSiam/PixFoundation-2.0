import argparse
import json
import os

import mmengine
import numpy as np
from PIL import Image

import cv2
import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer

from projects.llava_sam2.evaluation.dataset import RefVOSDataset
from projects.llava_sam2.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'DAVIS': {
        'data_root': 'davis17/',
        'image_folder': 'davis17/valid/JPEGImages/',
        'expression_file': 'davis17/meta_expressions/valid/meta_expressions.json',
        'mask_file': 'davis17/valid/mask_dict.pkl',
    },
    'MEVIS': {
        'data_root': 'valid/',
        'image_folder': 'valid/JPEGImages',
        'expression_file': 'valid/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U': {
        'data_root': 'valid_u/',
        'image_folder': 'valid_u/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': 'valid_u/mask_dict.json',
    },
    ################################ MoCentric-Bench ###############################
    'MEVIS_MOCENTRIC_TILE_SINGLE': {
        'data_root': 'valid_u_mocentric_tile_single/',
        'image_folder': 'valid_u_mocentric_tile_single/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': True
    },
    'MEVIS_MOCENTRIC_TILE_SINGLE_LEFT': {
        'data_root': 'valid_u_mocentric_tile_single_left/',
        'image_folder': 'valid_u_mocentric_tile_single_left/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': True
    },
    'MEVIS_MOCENTRIC_TILE_SINGLE_UP': {
        'data_root': 'valid_u_mocentric_tile_single_up/',
        'image_folder': 'valid_u_mocentric_tile_single_up/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': True
    },
    'MEVIS_MOCENTRIC_TILE_BLACK': {
        'data_root': 'valid_u_mocentric_tile_black/',
        'image_folder': 'valid_u_mocentric_tile_black/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': False
    },
    'MEVIS_MOCENTRIC_TILE_BLACK_LEFT': {
        'data_root': 'valid_u_mocentric_tile_black_left/',
        'image_folder': 'valid_u_mocentric_tile_black_left/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': False
    },
    'MEVIS_MOCENTRIC_SINGLE': {
        'data_root': 'valid_u_mocentric_single/',
        'image_folder': 'valid_u_mocentric_single/JPEGImages',
        'expression_file': 'valid_u/meta_expressions.json',
        'mask_file': None,
        'special_path': True
    },
    'MEVIS_MOCENTRIC_REVERSE': {
        'data_root': 'valid_u/',
        'image_folder': 'valid_u/JPEGImages',
        'expression_file': 'valid_u/meta_expressions_reverse_filtered.json',
        'mask_file': None,
        'special_path': False
    },
    'MEVIS_MOCENTRIC_TILE_REVERSE': {
        'data_root': 'valid_u_mocentric_tile_reverse/',
        'image_folder': 'valid_u_mocentric_tile_reverse/JPEGImages',
        'expression_file': 'valid_u/meta_expressions_reverse_filtered.json',
        'mask_file': None,
        'special_path': False
    },
    'MEVIS_MOCENTRIC_TILE_REVERSE_LEFT': {
        'data_root': 'valid_u_mocentric_tile_reverse_left/',
        'image_folder': 'valid_u_mocentric_tile_reverse_left/JPEGImages',
        'expression_file': 'valid_u/meta_expressions_reverse_filtered.json',
        'mask_file': None,
        'special_path': False
    },

    ################################ Other RefVOS Datasets ###########################
    'REFYTVOS': {
        'data_root': 'rvos/',
        'image_folder': 'rvos/valid/JPEGImages/',
        'expression_file': 'rvos/meta_expressions/valid/meta_expressions.json',
        'mask_file': None,
    },
    'REVOS': {
        'data_root': 'revos/',
        'image_folder': 'revos/',
        'expression_file': 'revos/meta_expressions_valid_.json',
        'mask_file': None,
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument('--dataset_root', type=str, default='data/')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()
    return args

def merge(result, result_flip):
    merged_result = {'prediction': result['prediction'], 'prediction_masks': []}
    for pred, pred_flip in zip(result['prediction_masks'], result_flip['prediction_masks']):
        t, h, w = pred.shape
        pred_ = np.concatenate((pred_flip[:, :, w//2:], pred_flip[:, :, :w//2]), axis=2)
        merged_pred = pred_ +  pred
        merged_result['prediction_masks'].append(merged_pred)
    return merged_result

def infer_with_flip(model, item, item_flip, tokenizer, flip):
    result = model.predict_forward(
        video=item['images'],
        text=item['text_prompt'],
        tokenizer=tokenizer,
    )

    if flip:
        result_flip = model.predict_forward(
            video=item_flip['images'],
            text=item_flip['text_prompt'],
            tokenizer=tokenizer,
        )
        result = merge(result, result_flip)
    return result

if __name__ == '__main__':
    args = parse_args()

    work_dir = args.work_dir

    model = AutoModel.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]
    for key in ['data_root', 'image_folder', 'expression_file', 'mask_file']:
        if dataset_info[key] is not None:
            dataset_info[key] = os.path.join(args.dataset_root, dataset_info[key])

    dataset = RefVOSDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
        special_path=dataset_info['special_path'] if 'special_path' in dataset_info else False
    )
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda x:x[0],
    )

    if args.flip:
        dataset_flip = RefVOSDataset(
            image_folder=dataset_info['image_folder'].replace('/JPEGImages', '')+'_left/JPEGImages',
            expression_file=dataset_info['expression_file'],
            mask_file=dataset_info['mask_file'],
            special_path=dataset_info['special_path'] if 'special_path' in dataset_info else False
        )
        sampler_flip = torch.utils.data.SequentialSampler(dataset_flip)
        dataloader_flip = torch.utils.data.DataLoader(
            dataset_flip,
            sampler=sampler_flip,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=lambda x:x[0],
        )
        data_iterator_flip = iter(dataloader_flip)

    results = []

    for item in tqdm.tqdm(dataloader):
        if 'REVERSE' in args.dataset and 'TILE' not in args.dataset:
            item['images'] = item['images'][::-1]

        if args.flip:
            item_flip = next(data_iterator_flip)
        else:
            item_flip = None

        with torch.no_grad():
            result = infer_with_flip(model, item, item_flip, tokenizer, args.flip)

        text_idx = 0
        text_prediction = result['prediction']
        if len(result['prediction_masks']) > 0:
            mask_prediction = result['prediction_masks'][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        save_dir = os.path.join(args.work_dir, item['video_id'], item['exp_id'])
        os.makedirs(save_dir, exist_ok=True)

        for mask, frame_name in zip(mask_prediction, item['frames']):
            mask = mask * 255
            frame_name = frame_name.split('.')[0]
            cv2.imwrite(os.path.join(save_dir, '%s.png'%frame_name), np.array(mask, np.uint8))

        if 'MOCENTRIC' not in args.dataset:
            encoded_mask = mask_to_rle(mask_prediction)

            result = {
                'index': item['index'],
                'video_id': item['video_id'],
                'exp_id': item['exp_id'],
                'text_prediction': text_prediction,
                'frames': item['frames'],
                'exp': item['text_prompt'],
                'prediction_masks': encoded_mask,

            }
            results.append(result)

    if 'MOCENTRIC' not in args.dataset:
        final_results = {}
        for item in results:
            vid_id = item['video_id']
            exp_id = item['exp_id']
            if vid_id not in final_results:
                final_results[vid_id] = {}
            assert exp_id not in final_results[vid_id]
            final_results[vid_id][exp_id] = item
        os.makedirs(work_dir, exist_ok=True)
        json.dump(final_results, open(f'{work_dir}/results.json', 'w'))
