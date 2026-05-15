import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import cv2
import os
from PIL import Image
import argparse
import json

from detectron2.engine import default_argument_parser

from configs.cfg import setup_cfg

from datasets_.mevis_variants.mevis_dataset_mapper import MeViSDatasetMapper
from datasets_.mevis_variants.mevis_build import build_detection_test_loader,\
                                                register_all_mevis, retrieve_masks_from_dict

def overlay_mask(img, mask, color):
    if mask is None:
        return img

    def PIL2array(img):
        return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    mask_color[mask==1, :] = color

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==1,:] = blended_arr[mask==1,:]
    return img2


if __name__ == "__main__":
    # Testing the pipeline used as is in loading the images and gt masks in both inference and eval
    # Original loader only used in LMPM code to retrieve the tokens from Roberta during its inference,
    # All other models rely on directly loading and processing the images
    # All models' evaluation rely on loading the gt masks and expressions directly
    parser = default_argument_parser()
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--dataset_type', type=str, default='mevis_variants')
    parser.add_argument('--dataset_split', type=str, default='mevis_val', choices= ['mevis_train', 'mevis_val', 'mevis_test',
                                                                                    'mevis_val_mocentric_single',
                                                                                    'mevis_val_mocentric_tile_single',
                                                                                    'mevis_val_mocentric_reverse',
                                                                                    'mevis_val_mocentric_tile_reverse'])
    parser.add_argument('--dataset_mask_path', type=str, default='')
    parser.add_argument('--dataset_exp_path', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--test_original_loader', action='store_true')
    parser.add_argument('--save_vis', action='store_true')
    args = parser.parse_args()

    cfg = setup_cfg(args)

    if args.dataset_type == 'mevis_variants':
        register_all_mevis(args.dataset_root)
        mapper = MeViSDatasetMapper(cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, args.dataset_split, mapper=mapper, num_workers=args.num_workers)

    else:
        raise NotImplementedError()

    if args.dataset_mask_path != '':
            mask_dict = json.load(open(args.dataset_mask_path))
    else:
        mask_dict = None

    if args.dataset_exp_path != '':
        exp_dict = json.load(open(args.dataset_exp_path))['videos']
    else:
        exp_dict = None

    for inputs in tqdm(data_loader):
        video_name = inputs[0]['video_name']
        exp = inputs[0]['sentence']
        exp_id = inputs[0]['exp_id']
        frames = inputs[0]['file_names']
        if 'annotations' in inputs[0]:
            annots = inputs[0]['annotations']

        print('Expression: ', exp)
        os.makedirs(os.path.join(args.out_dir, video_name, exp_id), exist_ok=True)

        if exp_dict is not None:
            vid = exp_dict[video_name]
            if args.dataset_type == 'mevis_variants':
                anno_ids = vid['expressions'][exp_id]['anno_id']
            else:
                anno_ids = []

        for fr_idx, frame in enumerate(frames):
            if args.test_original_loader:
                image = inputs[0]['image'][fr_idx].permute(1,2,0)
                h, w = inputs[0]['height'], inputs[0]['width']
                image = cv2.resize(np.array(image), (w, h))
            else:
                image = np.array(Image.open(frame))
                h, w = image.shape[:2]

            if args.dataset_type == 'mevis_variants':
                if mask_dict is not None:
                    gt_mask = retrieve_masks_from_dict(args.dataset_split, mask_dict, anno_ids,
                                                       fr_idx, h, w, len(frames))
                else:
                    gt_mask = None

            if args.save_vis:
                viz_img = overlay_mask(image, gt_mask, (255,0,0))
                if args.dataset_split == 'mevis_val_mocentric_reverse':
                    new_frame = '%05d.jpg'%(len(frames) - int(frame.split('/')[-1].split('.')[0]) - 1)
                    cv2.imwrite(os.path.join(args.out_dir, video_name, exp_id, new_frame), viz_img[:, :, ::-1])
                else:
                    cv2.imwrite(os.path.join(args.out_dir, video_name, exp_id, frame.split('/')[-1]), viz_img[:, :, ::-1])
