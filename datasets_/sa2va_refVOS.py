import os
import json

import mmengine

from PIL import Image
import copy

from mmengine.dist import master_only

from .base_eval_dataset import BaseEvalDataset
from pycocotools import mask as cocomask
import numpy as np

SEG_PROMPT = "<image>\nPlease segment {}."


class RefVOSDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 special_path=False,
                 meta_info={'split_name': 'MEVIS_U'},
                 multi_instances=False
    ):
        super().__init__()
        vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.mask_dict = mask_dict
        self.text_data = metas
        self.meta_info = meta_info

        self.image_folder = image_folder
        self.special_path = special_path
        self.multi_instances = multi_instances

    def __len__(self):
        return len(self.text_data)

    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, expression_file, mask_file):
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']
        metas = []
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]
            if 'frames' in vid_express_data:
                vid_frames = sorted(vid_express_data['frames'])
                vid_len = len(vid_frames)
            else:
                vid_frames = []
                vid_len = 0
            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']
                meta['exp_id'] = exp_id
                meta['frames'] = vid_frames
                meta['length'] = vid_len
                if 'anno_id' in exp_dict:
                    meta['anno_ids'] = exp_dict['anno_id']
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        if mask_file is not None:
            mask_dict = mmengine.load(mask_file)
        else:
            mask_dict = None
        return vid2metaid, metas, mask_dict

    def _retrieve_masks_from_dict(self, dataset_split, anno_ids, h, w, vid_len, multi_instances):
        gt_masks = []
        for fr_idx in range(vid_len):
            if 'TILE' in dataset_split:
                gt_mask = np.array(np.zeros((h, w//2)), dtype=np.uint8)
            else:
                gt_mask = np.array(np.zeros((h, w)), dtype=np.uint8)

            if 'REVERSE' in dataset_split:
                temp_fr_idx = vid_len - fr_idx - 1
            else:
                temp_fr_idx = fr_idx

            ninst = 1
            for anno_id in anno_ids:
                mask_rle = self.mask_dict[str(anno_id)][temp_fr_idx]
                if mask_rle:
                    if multi_instances:
                        temp_mask = cocomask.decode(mask_rle)
                        gt_mask[temp_mask==1] = ninst
                        ninst += 1
                    else:
                        gt_mask += cocomask.decode(mask_rle)

            empty_mask = np.zeros(gt_mask.shape, np.uint8)
            if 'TILE' in dataset_split:
                if 'REVERSE' in dataset_split:
                    gt_mask = np.concatenate((empty_mask, gt_mask), axis=1)
                else:
                    gt_mask = np.concatenate((gt_mask, empty_mask), axis=1)

            if not multi_instances:
                gt_mask[gt_mask != 0] = 1
            gt_masks.append(gt_mask)

        return gt_masks

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.text_data[index])
        exp = video_obj_info['exp']

        data_dict = {}

        video_id = video_obj_info['video']
        if video_obj_info['length'] != 0:
            frames_files = video_obj_info['frames']
            if self.special_path:
                frames_files = [
                    os.path.join(self.image_folder, video_id, video_obj_info['exp_id'], frame_file + ".jpg") for frame_file in frames_files
                ]
            else:
                frames_files = [
                    os.path.join(self.image_folder,video_id, frame_file + ".jpg") for frame_file in frames_files
                ]
        else:
            frames_path = os.path.join(self.image_folder, video_id)
            frames_files = sorted(os.listdir(frames_path))
            video_obj_info['frames'] = frames_files
            video_obj_info['length'] = len(frames_files)
            frames_files = [os.path.join(frames_path, file_) for file_ in frames_files]

        images = []
        ori_width, ori_height = None, None
        for frame_idx, frame_path in enumerate(frames_files):
            frame_image = Image.open(frame_path).convert('RGB')
            if ori_height is None:
                ori_width, ori_height = frame_image.size
            else:
                assert ori_width == frame_image.size[0]
                assert ori_height == frame_image.size[1]
            images.append(frame_image)

        if self.mask_dict is not None:
            vid_len = len(images)
            w, h = images[0].size
            # Only needed for region-level captioning tasks and only works for MEVIS
            if 'anno_ids' in video_obj_info:
                anno_ids = video_obj_info['anno_ids']
                gt_masks = self._retrieve_masks_from_dict(self.meta_info['split_name'], anno_ids,
                                                          h, w, vid_len, self.multi_instances)
            else:
                gt_masks = None
        else:
            gt_masks = None

        data_dict['type'] = 'video'
        data_dict['index'] = index
        data_dict['video_id'] = video_id
        data_dict['images'] = images
        data_dict['gt_masks'] = gt_masks
        data_dict['exp_id'] = video_obj_info['exp_id']
        data_dict['exp'] = exp

        data_dict['frames'] = video_obj_info['frames']
        data_dict['text_prompt'] = SEG_PROMPT.format(exp) if '?' not in exp else exp
        data_dict['image_folder'] = self.image_folder

        data_dict['length'] = video_obj_info['length']
        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width

        return data_dict
