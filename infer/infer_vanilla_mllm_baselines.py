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

from detectron2.engine import default_argument_parser

from configs.cfg import setup_cfg
from datasets_.mevis_variants.mevis_dataset_mapper import MeViSDatasetMapper
from datasets_.mevis_variants.mevis_build import build_detection_test_loader, register_all_mevis, \
                                                _PREDEFINED_SPLITS_mevis

from qwen_vl_utils import process_vision_info

from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor

import json
import markdown
from bs4 import BeautifulSoup

#################################################### Helper Functions ##############################
def parse_json(response):
    # Meta response for now is not being used, is used in InternVL Variant not Qwen
    html = markdown.markdown(response, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    json_text = soup.find('code').text
    data = json.loads(json_text)
    return data

def retrieve_closest_timestamp(tstamp, timestamps):
    if tstamp in timestamps:
        return tstamp

    closest_timestamp = None
    sorted_timestamps = sorted(list(timestamps.keys()))
    for idx, curr_stamp in enumerate(sorted_timestamps):
        if tstamp > curr_stamp and idx == len(sorted_timestamps)-1:
            closest_timestamp = curr_stamp
            break
        elif tstamp < curr_stamp and idx == 0:
            closest_timestamp = curr_stamp
            break
        elif tstamp < sorted_timestamps[idx+1] and tstamp > curr_stamp:
            if abs(tstamp-curr_stamp) < abs(tstamp-sorted_timestamps[idx+1]):
                closest_timestamp = curr_stamp
            else:
                closest_timestamp = sorted_timestamps[idx+1]
            break
    return closest_timestamp


def parse_json_tstamp_qwen3(response, nframes):
    # Meta response for now is not being used, is used in InternVL Variant not Qwen
    try:
        response = parse_json(response)
    except:
        response = json.loads(response)

    video_fps = 24
    merge_size = 2
    min_frames = 4
    max_frames = 768
    num_sampled_frames = int(nframes / video_fps * 2)
    num_sampled_frames = min(min(max(num_sampled_frames, min_frames), max_frames), nframes)
    indices = np.linspace(0, nframes - 1, num_sampled_frames).round().astype(int)
    timestamps = [idx / video_fps for idx in indices]
    # Original Q3VL uses merge_size of 2 and skipping every other index, I preferred more dense sampling
    timestamps = {(timestamps[i] + timestamps[i + merge_size - 1]) / 2: indices[i] for i in range(0, len(timestamps)-1)}

    for k, v in response.items():
        closest_timestamp = retrieve_closest_timestamp(float(v), timestamps)
        response[k] = timestamps[closest_timestamp]
    return response


def parse_response(response, meta_response):
    input_height, input_width, (width, height) = meta_response
    try:
        bounding_boxes = parse_json(response)
        boxes = []
        for bounding_box in bounding_boxes:
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
            abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
            abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
            abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

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
            # FIXME: I dont need to reset states and add the prompts again
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


#################################################### Main Functions ##############################
def image_inference(img_url, prompt, model, processor, system_prompt="You are a helpful assistant",
                    max_new_tokens=1024, temperature=0, mllm_type='Qwen2.5'):

    image = Image.open(img_url)
    messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=1024)#, temperature=temperature)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    return output_text[0], (input_height, input_width, image.size)

def keyframes_selection(video_frames_path, prompt, model, processor, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28,
                        reverse_flag=False, mllm_type='Qwen2.5'):

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_frames_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    skip = 1
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)

    video_len = len(video_inputs[0])
    half = video_len//2
    if not reverse_flag:
        video_inputs[0] = video_inputs[0][half:]
    else:
        video_inputs[0] = video_inputs[0][:half]

    transform = torchvision.transforms.Resize(400)
    video_inputs[0] = [transform(frame) for frame in video_inputs[0]]
    if type(video_inputs[0][0]) == Image.Image:
        # Qwen2.5 environment
        video_inputs[0] = torch.stack([torch.as_tensor(np.array(frame, np.uint8)).permute(2,0,1) for frame in video_inputs[0]])
    else:
        # Qwen3 environment
        video_inputs[0] = torch.stack(video_inputs[0])

    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))

    if mllm_type == 'Qwen2.5':
        fps_inputs = video_kwargs['fps']
        inputs = processor(text=[text], images=image_inputs, videos=[video_inputs[0][::skip]], fps=fps_inputs, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text], images=image_inputs, videos=[video_inputs[0][::skip]], return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    if mllm_type == 'Qwen2.5':
        data = parse_json(output_text[0])
        if type(data) == list:
            start = int(float(data[0]["start_time"])/fps_inputs[0] * (skip * fps_inputs[0]))
            end = int(float(data[0]["end_time"])/fps_inputs[0] * (skip * fps_inputs[0]))
        else:
            start = int(float(data["start_time"])/fps_inputs[0] * (skip * fps_inputs[0]))
            end = int(float(data["end_time"])/fps_inputs[0] * (skip * fps_inputs[0]))
    else:
        data = parse_json_tstamp_qwen3(output_text[0], num_frames)
        if type(data) == list:
            data = data[0]

        print("========>", data)
        try:
            start = data["first_second"] * skip
            end = data["last_second"] * skip
        except:
            start = data["first"] * skip
            end = data["last"] * skip

    keyframes = list(range(start+half, end+half))
    keyframes = [k for k in keyframes if k < video_len and k > 0]
    middle_keyframe = keyframes[len(keyframes)//2]
    keyframes.remove(middle_keyframe)
    return output_text[0], [middle_keyframe] + keyframes

def infer_current_frame(args, init_index, frames, frames_flip, prompt, model, processor, flip_flag=False,
                        flip_type='new', mllm_type='Qwen2.5'):
    init_frame = frames[init_index]
    response, meta_response = image_inference(init_frame, prompt, model, processor, temperature=args.temperature,
                                              mllm_type=mllm_type)
    final_response = parse_response(response, meta_response)
    ############################## DEBUGGING ONLY ###################################
    if args.debug:
        f = open('track_responses_vids_neg.txt', 'a')
        f.write('Frame: ' + init_frame + '|\n Response: '+ response + '|\n' + ' Final Response: ' + str(final_response) + '\n')
        f.close()
    ############################## DEBUGGING ONLY ###################################

    if flip_flag:
        # Infer flipped to accomodate for location bias
        init_frame_flipped = frames_flip[init_index]
        response_flipped, meta_response = image_inference(init_frame_flipped, prompt, model, processor, temperature=args.temperature,
                                                          mllm_type=mllm_type)
        final_response_flipped = parse_response(response_flipped,  meta_response)

        if final_response_flipped is not None:
            width, height = meta_response[2]
            # Flip bounding box locations to account for the flipping to the left
            for ridx, _ in enumerate(final_response_flipped):
                x1, x2 = final_response_flipped[ridx][::2]
                if flip_type == 'old':
                    final_response_flipped[ridx][0] += width//2
                    final_response_flipped[ridx][2] += width//2
                else:
                    if x1 < width//2 and x2 < width//2: # First half of image
                        final_response_flipped[ridx][0] += width//2
                        final_response_flipped[ridx][2] += width//2
                    elif x1 > width//2 and x2 > width//2: # second half of image
                        final_response_flipped[ridx][0] -= width//2
                        final_response_flipped[ridx][2] -= width//2

        if final_response is not None:
            final_response = final_response + final_response_flipped
        else:
            final_response = final_response_flipped

    return final_response

def infer_sam2(args, frames, frames_flip, model, processor, prompt, sam2_predictor,
               meta_info, flip_flag=False, reverse_idx4sam2_flag=False, flip_type='new', mllm_type='Qwen2.5'):
    selected_frames = {'video_name': [], 'frame': [], 'exp_id': []}

    video_len = len(frames)
    frame = frames[0].split('/')[-1]
    video_dir = frames[0].replace(frame, '')

    final_response = None

    if args.frames_selection == 'standard':
        # Try each frame from the beginning of the video till I find good initialization to SAM
        for init_index in range(video_len):
            try: # This to account for errors in the parsing of the returned response
                final_response = infer_current_frame(args, init_index, frames, frames_flip,
                                                     prompt, model, processor, flip_flag, flip_type, mllm_type)
                if final_response is None: # This is also occurs during errors in parsing the bounding boxes
                    continue
                if args.keyframes_save_flag:
                    selected_frames["video_name"].append(meta_info[0])
                    selected_frames["exp_id"].append(meta_info[1])
                    selected_frames["frame"].append(frames[init_index])
                break
            except:
                pass

    elif args.frames_selection == 'keyframes':
        # Try each frame in the keyframes retrieved from the video till I find good init to SAM 2
        prompt, keyframe_prompt = prompt
        try: # To account for problems in the parsing of the keyframes revert to reverse procedure
            output_text, keyframes = keyframes_selection(frames, keyframe_prompt, model, processor,
                                                         reverse_flag=('reverse' in args.dataset_split),
                                                         mllm_type=args.mllm_type)
            if 'reverse' in args.dataset_split:
                #FIXME: Test this change across all probing techniques
                all_frames = list(range(video_len))[::-1]
                for frame in keyframes:
                    all_frames.remove(frame)
                keyframes += all_frames
        except:
            print("=============================> Failed Keyrfames Selection")
            keyframes = list(range(video_len))[::-1]

        for init_index in keyframes:
            if init_index >= video_len:
                # This to account for any errors in retrieved keyframes from LLM
                break
            try: # This to account for errors in the parsing of the returned response
                final_response = infer_current_frame(args, init_index, frames, frames_flip,
                                                     prompt, model, processor, flip_flag, flip_type, mllm_type)
                if final_response is None: # This also occurs during errors in parsing the bounding boxes
                    continue
                if args.keyframes_save_flag:
                    selected_frames["video_name"].append(meta_info[0])
                    selected_frames["exp_id"].append(meta_info[1])
                    selected_frames["frame"].append(frames[init_index])
                break
            except:
                pass

    if final_response is not None and len(final_response) != 0:
        if reverse_idx4sam2_flag:
            init_index = len(frames) - init_index - 1
        all_pred_masks = prompt_sam2(video_dir, final_response, sam2_predictor, init_index)
    else:
        img = Image.open(frames[0])
        h, w = np.array(img).shape[:2]
        all_pred_masks = [[torch.zeros(h, w).cuda() for i in range(video_len)]]

    return all_pred_masks, selected_frames

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

    # Prepare Model/s
    if args.mllm_type == 'Qwen2.5':
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.mllm_checkpoint, device_map='cuda', torch_dtype=torch.bfloat16,
                                                               attn_implementation="flash_attention_2", local_files_only=True)
        processor = AutoProcessor.from_pretrained(args.mllm_checkpoint, local_files_only=True)
    elif args.mllm_type == 'Qwen3':
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(args.mllm_checkpoint, local_files_only=True)
        model, output_loading_info = AutoModelForVision2Seq.from_pretrained(args.mllm_checkpoint, torch_dtype="auto", device_map="auto",
                                                                        output_loading_info=True, local_files_only=True)


    model.eval()
    sam2_predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", args.sam_ckpt).cuda()

    # Setup prompt
    prompt = 'Locate the {}, output its bbox coordinates using JSON format.'
    print("Start inference on {} batches".format(len(data_loader)))
    if args.keyframes_save_flag:
        full_selected_frames = {'video_name': [], 'frame': [], 'exp_id': []}

    ############################## DEBUGGING ONLY ###################################
    if args.debug:
        vids_neg = np.load('vids_neg.npy', allow_pickle=True)

    for idx, inputs in enumerate(tqdm(data_loader)):
        if idx < args.start_index:
            continue

        ######## Setup Meta Information and Prompts
        video_name = inputs[0]['video_name']
        exp = inputs[0]['sentence']
        exp_id = inputs[0]['exp_id']
        frames = inputs[0]['file_names']
        video_len = inputs[0]['length']

        ############################## DEBUGGING ONLY ###################################
        if args.debug:
            if not (video_name+'_'+exp_id in vids_neg):
                continue
        ############################## DEBUGGING ONLY ###################################

        save_path = os.path.join(args.output_dir, video_name, exp_id)
        os.makedirs(save_path, exist_ok=True)

        print("==================> Index: ", idx)

        all_pred_masks = []
        if 'tile' in args.dataset_split:
            assert args.image_prefix_root_flip != '', "Image Prefix for the Flipped Tiling is missing"
            frames_flip = [frame.replace(image_prefix_root, args.image_prefix_root_flip) for frame in frames]
        else:
            frames_flip = None

        current_prompts = prompt.format(exp)
        if args.frames_selection == "keyframes":
            keyframe_prompt = f"Given the query: {exp}, when does the described content occur near the end of the video? \
                                Output the first and last seconds for this action in json format."
            current_prompts = [current_prompts, keyframe_prompt]

        ######## Infer Qwen2.5-VL + SAM 2.0
        all_pred_masks, selected_frames = infer_sam2(args, frames, frames_flip, model, processor, current_prompts,
                                                     sam2_predictor, meta_info=(video_name, exp_id),
                                                     flip_flag=('tile' in args.dataset_split),
                                                     reverse_idx4sam2_flag=(args.dataset_split=='mevis_val_mocentric_reverse'),
                                                     flip_type=args.flip_type, mllm_type=args.mllm_type)
        if args.keyframes_save_flag:
            full_selected_frames = {k: v + selected_frames[k] for k, v in full_selected_frames.items()}


        ######## Process predicted masks for saving
        torch.cuda.empty_cache()
        if len(all_pred_masks) != 0:
            out_masks = torch.zeros((len(all_pred_masks[0]), *all_pred_masks[0][0].shape))
            for pred_masks in all_pred_masks:
                pred_masks = torch.stack(pred_masks, dim=0)
                out_masks[pred_masks==1] = 1
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

    if args.keyframes_save_flag:
        selected_df = pd.DataFrame.from_dict(full_selected_frames)
        selected_df.to_csv(args.frames_selection_file)

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--mllm_type', type=str, default='Qwen2.5')
    parser.add_argument('--mllm_checkpoint', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--temperature', type=float, default=1e-10) # Currently disabled in the Code
    parser.add_argument('--sam_ckpt', type=str, default='sam2_hiera_large.pt')

    parser.add_argument('--flip_type', type=str, default='new')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--dataset_type', type=str, default='mevis_variants')
    parser.add_argument('--image_prefix_root_flip', type=str, default='')
    parser.add_argument('--dataset_split', type=str, default='mevis_val', choices= ['mevis_train', 'mevis_val', 'mevis_test',
                                                                                    'mevis_val_mocentric_single',
                                                                                    'mevis_val_mocentric_tile_single',
                                                                                    'mevis_val_mocentric_reverse',
                                                                                    'mevis_val_mocentric_tile_reverse',
                                                                                    'refdavis_val'])
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--frames_selection', type=str, default='keyframes')
    parser.add_argument('--frames_selection_file', type=str, default='keyframes.csv')
    parser.add_argument('--keyframes_save_flag', action='store_true')

    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    cfg = setup_cfg(args)
    inference(cfg, args)

