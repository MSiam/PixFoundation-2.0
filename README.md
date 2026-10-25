# PixFoundation-2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?
[Project webpage](https://msiam.github.io/PixFoundationSeries/), [NeurIPS version](), [Arxiv](https://arxiv.org/abs/2509.02807)

Official implementation of My work PixFoundation 2.0 in NeurIPS 2026 Evaluations and Datasets Track.

## Motion-Centric Benchmark (MoCentric-Bench)

<div align="center">
<img src="imgs/overview_1.png" width="70%" height="70%"><br><br>
</div>

## Benchmarking Pixel-level video MLLMs for visual grounding (RefVOS)

<div align="center">
<img src="imgs/overview_2.png" width="70%" height="70%"><br><br>
</div>

<div align="justify">
Multi-modal large language models (MLLMs) have shown impressive generalization across tasks using images and text modalities. While their extension to video has enabled tasks such as video question answering and video captioning, their pixel-level visual grounding abilities are less studied. In this work, we raise the pertinent question of whether motion is used in pixel-level visual grounding and whether video MLLMs can segment objects based on natural language expressions describing their motion patterns. We identify the shortcomings in the current benchmarks, where we show that a single frame can often suffice for capturing the referring expression without any temporal reasoning. To address this, we introduce novel motion-centric probing, particularly designed for the visual grounding task, to study video MLLMs' ability to identify true motion from a fake one and their ability to grasp the ordering of motion. Consequently, we introduce MoCentric-Bench, a motion-centric benchmark designed to evaluate video MLLMs based on their ability to capture the interaction between motion and language, rather than relying primarily on static cues. We further establish strong single-image baselines that either match or surpass prior methods. Finally, we explore a simple motion-centric adaptation that provides state-of-the-art performance on our MoCentric-Bench.
</div>

## Installation

* Clone the repository recursively to include the submodules
```
git clone --recursive https://github.com/MSiam/PixFoundation-2.0/
```
* Create conda environment
```
conda create --name pixfoundation2 python=3.10
conda activate pixfoundation2
```
* Install the base requirements for evaluations
```
pip install -r requirements.txt
```
* Setup detectron2 for some utilities used (**Optional**)
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
* Follow installation setup for each model you are evaluating, refer to their README and if necessary create its separate conda env.
* Refer to scripts/run_X.sh for each model X script and modify the conda environment if needed or use the common pixfoundation2

## Synthetic Dataset Setup
* Download [MeVIS](https://huggingface.co/datasets/FudanCVL/MeViSv2).
* Generate the multi-video-layout. Layouts l2, l3
```
python data/create_benchmark_mevis.py --config-file configs/mevis.yaml --dataset_root DATA_ROOT --frames_sel_file data/mevis_keyframes.csv --output_dir OUT_DIR
python data/create_benchmark_mevis.py --config-file configs/mevis.yaml --dataset_root DATA_ROOT --frames_sel_file data/mevis_keyframes.csv --output_dir OUT_DIR --left_flag
python data/create_benchmark_mevis.py --config-file configs/mevis.yaml --dataset_root DATA_ROOT --output_dir OUT_DIR --reverse_flag
python data/create_benchmark_mevis.py --config-file configs/mevis.yaml --dataset_root DATA_ROOT --output_dir OUT_DIR --reverse_flag --left_flag
```
* Move our reverse expressions under data/meta_expressions_reverse_filtered.json to the valid_u subset path.

* The final MeVIS directory is as follows:
```
|--- MeVIS
   |--- train
      |--- JPEGImages
      |--- mask_dict.json
      |--- meta_expressions.json
   |--- valid_u
      |--- JPEGImages
      |--- mask_dict.json
      |--- meta_expressions.json
      |--- meta_expressions_reverse_filtered.json
   |--- valid_u_mocentric_tile_single
      |--- JPEGImages
   |--- valid_u_mocentric_tile_single_left
      |--- JPEGImages
   |--- valid_u_mocentric_tile_reverse
      |--- JPEGImages
   |--- valid_u_mocentric_tile_reverse_left
      |--- JPEGImages
```

* Visualize the synthetic dataset with the segmentation masks
```
python datasets_/test_loaders.py --config-file configs/mevis.yaml --dataset_root DATA_ROOT --dataset_split mevis_val_mocentric_tile_single --dataset_mask_path DATA_ROOT/valid_u/mask_dict.json --dataset_exp_path DATA_ROOT/valid_u/EXPRS_JSON --out_dir OUT_DIR --save_vis
```
* You can follow similar procedure to [Molmo2Track](https://huggingface.co/datasets/allenai/Molmo2-VideoTrackEval).

## Evaluation
* Run common bash script to run the benchmarking after modifying the paths
```
cd scripts
bash run_all.sh
```

## Motion Centric Adaptation
The motion centric adaptation can be applied to any MLLM independant of the architecture relying on the synthetic training data. I provide an example with Sa2VA and Molmo2. The forked Sa2VA modified for motion-centric adaptation is provided [here](https://github.com/MSiam/Sa2VA/tree/Sa2va-mocentric). The forked Molmo2 modified for it is provided [here](https://github.com/MSiam/molmo2).

* Train Sa2VA and convert it to Hugging Face Checkpoint
```
cd scripts
bash lora_tune_sa2va.sh
```
* This is the set of [weights](https://www.dropbox.com/scl/fi/gy1prfuvtwea9ztm5jh39/lora_weights_mocentric_sa2va.pth?rlkey=mtxz498i259wqg6fu2rvy5ztz&st=769jsl0d&dl=0) for a run in the motion-centric Sa2VA. The Motion-centric Sa2VA reported in my work is an average of multiple runs. Upon merging with original ByteDance weights you can run the evaluation, refer to lora_tune_sa2va.sh for running the LoRA merging.

* Replace Sa2VA_CKPT in run_all.sh with the correct full path to hugging face checkpoint for evaluation.

* Modifications for the LoRA setup including LLM instead of ViT and other hyperparameters can be added to 'Sa2VA/projects/llava_sam2/configs/sa2va_8b_motion.py'.

* Similarly, train Molmo2 and convert it to Hugging Face Checkpoint
```
cd molmo2/launch_scripts/
bash run_train.sh
```

# Acknowledgements
* I thank [Sa2VA](https://github.com/bytedance/Sa2VA) ByteDance and [Molmo2](https://github.com/allenai/molmo2) authors as I built the experiments for motion-centric adaptation upon their codebases.

# References
Please cite my paper if you find it useful in your research
```
@article{siam2026pixfoundation2.0,
  title={PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?},
  author={Siam, Mennatullah},
  journal={NeurIPS},
  year={2026}
}
```

