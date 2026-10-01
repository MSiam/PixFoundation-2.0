# Ensure data/video_datas/ path includes davis17, mevis, revos, rvos simlinks to the data
export PYTHONPATH=${PWD}/../Sa2VA/:$PYTHONPATH

cd ${PWD}/../Sa2VA/
bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_motion.py 1

python projects/llava_sam2/hf/convert_to_hf.py projects/llava_sam2/configs/sa2va_8b_motion.py --pth-model PTH_FILE --save-path HF_CKPT_DIR
