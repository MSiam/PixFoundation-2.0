DATA=$1
DATATYPE=$2
DATASPLIT=$3
DATAFLIP=$4 # Only used in the Motion Centric Evaluation with Concatenation Otherwise send as ''

OUTPUT_DIR=$5

SAMTYPE=$6
SAMCKPT=$7

START=$8

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate pixfoundation2

export PYTHONPATH=$PWD/../:$PYTHONPATH

python ../infer/infer_gpt.py --config-file ../configs/mevis.yaml --output_dir=${OUTPUT_DIR} --dataset_split $DATASPLIT --dataset_type ${DATATYPE} --image_prefix_root_flip=${DATAFLIP} --dataset_root=${DATA} --sam_type ${SAMTYPE} --sam_ckpt ${SAMCKPT} --start_index $START 
