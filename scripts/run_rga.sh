OUTDIR=$1
DATA=$2
DATATYPE=$3

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate rgafixed

export PYTHONPATH=$PWD/../:$PWD/../RGA3-release/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python ../infer/infer_rga.py \
  --dataset_root "$DATA" \
  --version "SurplusDeficit/UniGR-7B" \
  --output_dir "$OUTDIR" \
  --dataset_split "${DATATYPE}" \
  --config-file ../configs/mevis.yaml
