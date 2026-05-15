DATATYPE=$1
DATA=$2
OUT=$3
CKPT=$4
FLIP=$5

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate sa2va

export PYTHONPATH=$PWD/../Sa2VA/:$PYTHONPATH

ARGS="${CKPT} --dataset ${DATATYPE} --dataset_root ${DATA} --work_dir ${OUT}"

if [ $FLIP == "True" ]
then
    ARGS="$ARGS --flip"
fi

python ../infer/infer_sa2va.py ${ARGS}
