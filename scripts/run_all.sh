DATA_ROOT=''
OUT_ROOT=''

###################################################### RGA
echo "RGA Inference Start valid_u"
bash run_rga.sh ${OUT_ROOT}/RGA_outputs/mevis_validu/ $DATA_ROOT "mevis_val"
echo "RGA Inference Done"

echo "RGA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/RGA_outputs/mevis_validu/ False False False None

#### MeVIS valid_u Tile Single
echo "RGA Inference Start valid_u_mocentric_tile_single"
bash run_rga.sh ${OUT_ROOT}/RGA_outputs/mevis_validu_tile_single/ $DATA_ROOT "mevis_val_mocentric_tile_single"
echo "RGA Inference Done"

echo "RGA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/RGA_outputs/mevis_validu_tile_single/ True True False None

#### MeVIS valid_u Reverse
echo "RGA Inference Start valid_u_reverse"
bash run_rga.sh ${OUT_ROOT}/RGA_outputs/mevis_validu_reverse/ $DATA_ROOT "mevis_val_mocentric_reverse"
echo "RGA Inference Done"

echo "RGA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/RGA_outputs/mevis_validu_reverse/ False False True None

##### MeVIS valid_u Tile Reverse
echo "RGA Inference Start valid_u_tile_reverse"
bash run_rga.sh ${OUT_ROOT}/RGA_outputs/mevis_validu_tile_reverse/ $DATA_ROOT "mevis_val_mocentric_tile_reverse"
echo "RGA Inference Done"

echo "RGA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/RGA_outputs/mevis_validu_tile_reverse/ True False True None

##################################################### Sa2VA
Sa2VA_CKPT='ByteDance/Sa2VA-8B'
echo "Sa2VA Inference Start valid_u"
bash run_sa2va.sh MEVIS_U $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu/ ${Sa2VA_CKPT} False
echo "Sa2VA Inference Done"

echo "Sa2VA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu/ False False False None

#### MeVIS valid_u Tile Single
echo "Sa2VA Inference Start valid_u_mocentric_tile_single"
bash run_sa2va.sh MEVIS_MOCENTRIC_TILE_SINGLE $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_tile_single/ ${Sa2VA_CKPT} False
echo "Sa2VA Inference Done"

echo "Sa2VA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_tile_single/ True True False None

#### MeVIS valid_u Reverse
echo "Sa2VA Inference Start valid_u_reverse"
bash run_sa2va.sh MEVIS_MOCENTRIC_REVERSE $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_reverse/ ${Sa2VA_CKPT} False
echo "Sa2VA Inference Done"

echo "Sa2VA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_reverse/ False False True None

##### MeVIS valid_u Tile Reverse
echo "Sa2VA Inference Start valid_u_tile_reverse"
bash run_sa2va.sh MEVIS_MOCENTRIC_TILE_REVERSE $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_tile_reverse/ ${Sa2VA_CKPT} False
echo "Sa2VA Inference Done"

echo "Sa2VA Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/Sa2VA_outputs/mevis_validu_tile_reverse/ True False True None


##################################################### GPT-5
SAM2_CKPT='sam2_hiera_large.pt'
echo "GPT Inference Start valid_u"
bash run_gpt.sh $DATA_ROOT 'mevis_variants' 'mevis_val' '' ${OUT_ROOT}/GPT_outputs/mevis_validu/ 'sam2' ${SAM2_CKPT} 0
echo "Sa2VA Inference Done"

echo "GPT Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/GPT_outputs/mevis_validu/ False False False None

#### MeVIS valid_u Tile Single
echo "GPT Inference Start valid_u_mocentric_tile_single"
bash run_gpt.sh $DATA_ROOT 'mevis_variants' 'mevis_val_mocentric_tile_single' DATA_FLIP_IMG_PREFIX ${OUT_ROOT}/GPT_outputs/mevis_validu_tile_single/ 'sam2' ${SAM2_CKPT} 0
echo "Sa2VA Inference Done"

echo "GPT Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/GPT_outputs/mevis_validu_tile_single/ True True False None

#### MeVIS valid_u Reverse
echo "GPT Inference Start valid_u_reverse"
bash run_gpt.sh $DATA_ROOT 'mevis_variants' 'mevis_val_mocentric_reverse' '' ${OUT_ROOT}/GPT_outputs/mevis_validu_reverse/ 'sam2' ${SAM2_CKPT} 0
echo "Sa2VA Inference Done"

echo "GPT Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/GPT_outputs/mevis_validu_reverse/ False False True None

##### MeVIS valid_u Tile Reverse
echo "GPT Inference Start valid_u_mocentric_tile_reverse"
bash run_gpt.sh $DATA_ROOT 'mevis_variants' 'mevis_val_mocentric_tile_reverse' DATA_FLIP_IMG_PREFIX ${OUT_ROOT}/GPT_outputs/mevis_validu_tile_reverse/ 'sam2' ${SAM2_CKPT} 0
echo "Sa2VA Inference Done"

echo "GPT Eval"
bash run_eval_mevis.sh $DATA_ROOT ${OUT_ROOT}/GPT_outputs/mevis_validu_tile_reverse/ True False True None

# You can do similar for the single image baselines
