set -x
item=180
model_base_folder=output/models/logs
model=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352
model_cf=CF_from_${model}_${item}_0628
ckpt=model_${item}.ckpt

mkdir ${model_base_folder}/${model_cf}
cp ${model_base_folder}/${model}/${ckpt} ${model_base_folder}/${model_cf}/
echo ${ckpt} > ${model_base_folder}/${model_cf}/checkpoint.txt