set -x

size=80
item=180
k=400
basis_n=10
train_date=0628
data=data/imgs/Seen400_S80F50_TRAIN800
model_base=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352
model_name=CF_from_${model_base}_${item}_0628
base_idxs="basis/${train_date}/original_basis_2024${train_date}.txt"
base_ws="basis/${train_date}/original_weights_2024${train_date}.pth"
original_centers="basis/${train_date}/original_centers_2024${train_date}.pth"
original_sigma="basis/${train_date}/original_sigma_2024${train_date}.pth"
original_pc="basis/${train_date}/original_pc_2024${train_date}.pth"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main_new.py \
    --content_fusion \
    --img_size ${size} \
    --data_path ${data} \
    --lr 1e-4 \
    --output_k ${k} \
    --batch_size 16 \
    --iters 1000 \
    --epoch 240 \
    --val_num 10 \
    --baseline_idx 0 \
    --save_path output/models \
    --load_model ${model_name} \
    --base_idxs ${base_idxs} --base_ws ${base_ws} \
    --original_centers ${original_centers} \
    --original_sigma ${original_sigma} \
    --original_pc ${original_pc} \
    --ddp \
    --no_val \
    --wdl --w_wdl 0.01