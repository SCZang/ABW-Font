n_basis=10
model_name=CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0505
item=185
item_minus_one=$((item-1))
train_data=0505
content_fm=output/embeddings/embedding_${model_name}_${item}/c_src.pth
temp_centers=basis/${train_data}/tc_400x10_t0.01_${item_minus_one}_999.pth
temp_sigma=basis/${train_data}/ts_400x10_t0.01_${item_minus_one}_999.pth
temp_pc=basis/${train_data}/tpc_400x10_t0.01_${item_minus_one}_999.pth

CUDA_VISIBLE_DEVICES=0 python cos_sim.py \
-tc ${temp_centers} \
-ts ${temp_sigma} \
-tpc ${temp_pc} \
-c ${content_fm} -lbs 5 -nb ${n_basis} -m ${model_name}_${item}