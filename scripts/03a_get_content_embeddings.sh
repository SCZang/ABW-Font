k=400
item=200
img_size=80
model_base=CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0628
model=output/models/logs/${model_base}/model_${item}.ckpt

# data=data/imgs/Seen240_S80F50_TRAIN800
# !!! The more characters used to cluster, the better. 
# However, since the memory limitation, you can also use a subset of data_K240_S80F50_TRAIN800 to cluster.
# like random choose 50 characters or simply use few-shot 16 characters.
data=data/imgs/Seen400_S80F50_FS16

CUDA_VISIBLE_DEVICES=0 python collect_content_embeddings.py --img_size ${img_size} \
--data_path ${data} \
--output_k ${k} \
--batch_size 32 \
--load_model ${model} \
--save_path output/embeddings/seen_embedding_${model_base}_${item} \
--baseline_idx 0 \
--n_atts 400 \
--no_skip