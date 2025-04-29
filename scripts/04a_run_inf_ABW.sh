set -x

for option in unseen
do
    item=220
    k=400
    img_size=80
    base_n=10

    py_file="inf_with_style_ft_ABW_new.py" # inf
    topk=-1 # -1:use all 1:one-hot 2:top2
    ftep=10 # 0
    wdl_ft=0.01
    wdl=0.01
    lr=0.01
    t=0.01

    model_base_src=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352
    model_cf=CF_from_${model_base_src}_180_0628
    model=output/models/logs/${model_cf}/model_${item}.ckpt

    if [ $option = 'seen' ];then
        font_len=400
        target_style=data/imgs/Seen400_S80F50_FS16
        basis_ws=basis/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20231208-225122_180_basis_400_id_10_ws_400x10_t0.01.pth
        save_path="output/test_rsts/${model_cf}_${item}_top${topk}_ft${ftep}_wdl${wdl}_lr${lr}"
    elif [ $option = 'unseen' ]; then
        font_len=100
        target_style=data/imgs/Unseen100_S80F50_FS16
        basis_ws=basis/0628/unseen_weights_20240628_195.pth
        save_path="output/test_rsts/unseen_${model_cf}_${item}_top${topk}_ft${ftep}_wdl${wdl}_lr${lr}"
    fi

    basis_content_folder=data/imgs/400BASIS_S80F50_TEST5646
    basis_style_ft_folder=data/imgs/400BASIS_S80F50_FS16
    #load_sv="output/models/fonts/test_rsts/${model_base}_${item}/style_vec.pth"

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --use_env --master_port 34545 ${py_file} \
    --img_size ${img_size} \
    --data_path ${target_style} \
    --output_k ${k} \
    --load_model ${model} \
    --save_path ${save_path} \
    --font_len ${font_len} \
    --baseline_idx 0 \
    --sty_batch_size 40 \
    --basis_ws  ${basis_ws} \
    --top_k ${topk} \
    --basis_folder ${basis_content_folder} \
    --basis_ft_folder ${basis_style_ft_folder} \
    --ft_epoch ${ftep} \
    --lr ${lr} \
    --wdl --w_wdl ${wdl_ft}
    # --load_style ${load_sv}
    #--pkl --w_pkl ${wdl_ft}
    #--wdl --w_wdl ${wdl_ft}
done