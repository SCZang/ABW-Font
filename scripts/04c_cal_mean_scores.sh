set -x
gid=${1:-"0"}

for option in unseen
do
    
    if [ $option = 'seen' ];then
        rst_name=seen_dgfont_GAN_20240520-234439_50
        font_len=400
    elif [ $option = 'unseen' ]; then
        rst_name=unseen_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0628_196_top-1_ft10_wdl0.01_lr0.01
        font_len=100
    fi
    pred_path=output/test_rsts/${rst_name}
    CUDA_VISIBLE_DEVICES=${gid} python eval/cal_mean.py \
    -f ${pred_path}/a_scores/ \
    -k ${font_len}
    # -j 1 3 4 ## !!! jump some fonts like basis font
done