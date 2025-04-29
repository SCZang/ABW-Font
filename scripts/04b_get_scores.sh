set -x
gid=${1:-"0"}

for option in unseen
do
    
    if [ $option = 'seen' ];then
        rst_name=CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0628_196_top-1_ft10_wdl0.01_lr0.01
        gt_path=data/imgs/Seen400_S80F50_TEST5646
    elif [ $option = 'unseen' ]; then
        rst_name=unseen_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20240401-191352_180_0628_196_top-1_ft10_wdl0.01_lr0.01
        gt_path=data/imgs/Unseen100_S80F50_TEST5646
    fi
    pred_path=output/test_rsts/${rst_name}
    CUDA_VISIBLE_DEVICES=${gid} python eval/get_scores_test.py \
    -gt ${gt_path} \
    -pred ${pred_path} \
    --gpu
    #-m l1
done