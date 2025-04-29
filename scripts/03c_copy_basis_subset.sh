base_n=10

for FLAG in TRAIN800 TEST5646 FS16
do
    basis=basis/0628/basis_400_epoch_199_999.txt
    in_folder=data/imgs/Seen400_S80F50_${FLAG}
    out_folder=data/imgs/400BASIS_S80F50_${FLAG}

    python scripts/basis/copy_basis_imgs.py -b ${basis} -i ${in_folder} -o ${out_folder}
done