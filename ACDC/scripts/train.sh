#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='acdc'
method='train_cross_match'
exp='unet'

note='cross_match'
config=configs/$dataset.yaml

etas=('0.3')
# Assuming split is now an array
splits=('3' '7')

for split in "${splits[@]}"
do
    for eta in "${etas[@]}"
    do
        labeled_id_path=splits/$dataset/$split/labeled.txt
        unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
        save_path=exp/$dataset/${method}_$note/$exp/$split/eta_$eta
        mkdir -p $save_path
        OMP_NUM_THREADS=4 torchrun \
            --nproc_per_node=$1 \
            --master_addr=localhost \
            --master_port=$2 \
            $method.py \
            --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
            --save-path $save_path --port $2 --eta $eta 2>&1 | tee $save_path/$now.log
    done
done