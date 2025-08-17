#!/bin/bash

experiments=(
    # "python -m src.trained_models.cnn_transformer.run_experiment --base_dir 'data/6 May 25/frames' --learning_rate 1e-6 --n_channels 3 --epochs 50 --t_max 50 --no_positional_encoding >> ablation_data_and_pos_enc/6_may_no_pos.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold1' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold1.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold2' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold2.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold3' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold3.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold4' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold4.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold5' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold5.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts/externals/folds/fold6' --batch_size 1 --epochs 100 --t_max 100 --learning_rate 1e-5 --weight_decay 1e-4 >> 500_to_1000_1s_withtest_f1_weighted_fold6.txt"

)
for experiment in "${experiments[@]}"
do
    echo "Running experiment: $experiment"
    eval $experiment  # Evaluate and run the command
done