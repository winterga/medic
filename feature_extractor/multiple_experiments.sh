#!/bin/bash

experiments=(
    # "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5  >> outputs/ablation/3x5_logits_alternating.txt"
    # "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --training_style sequential >> outputs/ablation/3x5_logits_sequential.txt"
    # "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --no_add_logits >> outputs/ablation/3x5_withoutlogits_alternating.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --training_style sequential --no_add_logits >> outputs/ablation/3x5_withoutlogits_sequential.txt"

    # "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --images_in_batch 15 >> outputs/ablation/15_logits_alternating.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --images_in_batch 15 --training_style sequential >> outputs/ablation/15_logits_sequential.txt"
    # "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --images_in_batch 15 --no_add_logits >> outputs/ablation/15_withoutlogits_alternating.txt"
    "python -m src.run_experiment --base_dir '../data/images_ts_fe_30_singles' --batch_size 1 --epochs 150 --learning_rate 1e-5 --images_in_batch 15 --training_style sequential --no_add_logits >> outputs/ablation/15_withoutlogits_sequential.txt"
    )
for experiment in "${experiments[@]}"
do
    echo "Running experiment: $experiment"
    eval $experiment  # Evaluate and run the command
done


# 3x5 logits squential
# 3x5 logits alternating
# 3x5 nologits sequential
# 3x5 nologits alternating
# 15 logits squential
# 15 logits alternating
# 15 nologits sequential
# 15 nologits alternating