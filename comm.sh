#!/bin/bash

python minishapeworld.py --n_images 10 \
    --img_type single \
    --train_configs 2000 --val_configs 500 --test_configs 500 \
    --n_train 9000 --n_val 500 --n_test 2000 \
    --save_dir comm --vis
