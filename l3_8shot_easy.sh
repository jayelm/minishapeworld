#!/bin/bash

python -m msw.generate --n_images 9 --min_correct 8 \
    --config_split --gen_same \
    --train_configs 2000 --val_configs 500 --test_configs 500 \
    --n_train 9000 --n_val 500 --n_val_same 500 --n_test 2000 --n_test_same 2000 \
    --n_distractors 0 \
    --save_dir l3_8shot_easy
