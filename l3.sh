#!/bin/bash

python -m msw.generate --n_images 5 --min_correct 4 \
    --config_split --gen_same \
    --train_configs 2000 --val_configs 500 --test_configs 500 \
    --n_train 9000 --n_val 500 --n_val_same 500 --n_test 2000 --n_test_same 2000 \
    --save_dir l3 --vis
