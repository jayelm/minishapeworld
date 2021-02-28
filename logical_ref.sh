#!/bin/bash

python -m msw.generate --n_images 40 --n_correct 1 \
    --config_type single \
    --config_split --gen_same \
    --enumerate_configs \
    --n_train 10000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --save_dir logical_ref --vis