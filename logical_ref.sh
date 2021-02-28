#!/bin/bash

python -m msw.generate --n_images 40 --n_correct 20 \
    --config_type logical \
    --logical_ops and \
    --min_logical_len 2 \
    --max_logical_len 2 \
    --config_split --gen_same \
    --enumerate_configs \
    --n_train 10000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --save_dir logical_ref --vis
