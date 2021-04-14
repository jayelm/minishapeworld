#!/bin/bash

python -m msw.generate --n_images 80 --n_correct 40 \
    --config_type logical \
    --config_split --gen_same \
    --enumerate_configs \
    --n_train 20000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --save_dir logical_80 --hdf5 --vis
