#!/bin/bash

python -m msw.generate --n_images 40 --n_correct 20 \
    --config_type logical \
    --config_split --gen_same \
    --enumerate_configs \
    --n_train 20000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --color_variance 0.02 \
    --save_dir logical_colorvar --hdf5 --vis
