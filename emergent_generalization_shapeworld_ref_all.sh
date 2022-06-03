#!/bin/bash

python -m msw.generate --n_images 40 --n_correct 20 \
    --config_type logical \
    --gen_same \
    --logical_ops and \
    --min_logical_len 2 \
    --max_logical_len 2 \
    --color_variance 0.02 \
    --enumerate_configs \
    --n_train 20000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --save_dir shapeworld_ref_all --vis --hdf5
