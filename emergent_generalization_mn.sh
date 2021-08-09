#!/bin/bash

N_SHAPES="$1"
N_COLORS="$2"

SHAPEWORLD_N_SHAPES="$1" SHAPEWORLD_N_COLORS="$2" python -m msw.generate --n_images 40 --n_correct 20 \
    --config_type logical \
    --config_split --gen_same \
    --enumerate_configs \
    --n_train 20000 --n_val 1000 --n_val_same 1000 --n_test 1000 --n_test_same 1000 \
    --color_variance 0.02 \
    --oversample_shape \
    --oversample_shape_strategy even \
    --save_dir shapeworld_"$N_SHAPES"_"$N_COLORS" --hdf5 --vis
