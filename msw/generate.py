"""
Generate shapeworld reference games
"""

import json
import os
import gzip

import numpy as np

from . import config, shapeworld, vis

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Fast ShapeWorld", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--n_train", type=int, default=100, help="Number of train examples"
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=100,
        help="Number of val examples (if 0 will not create)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of test examples (if 0 will not create)",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=10,
        help="Images per example (concept/reference only)",
    )
    parser.add_argument(
        "--config_split",
        action="store_true",
        help="Enforce unique configs across splits",
    )
    parser.add_argument(
        "--gen_same",
        action="store_true",
        help="Generate val_same/test_same datasets consisting of same configs as train",
    )
    parser.add_argument(
        "--n_val_same",
        type=int,
        default=100,
        help="Number of val same examples (if not --gen_same or 0 will not create)",
    )
    parser.add_argument(
        "--n_test_same",
        type=int,
        default=100,
        help="Number of test same examples (if not --gen_same or 0 will not create)",
    )
    parser.add_argument(
        "--n_configs",
        default=2500,
        type=int,
        help="If --config_split, how many total configs?"
    )
    parser.add_argument(
        "--train_configs",
        default=0.8,
        type=int,
        help="If --config_split, what percent of total configs are training?"
    )
    parser.add_argument(
        "--enumerate_configs",
        action="store_true",
        help="Enumerate configs rather than sampling (good for logical configs)",
    )
    parser.add_argument(
        "--logical_ops",
        nargs="+",
        default=["and", "or", "not"],
        help="Allowed logical conjunctions",
    )
    parser.add_argument(
        "--min_logical_len",
        type=int,
        default=1,
        help="Minimum logical argument length",
    )
    parser.add_argument(
        "--max_logical_len",
        type=int,
        default=2,
        help="Maximum logical argument length",
    )
    parser.add_argument(
        "--min_correct",
        type=int,
        default=None,
        help="Minimum number of correct images - generate this many, then choose randomly with --p_correct",
    )
    parser.add_argument(
        "--p_correct",
        type=float,
        default=0.5,
        help="Avg correct proportion of images (concept only)",
    )
    parser.add_argument(
        "--n_correct",
        type=int,
        default=None,
        help="Exact number of correct images (must be less than n_images; concept only; overrides --p_correct and --min_correct)",
    )
    parser.add_argument(
        "--color_variance",
        type=float,
        default=0.0,
        help="percentage variance on h/s/v colors (in [0, 1]). recommended: 0.02",
    )
    parser.add_argument(
        "--oversample_shape",
        action="store_true",
        help="Sample extra shapes for logical configs",
    )
    parser.add_argument(
        "--oversample_shape_strategy",
        type=str,
        default="singleton_half",
        choices=["singleton_half", "singleton_quarter", "any_2x", "even"],
        help="Sample extra shapes for logical configs",
    )
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="Number of workers (0 = no multiprocessing)",
    )
    parser.add_argument(
        "--data_type",
        choices=["concept", "reference", "caption"],
        default="concept",
        help="What kind of data to generate",
    )
    parser.add_argument(
        "--lang_type",
        choices=["standard", "simple", "conjunction"],
        default="standard",
        help="What kind of language to generate (only applicable to single config for now)",
    )
    parser.add_argument(
        "--n_distractors",
        default=[2, 3],
        nargs="*",
        type=int,
        help="Number of distractor shapes (for spatial only); "
        "either one int or (min, max)",
    )
    parser.add_argument(
        "--config_type",
        choices=list(config.CONFIGS.keys()),
        default="spatial",
        help="What kind of images to generate",
    )
    parser.add_argument(
        "--hdf5", action="store_true", help="Save in hdf5 format (vs npz)"
    )
    parser.add_argument(
        "--no_worlds", action="store_true", help="Don't save world JSONs"
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Sample visualization of train data to args.save_dir/vis",
    )
    parser.add_argument(
        "--n_vis", default=100, type=int, help="How many examples to visualize?"
    )
    parser.add_argument(
        "--save_dir", default="test", help="Save dataset to this directory"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed (doesn't work for MP)"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if len(args.n_distractors) == 1:
        args.n_distractors = args.n_distractors[0]
    elif len(args.n_distractors) == 2:
        args.n_distractors = tuple(args.n_distractors)
    else:
        parser.error("--n_distractors must be either 1 int or 2 (min, max)")

    cfg = config.CONFIGS[args.config_type]

    world = shapeworld.ShapeWorld(
        data_type=args.data_type,
        config=cfg,
        #  n_distractors=args.n_distractors,
        color_variance=args.color_variance,
    )

    if args.enumerate_configs:
        if args.config_type == "logical":
            configs = cfg.enumerate(min_formula_len=args.min_logical_len,
                                    max_formula_len=args.max_logical_len,
                                    ops=set(args.logical_ops))
        else:
            configs = cfg.enumerate()

    if args.config_split:
        if not args.enumerate_configs:
            # Sample many configs
            configs = cfg.generate(args.n_configs, verbose=True)

        n_train_configs = int(args.train_configs * len(configs))
        train_configs = configs[:n_train_configs]
        test_configs = configs[n_train_configs:]
        print(f"{len(train_configs)} train, {len(test_configs)} test")
    else:
        if args.enumerate_configs:
            print(f"{len(configs)} configs")
            train_configs = configs
            test_configs = configs
        else:
            train_configs = None
            test_configs = None

    if args.config_type == "logical" and args.oversample_shape:
        train_configs = config.configs.logical.oversample(train_configs, strategy=args.oversample_shape_strategy)
        test_configs = config.configs.logical.oversample(test_configs, strategy=args.oversample_shape_strategy)

    os.makedirs(args.save_dir, exist_ok=True)

    dsets = [
        ("train", args.n_train, train_configs),
        ("val", args.n_val, test_configs),
        ("test", args.n_test, test_configs),
    ]
    if args.gen_same:
        dsets.extend(
            [
                ("val_same", args.n_val_same, train_configs),
                ("test_same", args.n_test_same, train_configs),
            ]
        )

    train = None
    for dname, n, cfgs in dsets:
        if n > 0:
            d, dworlds = world.generate(
                n,
                n_images=args.n_images,
                min_correct=args.min_correct,
                p_correct=args.p_correct,
                n_correct=args.n_correct,
                workers=args.workers,
                configs=cfgs,
                verbose=True,
                split=dname,
                lang_type=args.lang_type,
                save_hdf5=args.hdf5,
                save_dir=args.save_dir,

            )

            if not args.hdf5:
                # Save np arrays directly to disk
                dfile = os.path.join(args.save_dir, f"{dname}.npz")
                np.savez_compressed(dfile, **d)

            if not args.no_worlds:
                wfile = os.path.join(args.save_dir, f"{dname}_worlds.json.gz")
                with gzip.open(wfile, 'wt', encoding="ascii") as zf:
                    json.dump(dworlds, zf)

            if dname == "train" and args.vis is not None:
                # Save train for vis
                train = d

    if args.vis is not None:
        vis_dir = os.path.join(args.save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis.visualize(vis_dir, train, n=args.n_vis)
