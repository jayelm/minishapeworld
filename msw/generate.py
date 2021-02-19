"""
Generate shapeworld reference games
"""

import json
import os

import numpy as np

from . import config, sw, vis

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
        help="Generate val_same/test_s datasets consisting of same configs as train (requires --config_split)",
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
        "--train_configs",
        default=2000,
        type=int,
        help="If --config_split, how many unique configs at train?",
    )
    parser.add_argument(
        "--val_configs",
        default=500,
        type=int,
        help="If --config_split, how many unique configs at val?",
    )
    parser.add_argument(
        "--test_configs",
        default=500,
        type=int,
        help="If --config_split, how many unique configs at test?",
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
        "--n_distractors",
        default=[2, 3],
        nargs="*",
        type=int,
        help="Number of distractor shapes (for spatial only); "
        "either one int or (min, max)",
    )
    parser.add_argument(
        "--config_type",
        choices=["single", "spatial"],
        default="spatial",
        help="What kind of images to generate",
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

    args = parser.parse_args()

    if len(args.n_distractors) == 1:
        args.n_distractors = args.n_distractors[0]
    elif len(args.n_distractors) == 2:
        args.n_distractors = tuple(args.n_distractors)
    else:
        parser.error("--n_distractors must be either 1 int or 2 (min, max)")

    if args.gen_same and not args.config_split:
        parser.error("--config_split must be set to use --gen_same")

    cfg = config.CONFIGS[args.config_type]

    world = sw.ShapeWorld(
        data_type=args.data_type, config=cfg, n_distractors=args.n_distractors
    )

    if args.config_split:
        # Pre-generate unique configs
        total_configs = args.train_configs + args.val_configs + args.test_configs
        configs = world.generate_configs(total_configs, verbose=True)
        train_configs = configs[: args.train_configs]
        val_configs = configs[
            args.train_configs : args.train_configs + args.val_configs
        ]
        test_configs = configs[args.train_configs + args.val_configs :]
    else:
        train_configs = None
        val_configs = None
        test_configs = None

    os.makedirs(args.save_dir, exist_ok=True)

    dsets = [
        ("train", args.n_train, train_configs),
        ("val", args.n_val, val_configs),
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
                desc=dname,
            )
            dfile = os.path.join(args.save_dir, f"{dname}.npz")
            np.savez_compressed(dfile, **d)
            if not args.no_worlds:
                wfile = os.path.join(args.save_dir, f"{dname}_worlds.json")
                with open(wfile, "w") as f:
                    json.dump(dworlds, f)

            if dname == "train" and args.vis is not None:
                # Save train for vis
                train = d

    if args.vis is not None:
        vis_dir = os.path.join(args.save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis.visualize(vis_dir, train, n=args.n_vis)
