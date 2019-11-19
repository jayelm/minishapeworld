"""
Generate shapeworld reference games
"""

import numpy as np
from numpy import random
from tqdm import tqdm
import os
import multiprocessing as mp
import json

import image
import color
import shape
import config
import constants as c
import vis


def world_json(config, shapes, lang_type='standard'):
    return {
        'lang': config.format(lang_type=lang_type),
        'config': config.json(),
        'shapes': [s.json() for s in shapes]
    }


class MiniShapeWorld:
    def __init__(self,
                 data_type='concept',
                 img_type='spatial',
                 colors=None,
                 shapes=None,
                 n_distractors=0,
                 unique_distractors=True,
                 unrestricted_distractors=True):
        """

        :param data_type: one of 'concept', 'reference' or 'caption'
        :param img_type: what kind of concept is represented in each image
        :param colors: optional subset of colors to sample from
        :param shapes: optional subset of colors to sample from
        :param n_distractors: number of distractor shapes in each image
            unrelated to concept (> 1 only valid for spatial). Either a single
            int for fixed number, or a tuple range (min, max)
        :param unique_distractors: enforce that distractors have different
            colors and shapes from concept shapes
        :param unrestricted_distractors: sample distractor colors/shapes freely
            from colors
        """
        if data_type not in ['concept', 'reference', 'caption']:
            raise NotImplementedError("data_type = {}".format(data_type))

        if img_type not in ['spatial', 'single']:
            raise NotImplementedError("img_type = {}".format(img_type))

        self.data_type = data_type
        self.img_type = img_type

        if colors is None:
            colors = color.COLORS
        if shapes is None:
            shapes = shape.SHAPES
        self.colors = colors
        self.shapes = shapes

        self.n_distractors = n_distractors
        self.unique_distractors = unique_distractors
        self.unrestricted_distractors = unrestricted_distractors

    def generate_configs(self, n, verbose=False):
        """
        Generate unique configs
        """
        total_configs = set()

        if self.img_type == 'single':
            cfg_func = self.random_config_single
        elif self.img_type == 'spatial':
            cfg_func = self.random_config_spatial

        if verbose:
            pbar = tqdm(total=n)

        while len(total_configs) < n:
            new_cfg = cfg_func()
            if new_cfg not in total_configs:
                total_configs.add(new_cfg)
                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()

        return list(total_configs)

    def generate(self,
                 n,
                 n_images=10,
                 min_correct=None,
                 p_correct=0.5,
                 n_correct=None,
                 float_type=False,
                 configs=None,
                 lang_type='standard',
                 pool=None,
                 workers=0,
                 verbose=False,
                 desc=''):
        """
        Generate dataset
        :param n: number of examples to generate
        :param n_images: number of images per example
        :param min_correct: minimum number of positive examples (after minimum,
            will randomly sample according to p_correct)
        :param p_correct: proportion of positive examples to sample
        :param n_correct: exact number of positive examples (will override
            min_correct/p_correct behavior); cannot be 0 or more than n_images
        :param float_type: return images as np.float32 array between 0.0 and
            1.0, rather than uint8 between 0 and 255
        :param lang_type: language type - either standard (full language) or
            simple (just the essentials)
        :param configs: sample from these possible configs (else will generate random configs)
        :param pool: use this multiprocessing pool
        :param workers: number of workers to use (will create own pool; cannot use with pool)
        :param verbose: print progress

        :return data: a dict with keys ['langs', 'imgs', 'labels'], each
        np.arrays
        :return worlds: a JSON object: list of [lists of worlds], each world
            being the shapes and config (possibly invalidated) for the image
        """
        do_mp = workers > 0
        if not do_mp and pool is not None:
            raise ValueError("Can't specify pool if workers > 0")
        if do_mp:
            pool_was_none = False
            if pool is None:
                pool_was_none = True
                pool = mp.Pool(workers)
        if lang_type not in ['standard', 'simple']:
            raise NotImplementedError(f"lang_type = {lang_type}")

        if self.data_type == 'concept':
            assert n_images > 4, "Too few n_images"
        elif self.data_type == 'reference':
            assert n_images > 1, "Too few n_images"
        elif self.data_type == 'caption':
            n_images = 1

        if n_correct is not None:
            assert 0 < n_correct <= n_images, f"n_correct ({n_correct}) must be > 0 and <= n_images ({n_images})"

        all_imgs = np.zeros((n, n_images, 3, c.DIM, c.DIM), dtype=np.uint8)
        all_labels = np.zeros((n, n_images), dtype=np.uint8)

        mp_args = [(n_images, min_correct, p_correct, n_correct, configs, i) for i in range(n)]

        if do_mp:
            gen_iter = pool.imap(self._generate_one_mp, mp_args)
        else:
            gen_iter = map(self._generate_one_mp, mp_args)
        if verbose:
            gen_iter = tqdm(gen_iter, total=n, desc=desc)

        target_configs = []
        all_configs = []
        world_jsons = []
        for imgs, labels, target_cfg, cfgs, shapes, i in gen_iter:
            all_imgs[i, ] = imgs
            all_labels[i, ] = labels
            target_configs.append(target_cfg)
            all_configs.append(cfgs)
            wjsons = [world_json(cfg, s) for cfg, s in zip(cfgs, shapes)]
            if self.data_type == 'caption':
                # No multiple images per example
                wjsons = wjsons[0]
            world_jsons.append(wjsons)

        if do_mp and pool_was_none:  # Remember to close the pool
            pool.close()
            pool.join()

        if float_type:
            all_imgs = np.divide(all_imgs, 255.0)
            all_labels = all_labels.astype(np.float32)
        langs = np.array([cfg.format(lang_type=lang_type) for cfg in target_configs],
                         dtype=np.unicode)

        if self.data_type == 'caption':
            # Squeeze out the images per example dim
            all_imgs = all_imgs.squeeze(1)
            all_labels = all_labels.squeeze(1)

        return {
            'imgs': all_imgs,
            'labels': all_labels,
            'langs': langs,
        }, world_jsons

    def _generate_one_mp(self, mp_args):
        """
        Wrapper around generate_one which accepts a tuple of args (and an index
        i) for multiprocsesing purposes

        mp_args is a tuple of (n_images, min_correct, p_correct, n_correct,
        configs, i); see self.generate_one
        """
        *mp_args, i = mp_args
        return self.generate_one(*mp_args) + (i, )

    def generate_one(self, n_images, min_correct=None, p_correct=0.5, n_correct=None, configs=None):
        """
        Generate a single example

        :param n_images: number of images in the example
        :param min_correct: minimum number of targets correct; after targets
            filled, will sample randomly. If None, defaults to minimum of 2
            targets and 2 distractors
        :param p_correct: probability of correct targets (after min_correct has
            been filled)
        :param n_correct: exact number of correct targets. If specified, will
            sample exactly n_correct positive examples and (n_images -
            n_correct) negative examples. This OVERRIDES min_correct and
            p_correct behavior!
        :param configs: optional list of configs to sample from. If None, then
            just create randomly
        :return images: an np.array of shape [n_images x 3 x c.DIM x c.DIM]
            representing the images in NCHW format
        :return labels: an np.array of shape [n_images] representing the
            labels. *These are not shuffled*, and minimum/exact # of targets
            will be first labels
        :return config: The original config for the positive examples
        """
        random.seed()  # For multiprocessing (TODO: can we avoid this?)
        imgs = np.zeros((n_images, 3, c.DIM, c.DIM), dtype=np.uint8)
        labels = np.zeros((n_images, ), dtype=np.uint8)
        if configs is not None:
            cfg_idx = random.choice(len(configs))
            target_cfg = configs[cfg_idx]
        else:
            # FIXME: Make this OOP to avoid these if statements
            if self.img_type == 'spatial':
                target_cfg = self.random_config_spatial()
            elif self.img_type == 'single':
                target_cfg = self.random_config_single()

        if self.data_type == 'concept':
            if n_correct is not None:
                # Fixed number of targets and distractors
                n_target = n_correct
                n_distract = n_images - n_target
            elif min_correct is not None:
                # Minimum number of targets, otherwise sample whatever
                # TODO: combine min_correct and min_incorrect
                n_target = min_correct
                n_distract = 0
            else:
                # Minimum of 2 targets and distractors each
                n_target = 2
                n_distract = 2
        else:
            n_target = 1
            n_distract = n_images  # Never run out of distractors

        cfgs = []
        shapes = []
        for i in range(n_images):
            if n_target > 0:
                label = 1
                n_target -= 1
            elif n_distract > 0:
                label = 0
                n_distract -= 1
            else:
                label = (random.random() < p_correct)

            if self.img_type == 'spatial':
                new_cfg, new_shapes = self.generate_spatial(target_cfg, label)
            elif self.img_type == 'single':
                new_cfg, new_shapes = self.generate_single(target_cfg, label)

            # Create image and draw shapes
            img = self.create_image(new_shapes)
            imgs[i] = img
            labels[i] = label
            cfgs.append(new_cfg)
            shapes.append(new_shapes)

        return imgs, labels, target_cfg, cfgs, shapes

    def generate_spatial(self, cfg, label):
        """
        Generate a single spatial relation according to the config,
        invalidating it if the label is 0.
        """
        cfg_attempts = 0
        while cfg_attempts < c.MAX_INVALIDATE_ATTEMPTS:
            new_cfg = cfg if label else self.invalidate_spatial(cfg)
            (ss1, ss2), relation, relation_dir = new_cfg
            s2 = self.add_shape_from_spec(ss2, relation, relation_dir)

            # Place second shape
            attempts = 0
            while attempts < c.MAX_PLACEMENT_ATTEMPTS:
                s1 = self.add_shape_rel(ss1, s2, relation, relation_dir)
                if not s2.intersects(s1):
                    break
                attempts += 1
            else:
                raise RuntimeError(
                    "Could not place shape onto image without intersection")

            if label:  # Positive example
                break
            elif cfg.does_not_validate([s1], s2):
                break
            else:
                # print(f"Shapes {[s1, s2]} for invalid config '{str(new_cfg)}' validates the original config '{str(cfg)}' (attempt {cfg_attempts})")
                assert True

            cfg_attempts += 1

        # Place distractor shapes
        shapes = [s1, s2]
        n_dist = self.sample_n_distractor()
        for _ in range(n_dist):
            attempts = 0
            while attempts < c.MAX_DISTRACTOR_PLACEMENT_ATTEMPTS:
                dss = self.sample_distractor(
                    existing_shapes=shapes)
                ds = self.add_shape(dss)
                # No intersections
                if not any(ds.intersects(s) for s in shapes):
                    if label:
                        shapes.append(ds)
                        break
                    else:
                        # If this is a *negative* example, we should not have
                        # the relation expressed by the original config
                        if cfg.does_not_validate(shapes, ds):
                            shapes.append(ds)
                            break
                        else:
                            #  print(f"Adding new shape {ds} to existing shapes {shapes} validates the original config {str(cfg)} (attempt {attempts})")
                            assert True
                attempts += 1
            else:
                raise RuntimeError("Could not place distractor onto "
                                   "image without intersection")

        return new_cfg, shapes

    def invalidate_spatial(self, cfg):
        # Invalidate by randomly choosing one property to change:
        ((shape_1_color, shape_1_shape),
         (shape_2_color, shape_2_shape)), relation, relation_dir = cfg
        properties = []
        if shape_1_color is not None:
            properties.append(config.ConfigProps.SHAPE_1_COLOR)
        if shape_1_shape is not None:
            properties.append(config.ConfigProps.SHAPE_1_SHAPE)
        if shape_2_color is not None:
            properties.append(config.ConfigProps.SHAPE_2_COLOR)
        if shape_2_shape is not None:
            properties.append(config.ConfigProps.SHAPE_2_SHAPE)
        sp = len(properties)
        # Invalidate relations half of the time
        for _ in range(sp):
            properties.append(config.ConfigProps.RELATION_DIR)
        # Randomly select property to invalidate
        # TODO: Support for invalidating multiple properties
        invalid_prop = random.choice(properties)

        if invalid_prop == config.ConfigProps.SHAPE_1_COLOR:
            inv_cfg = ((self.new_color(shape_1_color), shape_1_shape),
                       (shape_2_color, shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_1_SHAPE:
            inv_cfg = ((shape_1_color, self.new_shape(shape_1_shape)),
                       (shape_2_color, shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_2_COLOR:
            inv_cfg = ((shape_1_color, shape_1_shape),
                       (self.new_color(shape_2_color),
                        shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_2_SHAPE:
            inv_cfg = ((shape_1_color, shape_1_shape),
                       (shape_2_color,
                        self.new_shape(shape_2_shape))), relation, relation_dir
        elif invalid_prop == config.ConfigProps.RELATION_DIR:
            inv_cfg = ((shape_1_color, shape_1_shape),
                       (shape_2_color,
                        shape_2_shape)), relation, 1 - relation_dir
        else:
            raise RuntimeError
        return config.SpatialConfig(*inv_cfg)

    def generate_single(self, cfg, label):
        new_cfg = cfg if label else self.invalidate_single(cfg)

        color_, shape_ = new_cfg
        if shape_ is None:
            shape_ = self.random_shape()
        if color_ is None:
            color_ = self.random_color()
        s = shape.SHAPE_IMPLS[shape_](color_=color_)

        return new_cfg, [s]

    def create_image(self, shapes):
        """
        Create an image and place shapes onto it.

        :param shapes: list[shape.Shape]
        :returns: np.array in NCHW format
        """
        img = image.IMG()
        img.draw_shapes(shapes)
        img = np.transpose(img.array(), (2, 0, 1))
        return img

    def invalidate_single(self, cfg):
        color_, shape_ = cfg
        if shape_ is not None and color_ is not None:
            # Sample random part to invalidate
            # Here, we can invalidate shape, or invalidate color, OR
            # invalidate both
            part_to_invalidate = random.randint(3)
            if part_to_invalidate == 0:
                return (self.new_color(color_), shape_)
            elif part_to_invalidate == 1:
                return (color_, self.new_shape(shape_))
            elif part_to_invalidate == 2:
                return (self.new_color(color_), self.new_shape(shape_))
            else:
                raise RuntimeError
        elif shape_ is not None:
            assert color_ is None
            return (None, self.new_shape(shape_))
        elif color_ is not None:
            assert shape_ is None
            return (self.new_color(color_), None)
        else:
            raise RuntimeError

    def random_shape(self, unrestricted=False):
        if unrestricted:
            return random.choice(shape.SHAPES)
        else:
            return random.choice(self.shapes)

    def random_color(self, unrestricted=False):
        if unrestricted:
            return random.choice(color.COLORS)
        else:
            return random.choice(self.colors)

    def random_shape_from_spec(self, spec):
        color_ = None
        shape_ = None
        if spec == config.ShapeSpec.SHAPE:
            shape_ = self.random_shape()
        elif spec == config.ShapeSpec.COLOR:
            color_ = self.random_color()
        elif spec == config.ShapeSpec.BOTH:
            shape_ = self.random_shape()
            color_ = self.random_color()
        else:
            raise ValueError("Unknown spec {}".format(spec))
        return (color_, shape_)

    def random_config_single(self):
        shape_spec = config.ShapeSpec(random.randint(3))
        shape_ = self.random_shape_from_spec(shape_spec)
        return config.SingleConfig(*shape_)

    def sample_n_distractor(self):
        if isinstance(self.n_distractors, tuple):
            n_dist = random.randint(self.n_distractors[0],
                                    self.n_distractors[1] +
                                    1)  # Exclusive range
        else:
            n_dist = self.n_distractors
        return n_dist

    def sample_distractor(self, existing_shapes=()):
        d = (self.random_color(unrestricted=self.unrestricted_distractors),
             self.random_shape(unrestricted=self.unrestricted_distractors))
        if d in existing_shapes:
            return self.sample_distractor(existing_shapes=existing_shapes)
        return d

    def sample_distractors(self, existing_shapes=()):
        n_dist = self.sample_n_distractor()

        distractors = []
        for _ in range(n_dist):
            d = self.sample_distractor()
            distractors.append(d)
        return distractors

    def random_config_spatial(self):
        # 0 -> only shape specified
        # 1 -> only color specified
        # 2 -> only both specified
        shape_1_spec = config.ShapeSpec(random.randint(3))
        shape_2_spec = config.ShapeSpec(random.randint(3))
        shape_1 = self.random_shape_from_spec(shape_1_spec)
        shape_2 = self.random_shape_from_spec(shape_2_spec)
        if shape_1 == shape_2:
            return self.random_config_spatial()
        relation = random.randint(2)
        relation_dir = random.randint(2)
        return config.SpatialConfig((shape_1, shape_2), relation, relation_dir)

    def add_shape_from_spec(self,
                            spec,
                            relation,
                            relation_dir,
                            shapes=None,
                            attempt=1):
        if attempt > c.MAX_PLACEMENT_ATTEMPTS:
            return None
        color_, shape_ = spec
        if shape_ is None:
            shape_ = self.random_shape()
        if color_ is None:
            color_ = self.random_color()
        s = shape.SHAPE_IMPLS[shape_](relation=relation,
                                      relation_dir=relation_dir,
                                      color_=color_)
        if shapes is not None:
            for oth in shapes:
                if s.intersects(oth):
                    return self.add_shape_from_spec(spec,
                                                    relation,
                                                    relation_dir,
                                                    shapes=shapes,
                                                    attempt=attempt + 1)
            shapes.append(s)
            return s
        return s

    def add_shape_rel(self, spec, oth_shape, relation, relation_dir):
        """
        Add shape, obeying the relation/relation_dir w.r.t. oth shape
        """
        color_, shape_ = spec
        if shape_ is None:
            shape_ = self.random_shape()
        if color_ is None:
            color_ = self.random_color()
        if relation == 0:
            new_y = shape.rand_pos()
            if relation_dir == 0:
                # Shape must be LEFT of oth shape
                new_x = random.randint(c.X_MIN, oth_shape.x - c.BUFFER)
            else:
                # Shape RIGHT of oth shape
                new_x = random.randint(oth_shape.x + c.BUFFER, c.X_MAX)
        else:
            new_x = shape.rand_pos()
            if relation_dir == 0:
                # BELOW (remember y coords reversed)
                new_y = random.randint(oth_shape.y + c.BUFFER, c.X_MAX)
            else:
                # ABOVE
                new_y = random.randint(c.X_MIN, oth_shape.y - c.BUFFER)
        return shape.SHAPE_IMPLS[shape_](x=new_x, y=new_y, color_=color_)

    def add_shape(self, spec):
        """
        Add shape according to spec
        """
        color_, shape_ = spec
        if shape_ is None:
            shape_ = self.random_shape()
        if color_ is None:
            color_ = self.random_color()
        x = shape.rand_pos()
        y = shape.rand_pos()
        return shape.SHAPE_IMPLS[shape_](x=x, y=y, color_=color_)

    def new_color(self, existing_color):
        if len(self.colors) == 1 and self.colors[0] == existing_color:
            raise RuntimeError("No additional colors to generate")
        new_c = existing_color
        while new_c == existing_color:
            new_c = random.choice(self.colors)
        return new_c

    def new_shape(self, existing_shape):
        if len(self.shapes) == 1 and self.shapes[0] == existing_shape:
            raise RuntimeError("No additional shapes to generate")
        new_s = existing_shape
        while new_s == existing_shape:
            new_s = random.choice(self.shapes)
        return new_s


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Fast ShapeWorld',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_train',
                        type=int,
                        default=100,
                        help='Number of train examples')
    parser.add_argument('--n_val',
                        type=int,
                        default=100,
                        help='Number of val examples (if 0 will not create)')
    parser.add_argument('--n_test',
                        type=int,
                        default=100,
                        help='Number of test examples (if 0 will not create)')
    parser.add_argument('--n_images',
                        type=int,
                        default=10,
                        help='Images per example (concept/reference only)')
    parser.add_argument('--config_split',
                        action='store_true',
                        help='Enforce unique configs across splits')
    parser.add_argument('--gen_same',
                        action='store_true',
                        help='Generate val_same/test_s datasets consisting of same configs as train (requires --config_split)')
    parser.add_argument('--n_val_same', type=int, default=100, help='Number of val same examples (if not --gen_same or 0 will not create)')
    parser.add_argument('--n_test_same', type=int, default=100, help='Number of test same examples (if not --gen_same or 0 will not create)')
    parser.add_argument('--train_configs',
                        default=2000,
                        type=int,
                        help='If --config_split, how many unique configs at train?')
    parser.add_argument('--val_configs',
                        default=500,
                        type=int,
                        help='If --config_split, how many unique configs at val?')
    parser.add_argument('--test_configs',
                        default=500,
                        type=int,
                        help='If --config_split, how many unique configs at test?')
    parser.add_argument('--min_correct',
                        type=int,
                        default=None,
                        help='Minimum number of correct images - generate this many, then choose randomly with --p_correct')
    parser.add_argument('--p_correct',
                        type=float,
                        default=0.5,
                        help='Avg correct proportion of images (concept only)')
    parser.add_argument('--n_correct',
                        type=int,
                        default=None,
                        help='Exact number of correct images (must be less than n_images; concept only; overrides --p_correct and --min_correct)')
    parser.add_argument('--workers',
                        default=0,
                        type=int,
                        help="Number of workers (0 = no multiprocessing)")
    parser.add_argument('--data_type',
                        choices=['concept', 'reference', 'caption'],
                        default='concept',
                        help='What kind of data to generate')
    parser.add_argument('--n_distractors',
                        default=[2, 3],
                        nargs='*',
                        type=int,
                        help='Number of distractor shapes (for spatial only); '
                        'either one int or (min, max)')
    parser.add_argument('--img_type',
                        choices=['single', 'spatial'],
                        default='spatial',
                        help='What kind of images to generate')
    parser.add_argument('--no_worlds',
                        action='store_true',
                        help='Don\'t save world JSONs')
    parser.add_argument('--vis',
                        action='store_true',
                        help='Sample visualization of train data to args.save_dir/vis')
    parser.add_argument('--n_vis',
                        default=100,
                        type=int,
                        help='How many examples to visualize?')
    parser.add_argument('--save_dir',
                        default='test',
                        help='Save dataset to this directory')

    args = parser.parse_args()

    if len(args.n_distractors) == 1:
        args.n_distractors = args.n_distractors[0]
    elif len(args.n_distractors) == 2:
        args.n_distractors = tuple(args.n_distractors)
    else:
        parser.error("--n_distractors must be either 1 int or 2 (min, max)")

    if args.gen_same and not args.config_split:
        parser.error("--config_split must be set to use --gen_same")

    msw = MiniShapeWorld(data_type=args.data_type,
                         img_type=args.img_type,
                         n_distractors=args.n_distractors)

    if args.config_split:
        # Pre-generate unique configs
        total_configs = args.train_configs + args.val_configs + args.test_configs
        configs = msw.generate_configs(total_configs, verbose=True)
        train_configs = configs[:args.train_configs]
        val_configs = configs[args.train_configs:args.train_configs + args.val_configs]
        test_configs = configs[args.train_configs + args.val_configs:]
    else:
        train_configs = None
        val_configs = None
        test_configs = None

    os.makedirs(args.save_dir, exist_ok=True)

    dsets = [
        ('train', args.n_train, train_configs),
        ('val', args.n_val, val_configs),
        ('test', args.n_test, test_configs),
    ]
    if args.gen_same:
        dsets.extend([
            ('val_same', args.n_val_same, train_configs),
            ('test_same', args.n_test_same, train_configs),
        ])

    train = None
    for dname, n, cfgs in dsets:
        if n > 0:
            d, dworlds = msw.generate(n,
                                      n_images=args.n_images,
                                      min_correct=args.min_correct,
                                      p_correct=args.p_correct,
                                      n_correct=args.n_correct,
                                      workers=args.workers,
                                      configs=cfgs,
                                      verbose=True,
                                      desc=dname)
            dfile = os.path.join(args.save_dir, f'{dname}.npz')
            np.savez_compressed(dfile, **d)
            if not args.no_worlds:
                wfile = os.path.join(args.save_dir, f'{dname}_worlds.json')
                with open(wfile, 'w') as f:
                    json.dump(dworlds, f)

            if dname == 'train' and args.vis is not None:
                # Save train for vis
                train = d

    if args.vis is not None:
        vis_dir = os.path.join(args.save_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis.visualize(vis_dir, train, n=args.n_vis)
