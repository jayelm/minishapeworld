"""
Generate shapeworld reference games
"""

import numpy as np
from numpy import random
from tqdm import tqdm
import os
import multiprocessing as mp

import image
import color
import shape
import config
import constants as c
import vis


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

        self.data_type = data_type
        self.img_type = img_type
        if img_type not in ['spatial', 'single']:
            raise NotImplementedError("img_type = {}".format(img_type))
        if img_type == 'spatial':
            self.img_func = self.generate_spatial
        elif img_type == 'single':
            self.img_func = self.generate_single

        if colors is None:
            colors = color.COLORS
        if shapes is None:
            shapes = shape.SHAPES
        self.colors = colors
        self.shapes = shapes

        self.n_distractors = n_distractors
        self.unique_distractors = unique_distractors
        self.unrestricted_distractors = unrestricted_distractors

    def generate(self,
                 n,
                 n_images=10,
                 correct=0.5,
                 float_type=False,
                 pool=None,
                 workers=0,
                 verbose=False):
        do_mp = workers > 0
        if not do_mp and pool is not None:
            raise ValueError("Can't specify pool if do_mp=True")
        if do_mp:
            pool_was_none = False
            if pool is None:
                pool_was_none = True
                pool = mp.Pool(workers)

        if self.data_type == 'concept':
            assert n_images > 4, "Too few n_images"
        elif self.data_type == 'reference':
            assert n_images > 1, "Too few n_images"
        elif self.data_type == 'caption':
            n_images = 1

        all_imgs = np.zeros((n, n_images, 64, 64, 3), dtype=np.uint8)
        all_labels = np.zeros((n, n_images), dtype=np.uint8)
        configs = []

        mp_args = [(n_images, correct, i) for i in range(n)]

        if do_mp:
            gen_iter = pool.imap(self.img_func, mp_args)
        else:
            gen_iter = map(self.img_func, mp_args)
        if verbose:
            gen_iter = tqdm(gen_iter, total=n)
        for imgs, labels, cfg, i in gen_iter:
            all_imgs[i, ] = imgs
            all_labels[i, ] = labels
            configs.append(cfg)

        if do_mp and pool_was_none:  # Remember to close the pool
            pool.close()
            pool.join()

        if float_type:
            all_imgs = np.divide(all_imgs, 255.0)
            all_labels = all_labels.astype(np.float32)
        langs = np.array([fmt_config(c) for c in configs], dtype=np.unicode)

        if self.data_type == 'caption':
            # Squeeze out the images per example dim
            all_imgs = all_imgs.squeeze(1)
            all_labels = all_labels.squeeze(1)

        return {'imgs': all_imgs, 'labels': all_labels, 'langs': langs}

    def generate_spatial(self, mp_args):
        """
        Generate a single image
        """
        random.seed()
        n_images, correct, i = mp_args
        # Get shapes and relations
        imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
        labels = np.zeros((n_images, ), dtype=np.uint8)
        cfg = self.random_config_spatial()
        # Minimum of 2 correct worlds/2 distractors
        if self.data_type == 'concept':
            n_target = 2
            n_distract = 2
        else:
            n_target = 1
            n_distract = n_images  # Never run out of distractors
        idx_rand = list(range(n_images))

        for w_idx in idx_rand:
            if n_target > 0:
                label = 1
                n_target -= 1
            elif n_distract > 0:
                label = 0
                n_distract -= 1
            else:
                label = (random.random() < correct)
            new_cfg = cfg if label else self.invalidate_spatial(cfg)
            (ss1, ss2), relation, relation_dir = new_cfg
            s2 = self.add_shape_from_spec(ss2, relation, relation_dir)

            # Place second shape
            attempts = 0
            while attempts < c.MAX_PLACEMENT_ATTEMPTS:
                s1 = self.add_shape_rel(ss1, s2, relation, relation_dir)
                if not s2.intersects(s1):
                    break
            else:
                # Failed
                raise RuntimeError(
                    "Could not place shape onto image without intersection")

            # Place distractor shapes
            existing_shapes = [s1, s2]
            distractors = self.sample_distractors(
                existing_shapes=existing_shapes)
            for dss in distractors:
                attempts = 0
                while attempts < c.MAX_DISTRACTOR_PLACEMENT_ATTEMPTS:
                    ds = self.add_shape(dss)
                    if not any(ds.intersects(s) for s in existing_shapes):
                        existing_shapes.append(ds)
                        break
                else:
                    # Failed
                    raise RuntimeError("Could not place distractor onto "
                                       "image without intersection")

            # Create image and draw shapes
            img = image.IMG()
            img.draw_shapes(existing_shapes)
            imgs[w_idx] = img.array()
            labels[w_idx] = label
        return imgs, labels, cfg, i

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
        properties.append(config.ConfigProps.RELATION_DIR)
        # Randomly select property to invalidate
        # TODO: Support for invalidating multiple properties
        invalid_prop = random.choice(properties)

        if invalid_prop == config.ConfigProps.SHAPE_1_COLOR:
            return ((self.new_color(shape_1_color), shape_1_shape),
                    (shape_2_color, shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_1_SHAPE:
            return ((shape_1_color, self.new_shape(shape_1_shape)),
                    (shape_2_color, shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_2_COLOR:
            return ((shape_1_color, shape_1_shape),
                    (self.new_color(shape_2_color),
                     shape_2_shape)), relation, relation_dir
        elif invalid_prop == config.ConfigProps.SHAPE_2_SHAPE:
            return ((shape_1_color, shape_1_shape),
                    (shape_2_color,
                     self.new_shape(shape_2_shape))), relation, relation_dir
        elif invalid_prop == config.ConfigProps.RELATION_DIR:
            return ((shape_1_color, shape_1_shape),
                    (shape_2_color, shape_2_shape)), relation, 1 - relation_dir
        else:
            raise RuntimeError

    def generate_single(self, mp_args):
        random.seed()
        n_images, correct, i = mp_args
        imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
        labels = np.zeros((n_images, ), dtype=np.uint8)
        cfg = self.random_config_single()
        if self.data_type == 'concept':
            n_target = 2
            n_distract = 2
        else:
            n_target = 1
            n_distract = n_images  # Never run out of distractors
        idx_rand = list(range(n_images))
        # random.shuffle(idx_rand)
        for w_idx in idx_rand:
            if n_target > 0:
                label = 1
                n_target -= 1
            elif n_distract > 0:
                label = 0
                n_distract -= 1
            else:
                label = (random.random() < correct)
            new_cfg = cfg if label else self.invalidate_single(cfg)

            color_, shape_ = new_cfg
            if shape_ is None:
                shape_ = self.random_shape()
            if color_ is None:
                color_ = self.random_color()
            s = shape.SHAPE_IMPLS[shape_](color_=color_)

            # Create image and draw shape
            img = image.IMG()
            img.draw_shapes([s])
            imgs[w_idx] = img.array()
            labels[w_idx] = label
        return imgs, labels, cfg, i

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

    def sample_distractors(self, existing_shapes=()):
        if isinstance(self.n_distractors, tuple):
            n_dist = random.randint(self.n_distractors[0],
                                    self.n_distractors[1] +
                                    1)  # Exclusive range
        else:
            n_dist = self.n_distractors

        distractors = []
        for _ in range(n_dist):
            d = None
            while d is None or d in existing_shapes:
                d = (self.random_color(
                    unrestricted=self.unrestricted_distractors),
                     self.random_shape(
                         unrestricted=self.unrestricted_distractors))
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
        return config.SpatialConfig([shape_1, shape_2], relation, relation_dir)

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


def fmt_config(cfg):
    if isinstance(cfg, config.SingleConfig):
        return _fmt_config_single(cfg)
    elif isinstance(cfg, config.SpatialConfig):
        return _fmt_config_spatial(cfg)
    else:
        raise NotImplementedError(type(cfg))


def _fmt_config_single(cfg):
    color_, shape_ = cfg
    shape_txt = 'shape'
    color_txt = ''
    if shape_ is not None:
        shape_txt = shape_
    if color_ is not None:
        color_txt = color_ + ' '
    return '{}{}'.format(color_txt, shape_txt)


def _fmt_config_spatial(cfg):
    (s1, s2), relation, relation_dir = cfg
    if relation == 0:
        if relation_dir == 0:
            rel_txt = 'left'
        else:
            rel_txt = 'right'
    else:
        if relation_dir == 0:
            rel_txt = 'below'
        else:
            rel_txt = 'above'
    if s1[0] is None:
        s1_0_txt = ''
    else:
        s1_0_txt = s1[0]
    if s1[1] is None:
        s1_1_txt = 'shape'
    else:
        s1_1_txt = s1[1]
    if s2[0] is None:
        s2_0_txt = ''
    else:
        s2_0_txt = s2[0]
    if s2[1] is None:
        s2_1_txt = 'shape'
    else:
        s2_1_txt = s2[1]
    return '{} {} {} {} {}'.format(s1_0_txt, s1_1_txt, rel_txt, s2_0_txt,
                                   s2_1_txt)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Fast ShapeWorld',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_examples',
                        type=int,
                        default=100,
                        help='Number of examples')
    parser.add_argument('--n_images',
                        type=int,
                        default=10,
                        help='Images per example (concept/reference only)')
    parser.add_argument('--correct',
                        type=float,
                        default=0.5,
                        help='Avg correct proportion of images (concept only)')
    parser.add_argument('--workers',
                        default=0,
                        type=int,
                        help="Number of workers (0 = no multiprocessing)")
    parser.add_argument('--data_type',
                        choices=['concept', 'reference', 'caption'],
                        default='concept',
                        help='What kind of data to generate')
    parser.add_argument(
        '--n_distractors',
        default=[2, 3],
        nargs='*',
        type=int,
        help='Number of distractor shapes (for spatial only); '
             'either one int or (min, max)'
    )
    parser.add_argument('--img_type',
                        choices=['single', 'spatial'],
                        default='spatial',
                        help='What kind of images to generate')
    parser.add_argument(
        '--vis_dir',
        default=None,
        type=str,
        help='If specified, save sample visualization of data '
             '(100 images) to this folder'
    )
    parser.add_argument('--n_vis',
                        default=100,
                        type=int,
                        help='How many examples to visualize?')
    parser.add_argument('--save',
                        default='test.npz',
                        help='Save dataset to this file')

    args = parser.parse_args()

    if len(args.n_distractors) == 1:
        args.n_distractors = args.n_distractors[0]
    elif len(args.n_distractors) == 2:
        args.n_distractors = tuple(args.n_distractors)
    else:
        parser.error("--n_distractors must be either 1 int or 2 (min, max)")

    msw = MiniShapeWorld(data_type=args.data_type,
                         img_type=args.img_type,
                         n_distractors=args.n_distractors)
    data = msw.generate(args.n_examples,
                        n_images=args.n_images,
                        correct=args.correct,
                        workers=args.workers,
                        verbose=True)

    np.savez_compressed(args.save, **data)

    if args.vis_dir is not None:
        os.makedirs(args.vis_dir, exist_ok=True)
        vis.visualize(args.vis_dir, data, n=args.n_vis)
