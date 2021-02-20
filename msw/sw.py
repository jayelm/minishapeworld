import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from . import color, config
from . import constants as C
from . import image, shape


class ShapeWorld:
    def __init__(
        self,
        data_type="concept",
        config=config.SingleConfig,
        colors=None,
        shapes=None,
        n_distractors=0,
        unique_distractors=True,
        unrestricted_distractors=True,
    ):
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
        if data_type not in ["concept", "reference", "caption"]:
            raise NotImplementedError("data_type = {}".format(data_type))

        self.data_type = data_type
        self.config = config

        # FIXME - colors/shapes are noop right now.
        if colors is None:
            colors = color.COLORS
        if shapes is None:
            shapes = shape.SHAPES
        self.colors = colors
        self.shapes = shapes

        self.n_distractors = n_distractors
        self.unique_distractors = unique_distractors
        self.unrestricted_distractors = unrestricted_distractors

    def generate(
        self,
        n,
        n_images=10,
        min_correct=None,
        p_correct=0.5,
        n_correct=None,
        float_type=False,
        configs=None,
        lang_type="standard",
        pool=None,
        workers=0,
        verbose=False,
        desc="",
    ):
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
        if lang_type not in ["standard", "simple"]:
            raise NotImplementedError(f"lang_type = {lang_type}")

        if self.data_type == "concept":
            assert n_images > 4, "Too few n_images"
        elif self.data_type == "reference":
            assert n_images > 1, "Too few n_images"
        elif self.data_type == "caption":
            n_images = 1

        if n_correct is not None:
            assert (
                0 < n_correct <= n_images
            ), f"n_correct ({n_correct}) must be > 0 and <= n_images ({n_images})"

        all_imgs = np.zeros((n, n_images, 3, C.DIM, C.DIM), dtype=np.uint8)
        all_labels = np.zeros((n, n_images), dtype=np.uint8)

        mp_args = [
            (n_images, min_correct, p_correct, n_correct, configs, i) for i in range(n)
        ]

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
            all_imgs[i,] = imgs
            all_labels[i,] = labels
            target_configs.append(target_cfg)
            all_configs.append(cfgs)
            wjsons = [cfg.world_json(s) for cfg, s in zip(cfgs, shapes)]
            if self.data_type == "caption":
                # No multiple images per example
                wjsons = wjsons[0]
            world_jsons.append(wjsons)

        if do_mp and pool_was_none:  # Remember to close the pool
            pool.close()
            pool.join()

        if float_type:
            all_imgs = np.divide(all_imgs, 255.0)
            all_labels = all_labels.astype(np.float32)
        langs = np.array(
            [cfg.format(lang_type=lang_type) for cfg in target_configs],
            dtype=np.unicode,
        )

        if self.data_type == "caption":
            # Squeeze out the images per example dim
            all_imgs = all_imgs.squeeze(1)
            all_labels = all_labels.squeeze(1)

        return {"imgs": all_imgs, "labels": all_labels, "langs": langs,}, world_jsons

    def _generate_one_mp(self, mp_args):
        """
        Wrapper around generate_one which accepts a tuple of args (and an index
        i) for multiprocsesing purposes

        mp_args is a tuple of (n_images, min_correct, p_correct, n_correct,
        configs, i); see self.generate_one
        """
        *mp_args, i = mp_args
        return self.generate_one(*mp_args) + (i,)

    def generate_one(
        self, n_images, min_correct=None, p_correct=0.5, n_correct=None, configs=None
    ):
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
        np.random.seed()  # For multiprocessing (TODO: can we avoid this?)
        imgs = np.zeros((n_images, 3, C.DIM, C.DIM), dtype=np.uint8)
        labels = np.zeros((n_images,), dtype=np.uint8)
        if configs is not None:
            cfg_idx = np.random.choice(len(configs))
            target_cfg = configs[cfg_idx]
        else:
            target_cfg = self.config.random()

        if self.data_type == "concept":
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
                label = np.random.random() < p_correct

            new_cfg, new_shapes = target_cfg.instantiate(label)

            # Create image and draw shapes
            img = self.create_image(new_shapes)
            imgs[i] = img
            labels[i] = label
            cfgs.append(new_cfg)
            shapes.append(new_shapes)

        return imgs, labels, target_cfg, cfgs, shapes

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
