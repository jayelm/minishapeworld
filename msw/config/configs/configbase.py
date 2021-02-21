import abc

import numpy as np
from tqdm import tqdm

from ... import color
from ... import constants as C
from ... import shape


class _ConfigBase:
    @abc.abstractmethod
    def format(self, lang_type="standard"):
        return ""

    @abc.abstractmethod
    def invalidate(self):
        return

    @abc.abstractmethod
    def instantiate(self, label, **kwargs):
        """
        Actually create shapes according to label
        """
        return

    @abc.abstractmethod
    def json(self):
        return {}

    @abc.abstractclassmethod
    def random(cls):
        return

    def __str__(self):
        return self.format(lang_type="standard")

    def sample_distractor(self, existing_shapes=()):
        d = (color.random(), shape.random())
        if d in existing_shapes:
            return self.sample_distractor(existing_shapes=existing_shapes)
        return d

    def sample_distractors(self, n_distractors, existing_shapes=()):
        n_dist = self.sample_n_distractor(n_distractors)

        distractors = []
        for _ in range(n_dist):
            d = self.sample_distractor()
            distractors.append(d)
        return distractors

    def sample_n_distractor(self, n_distractors):
        if isinstance(n_distractors, tuple):
            n_dist = np.random.randint(
                n_distractors[0], n_distractors[1] + 1
            )  # Exclusive range
        else:
            n_dist = n_distractors
        return n_dist

    def world_json(self, shapes, lang_type="standard"):
        return {
            "lang": self.format(lang_type=lang_type),
            "config": self.json(),
            "shapes": [s.json() for s in shapes],
        }

    def add_shape_from_spec(self, spec, relation, relation_dir, shapes=None, attempt=1):
        if attempt > C.MAX_PLACEMENT_ATTEMPTS:
            return None
        color_, shape_ = spec
        if shape_ is None:
            shape_ = shape.random()
        if color_ is None:
            color_ = color.random()
        s = shape.SHAPE_IMPLS[shape_](
            relation=relation, relation_dir=relation_dir, color_=color_
        )
        if shapes is not None:
            for oth in shapes:
                if s.intersects(oth):
                    return self.add_shape_from_spec(
                        spec, relation, relation_dir, shapes=shapes, attempt=attempt + 1
                    )
            shapes.append(s)
            return s
        return s

    def add_shape(self, spec):
        """
        Add shape according to spec
        """
        color_, shape_ = spec
        if shape_ is None:
            shape_ = shape.random()
        if color_ is None:
            color_ = color.random()
        x = shape.rand_pos()
        y = shape.rand_pos()
        return shape.SHAPE_IMPLS[shape_](x=x, y=y, color_=color_)

    @classmethod
    def generate(cls, n, verbose=False):
        """
        Generate unique configs
        """
        total_configs = set()

        if verbose:
            pbar = tqdm(total=n)

        while len(total_configs) < n:
            new_cfg = cls.random()
            if new_cfg not in total_configs:
                total_configs.add(new_cfg)
                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()

        return list(total_configs)
