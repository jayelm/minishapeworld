from enum import Enum

import numpy as np

from .. import color, shape


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2

    def random_shape(self):
        color_ = None
        shape_ = None
        if self == self.SHAPE:
            shape_ = shape.random()
        elif self == self.COLOR:
            color_ = color.random()
        else:
            shape_ = shape.random()
            color_ = color.random()
        return (color_, shape_)

    @classmethod
    def random(cls):
        return cls(np.random.randint(len(cls)))


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


SHAPE_SPECS = list(ShapeSpec)
