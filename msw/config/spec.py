from enum import Enum
from collections import namedtuple
import itertools

from pyeda.boolalg import expr

import numpy as np

from .. import color, shape


class ShapeSpecTypes(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


_ShapeSpecBase = namedtuple('ShapeSpec', ["color", "shape"])


class ShapeSpec(_ShapeSpecBase):
    def __init__(self, *args, **kwargs):
        #  super().__new__(*args, **kwargs)
        if self.color is None and self.shape is None:
            raise ValueError("Generic shapes not yet supported")
        elif self.color is not None and self.shape is not None:
            self.spec_type = ShapeSpecTypes.BOTH
        elif self.color is not None:
            self.spec_type = ShapeSpecTypes.COLOR
        elif self.shape is not None:
            self.spec_type = ShapeSpecTypes.SHAPE
        else:
            assert False
        # Calculate the type

    @classmethod
    def enumerate(cls):
        """
        Generator to loop through all possible shapespecs
        """
        yield from cls.enumerate_color()
        yield from cls.enumerate_shape()
        yield from cls.enumerate_both()

    @classmethod
    def enumerate_color(cls):
        for color_ in color.COLORS:
            yield cls(color_, None)

    @classmethod
    def enumerate_shape(cls):
        for shape_ in shape.SHAPES:
            yield cls(None, shape_)

    @classmethod
    def enumerate_both(cls):
        for color_, shape_ in itertools.product(color.COLORS, shape.SHAPES):
            yield cls(color_, shape_)

    @classmethod
    def random(cls):
        spec_type = np.random.choice(ShapeSpecTypes)

        color_ = None
        shape_ = None
        if spec_type == ShapeSpecTypes.SHAPE:
            shape_ = shape.random()
        elif spec_type == ShapeSpecTypes.COLOR:
            color_ = color.random()
        else:
            shape_ = shape.random()
            color_ = color.random()

        return cls(color_, shape_)

    def to_expr(self):
        if self.spec_type == ShapeSpecTypes.SHAPE:
            return shape.S2V[self.shape]
        elif self.spec_type == ShapeSpecTypes.COLOR:
            return color.C2V[self.color]
        else:
            return expr.And(
                color.C2V[self.color],
                shape.S2V[self.shape],
            )
