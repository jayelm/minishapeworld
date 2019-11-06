"""
Config implementations
"""

from enum import Enum
from collections import namedtuple

_SpatialConfigBase = namedtuple('SpatialConfig', ['shapes', 'relation', 'dir'])
_SingleConfigBase = namedtuple('SingleConfig', ['shape', 'color'])


def matches(spec, shape_):
    """
    Return True if the shape adheres to the spec (spec has optional color/shape
    restrictions)
    """
    (c, s) = spec
    matches_color = c is None or (shape_.color == c)
    matches_shape = s is None or (shape_.name == s)
    return matches_color and matches_shape


def has_relation(d1, d2, relation, relation_dir):
    if relation == 0:
        if relation_dir == 0:
            return d1.left(d2)
        else:
            return d1.right(d2)
    else:
        if relation_dir == 0:
            return d1.below(d2)
        else:
            return d1.above(d2)


class SpatialConfig(_SpatialConfigBase):
    def matches_shapes(self, shape_):
        """
        Return a list which contains 0 or 1 if the shape adheres to the match
        """
        m = []
        for i, spec in enumerate(self.shapes):
            if matches(spec, shape_):
                m.append(i)
        return m

    def does_not_validate(self, existing_shapes, new_shape):
        """
        Check if a new shape has the config's spatial relation with any
        existing shapes.
        Only compares a single new shape with existing shapes O(n) rather than
        pairwise comparisons among existing shapes (O(n^2)).
        """
        #  import ipdb; ipdb.set_trace()
        for match_i in self.matches_shapes(new_shape):
            shapes = [None, None]
            # target is the opposite
            target_i = 1 - match_i
            shapes[match_i] = new_shape
            for es in existing_shapes:
                shapes[target_i] = es
                if matches(self.shapes[target_i], es):
                    if has_relation(*shapes, self.relation, self.dir):
                        return False
        return True

    def __str__(self):
        (s1, s2), relation, relation_dir = self
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
        parts = [s1_0_txt, s1_1_txt, rel_txt, s2_0_txt, s2_1_txt]
        return ' '.join(s for s in parts if s != '')


class SingleConfig(_SingleConfigBase):
    def __str__(self):
        color_, shape_ = self
        shape_txt = 'shape'
        color_txt = ''
        if shape_ is not None:
            shape_txt = shape_
        if color_ is not None:
            color_txt = color_ + ' '
        return '{}{}'.format(color_txt, shape_txt)


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


SHAPE_SPECS = list(ShapeSpec)
