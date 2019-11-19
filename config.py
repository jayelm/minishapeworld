"""
Config implementations
"""

from enum import Enum
from collections import namedtuple

_SpatialConfigBase = namedtuple('SpatialConfig', ['shapes', 'relation', 'dir'])
_SingleConfigBase = namedtuple('SingleConfig', ['shape', 'color'])


def a_or_an(s):
    """
    Return `a` or `an` depending on the first voewl sound of s. Dumb heuristic
    """
    if s[0].lower() in 'aeiou':
        return 'an'
    return 'a'


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

    def format(self, lang_type='standard'):
        if lang_type not in ('standard', 'simple'):
            raise NotImplementedError(f"lang_type = {lang_type}")
        (s1, s2), relation, relation_dir = self
        if relation == 0:
            if relation_dir == 0:
                if lang_type == 'standard':
                    rel_txt = 'is to the left of'
                else:
                    rel_txt = 'left'
            else:
                if lang_type == 'standard':
                    rel_txt = 'is to the right of'
                else:
                    rel_txt = 'right'
        else:
            if relation_dir == 0:
                if lang_type == 'standard':
                    rel_txt = 'is below'
                else:
                    rel_txt = 'below'
            else:
                if lang_type == 'standard':
                    rel_txt = 'is above'
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
        if lang_type == 'standard':
            s1_article = a_or_an(s1_0_txt + s1_1_txt)
            s2_article = a_or_an(s2_0_txt + s2_1_txt)
            period_txt = '.'
        else:
            s1_article = ''
            s2_article = ''
            period_txt = ''
        parts = [s1_article, s1_0_txt, s1_1_txt, rel_txt, s2_article, s2_0_txt, s2_1_txt, period_txt]
        return ' '.join(s for s in parts if s != '')

    def __str__(self):
        return self.format(lang_type='standard')


class SingleConfig(_SingleConfigBase):
    def format(self, lang_type):
        if lang_type not in ('standard', 'simple'):
            raise NotImplementedError(f"lang_type = {lang_type}")
        color_, shape_ = self
        shape_txt = 'shape'
        color_txt = ''
        if shape_ is not None:
            shape_txt = shape_
        if color_ is not None:
            color_txt = color_ + ' '
        if lang_type == 'standard':
            exists_txt = 'there is {} '.format(a_or_an(color_txt + shape_txt))
            period_txt = ' .'
        else:
            exists_txt = ''
            period_txt = ''
        return f"{exists_txt}{color_txt}{shape_txt}{period_txt}"

    def __str__(self):
        return self.format(lang_type='standard')


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
