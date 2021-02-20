from collections import namedtuple
from enum import Enum

import numpy as np

from ... import color
from ... import constants as C
from ... import shape
from .. import spec
from .. import util as config_util
from . import configbase

_SpatialConfigBase = namedtuple("SpatialConfig", ["shapes", "relation", "dir"])


def matches(spc, shape_):
    """
    Return True if the shape adheres to the spc (spc has optional color/shape
    restrictions)
    """
    (c, s) = spc
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


class SpatialConfig(configbase._ConfigBase, _SpatialConfigBase):
    def matches_shapes(self, shape_):
        """
        Return a list which contains 0 or 1 if the shape adheres to the match
        """
        m = []
        for i, spc in enumerate(self.shapes):
            if matches(spc, shape_):
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

    def format(self, lang_type="standard"):
        if lang_type not in ("standard", "simple"):
            raise NotImplementedError(f"lang_type = {lang_type}")
        (s1, s2), relation, relation_dir = self
        if relation == 0:
            if relation_dir == 0:
                if lang_type == "standard":
                    rel_txt = "is to the left of"
                else:
                    rel_txt = "left"
            else:
                if lang_type == "standard":
                    rel_txt = "is to the right of"
                else:
                    rel_txt = "right"
        else:
            if relation_dir == 0:
                if lang_type == "standard":
                    rel_txt = "is below"
                else:
                    rel_txt = "below"
            else:
                if lang_type == "standard":
                    rel_txt = "is above"
                else:
                    rel_txt = "above"
        if s1[0] is None:
            s1_0_txt = ""
        else:
            s1_0_txt = s1[0]
        if s1[1] is None:
            s1_1_txt = "shape"
        else:
            s1_1_txt = s1[1]
        if s2[0] is None:
            s2_0_txt = ""
        else:
            s2_0_txt = s2[0]
        if s2[1] is None:
            s2_1_txt = "shape"
        else:
            s2_1_txt = s2[1]
        if lang_type == "standard":
            s1_article = config_util.a_or_an(s1_0_txt + s1_1_txt)
            s2_article = config_util.a_or_an(s2_0_txt + s2_1_txt)
            period_txt = "."
        else:
            s1_article = ""
            s2_article = ""
            period_txt = ""
        parts = [
            s1_article,
            s1_0_txt,
            s1_1_txt,
            rel_txt,
            s2_article,
            s2_0_txt,
            s2_1_txt,
            period_txt,
        ]
        return " ".join(s for s in parts if s != "")

    def json(self):
        return {
            "type": "spatial",
            "shapes": [{"color": s[0], "shape": s[1]} for s in self.shapes],
            "relation": self.relation,
            "relation_dir": self.dir,
        }

    @classmethod
    def random(cls):
        # 0 -> only shape specified
        # 1 -> only color specified
        # 2 -> only both specified
        shape_1 = spec.ShapeSpec.random()
        shape_2 = spec.ShapeSpec.random()
        if shape_1 == shape_2:
            return cls.random()  # Try again
        relation = np.random.randint(2)
        relation_dir = np.random.randint(2)
        return cls((shape_1, shape_2), relation, relation_dir)

    def invalidate(self):
        # Invalidate by randomly choosing one property to change:
        (
            (shape_1, shape_2),
            relation,
            relation_dir,
        ) = self
        properties = []
        if shape_1.color is not None:
            properties.append(SpatialInvalidateProps.SHAPE_1_COLOR)
        if shape_1.shape is not None:
            properties.append(SpatialInvalidateProps.SHAPE_1_SHAPE)
        if shape_2.color is not None:
            properties.append(SpatialInvalidateProps.SHAPE_2_COLOR)
        if shape_2.shape is not None:
            properties.append(SpatialInvalidateProps.SHAPE_2_SHAPE)
        sp = len(properties)
        # Invalidate relations half of the time
        for _ in range(sp):
            properties.append(SpatialInvalidateProps.RELATION_DIR)
        # Randomly select property to invalidate
        # TODO: Support for invalidating multiple properties
        invalid_prop = np.random.choice(properties)

        if invalid_prop == SpatialInvalidateProps.SHAPE_1_COLOR:
            inv_cfg = (
                (
                    spec.ShapeSpec(color.new_color(shape_1.color), shape_1.shape),
                    shape_2,
                ),
                relation,
                relation_dir,
            )
        elif invalid_prop == SpatialInvalidateProps.SHAPE_1_SHAPE:
            inv_cfg = (
                (
                    spec.ShapeSpec(shape_1.color, shape.new_shape(shape_1.shape)),
                    shape_2,
                ),
                relation,
                relation_dir,
            )
        elif invalid_prop == SpatialInvalidateProps.SHAPE_2_COLOR:
            inv_cfg = (
                (
                    shape_1,
                    spec.ShapeSpec(color.new_color(shape_2.color), shape_2.shape),
                ),
                relation,
                relation_dir,
            )
        elif invalid_prop == SpatialInvalidateProps.SHAPE_2_SHAPE:
            inv_cfg = (
                (
                    shape_1,
                    spec.ShapeSpec(shape_2.color, shape.new_shape(shape_2.shape)),
                ),
                relation,
                relation_dir,
            )
        elif invalid_prop == SpatialInvalidateProps.RELATION_DIR:
            inv_cfg = (
                (shape_1, shape_2),
                relation,
                1 - relation_dir,
            )
        else:
            raise RuntimeError
        return type(self)(*inv_cfg)

    def instantiate(self, label, n_distractors=0):
        """
        Generate a single spatial relation according to the config,
        invalidating it if the label is 0.
        """
        cfg_attempts = 0
        while cfg_attempts < C.MAX_INVALIDATE_ATTEMPTS:
            new_cfg = self if label else self.invalidate()
            (ss1, ss2), relation, relation_dir = new_cfg
            s2 = self.add_shape_from_spec(ss2, relation, relation_dir)

            # Place second shape
            attempts = 0
            while attempts < C.MAX_PLACEMENT_ATTEMPTS:
                s1 = self.add_shape_rel(ss1, s2, relation, relation_dir)
                if not s2.intersects(s1):
                    break
                attempts += 1
            else:
                raise RuntimeError(
                    "Could not place shape onto image without intersection"
                )

            if label:  # Positive example
                break
            elif self.does_not_validate([s1], s2):
                break

            cfg_attempts += 1

        # Place distractor shapes
        shapes = [s1, s2]
        n_dist = self.sample_n_distractor(n_distractors)
        for _ in range(n_dist):
            attempts = 0
            while attempts < C.MAX_DISTRACTOR_PLACEMENT_ATTEMPTS:
                dss = self.sample_distractor(existing_shapes=shapes)
                ds = self.add_shape(dss)
                # No intersections
                if not any(ds.intersects(s) for s in shapes):
                    if label:
                        shapes.append(ds)
                        break
                    else:
                        # If this is a *negative* example, we should not have
                        # the relation expressed by the original config
                        if self.does_not_validate(shapes, ds):
                            shapes.append(ds)
                            break
                attempts += 1
            else:
                raise RuntimeError(
                    "Could not place distractor onto " "image without intersection"
                )

        return new_cfg, shapes

    def add_shape_rel(self, spec, oth_shape, relation, relation_dir):
        """
        Add shape, obeying the relation/relation_dir w.r.t. oth shape
        """
        color_, shape_ = spec
        if shape_ is None:
            shape_ = shape.random()
        if color_ is None:
            color_ = color.random()
        if relation == 0:
            new_y = shape.rand_pos()
            if relation_dir == 0:
                # Shape must be LEFT of oth shape
                new_x = np.random.randint(C.X_MIN, oth_shape.x - C.BUFFER)
            else:
                # Shape RIGHT of oth shape
                new_x = np.random.randint(oth_shape.x + C.BUFFER, C.X_MAX)
        else:
            new_x = shape.rand_pos()
            if relation_dir == 0:
                # BELOW (remember y coords reversed)
                new_y = np.random.randint(oth_shape.y + C.BUFFER, C.X_MAX)
            else:
                # ABOVE
                new_y = np.random.randint(C.X_MIN, oth_shape.y - C.BUFFER)
        return shape.SHAPE_IMPLS[shape_](x=new_x, y=new_y, color_=color_)


class SpatialInvalidateProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4
