from collections import namedtuple

from ... import color, shape
from .. import spec
from .. import util as config_util
from . import configbase

_SingleConfigBase = namedtuple("SingleConfig", ["shape", "color"])


class SingleConfig(configbase._ConfigBase, _SingleConfigBase):
    def format(self, lang_type):
        if lang_type not in ("standard", "simple"):
            raise NotImplementedError(f"lang_type = {lang_type}")
        color_, shape_ = self
        shape_txt = "shape"
        color_txt = ""
        if shape_ is not None:
            shape_txt = shape_
        if color_ is not None:
            color_txt = color_ + " "
        if lang_type == "standard":
            exists_txt = "there is {} ".format(
                config_util.a_or_an(color_txt + shape_txt)
            )
            period_txt = " ."
        else:
            exists_txt = ""
            period_txt = ""
        return f"{exists_txt}{color_txt}{shape_txt}{period_txt}"

    def __str__(self):
        return self.format(lang_type="standard")

    def json(self):
        return {"type": "single", "shape": self.shape, "color": self.color}

    def instantiate(self, label):
        new_cfg = self if label else self.invalidate()

        color_, shape_ = new_cfg
        if shape_ is None:
            shape_ = shape.random_shape()
        if color_ is None:
            color_ = color.random_color()
        s = shape.SHAPE_IMPLS[shape_](color_=color_)

        return new_cfg, [s]

    @classmethod
    def random_config_single(cls):
        shape_spec = spec.ShapeSpec.random()
        shape_ = shape_spec.random_shape()
        return cls(*shape_)
