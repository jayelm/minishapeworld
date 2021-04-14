from collections import namedtuple
import numpy as np

from ... import color, shape
from .. import spec
from .. import util as config_util
from . import configbase

_SingleConfigBase = namedtuple("SingleConfig", ["shape", "color"])


class SingleConfig(configbase._ConfigBase, _SingleConfigBase):
    def format(self, lang_type):
        if lang_type not in ("standard", "simple", "conjunction"):
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

        if lang_type == "conjunction":
            conj_txt = " and "
        else:
            conj_txt = ""

        return f"{exists_txt}{color_txt}{conj_txt}{shape_txt}{period_txt}"

    def __str__(self):
        return self.format(lang_type="standard")

    def json(self):
        return {"type": "single", "shape": self.shape, "color": self.color}

    def instantiate(self, label, shape_kwargs=None, **kwargs):
        new_cfg = self if label else self.invalidate()

        s = self.add_shape(new_cfg, shape_kwargs=shape_kwargs)
        return new_cfg, [s]

    def invalidate(self):
        color_, shape_ = self
        if shape_ is not None and color_ is not None:
            # Sample random part to invalidate
            # Here, we can invalidate shape, or invalidate color, OR
            # invalidate both
            part_to_invalidate = np.random.randint(3)
            if part_to_invalidate == 0:
                return type(self)(color.new(color_), shape_)
            elif part_to_invalidate == 1:
                return type(self)(color_, shape.new(shape_))
            elif part_to_invalidate == 2:
                return type(self)(color.new(color_), shape.new(shape_))
            else:
                raise RuntimeError
        elif shape_ is not None:
            assert color_ is None
            return type(self)(None, shape.new(shape_))
        elif color_ is not None:
            assert shape_ is None
            return type(self)(color.new(color_), None)
        else:
            raise RuntimeError

    @classmethod
    def random(cls):
        shape_spec = spec.ShapeSpec.random()
        return cls(*shape_spec)

    @classmethod
    def enumerate_both(cls):
        configs = []
        for spc in spec.ShapeSpec.enumerate_both():
            configs.append(cls(*spc))
        return configs

    @classmethod
    def enumerate(cls):
        # For now, only enumerate through defined shapes
        return cls.enumerate_both()
