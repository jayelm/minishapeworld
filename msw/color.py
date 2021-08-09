import aggdraw
import numpy as np
from pyeda.boolalg.bfarray import exprvars
from pyeda.boolalg import expr
import os


class HSVColor:
    def __init__(self, h, s=0.9, v=0.9):
        # Not full saturation so it can sometimes be higher due to variance
        if h > 1.0 or h < 0.0:
            raise ValueError(f"Invalid h value {h} (must be be between 0 and 1)")
        if s > 1.0 or s < 0.0:
            raise ValueError(f"Invalid s value {s} (must be be between 0 and 1)")
        if v > 1.0 or v < 0.0:
            raise ValueError(f"Invalid v value {v} (must be be between 0 and 1)")
        self.h = h
        self.s = s
        self.v = v

    def variant(self, percent):
        # At most it varies percent% in one direction
        h_noise = np.random.uniform(-percent, percent)
        new_h = self.h + h_noise
        # wraparound
        if new_h > 1:
            new_h -= 1
        elif new_h < 0:
            new_h += 1

        # Higher scale for s_noise/v_noise
        s_noise = np.random.uniform(-percent, percent) * 10
        new_s = max(min(self.s + s_noise, 1), 0)

        v_noise = np.random.uniform(-percent, percent) * 10
        new_v = max(min(self.v + v_noise, 1), 0)

        return HSVColor(new_h, new_s, new_v)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"HSV({self.h:.2f}, {self.s:.2f}, {self.v:.2f})"

    def to_rgb(self):
        h = self.h
        s = self.s
        v = self.v

        if s == 0.0:
            return (v, v, v)

        i = int(h * 6.)
        f = (h * 6.) - i
        p = (v * (1. - s))
        q = (v * (1. - s * f))
        t = (v * (1. - s * (1. - f)))
        i %= 6

        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        elif i == 5:
            return (v, p, q)
        assert False

    def to_rgba(self):
        return self.to_rgb() + (1.0, )

    def to_color_integer(self):
        """
        Hacky way to obtain color integer
        """
        (r, g, b) = self.to_rgb()
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        r_str = hex(r)[2:]
        g_str = hex(g)[2:]
        b_str = hex(b)[2:]

        return int(f"ff{r_str}{g_str}{b_str}", 16)


def color_factory(color_, aggdraw_classes):
    def new_cls(color_variance=None):
        new_color_ = color_
        if color_variance is not None and color_variance != 0.0:
            # TODO - vary should take the color and make some different hex values.
            # Also maybe it should be rgb2hex in general.
            new_color_ = new_color_.variant(percent=color_variance)

        rgb = new_color_.to_rgb()
        # To uint8
        rgb = tuple(map(lambda x: int(x * 255), rgb))

        return {
            c: cls(rgb) for c, cls in aggdraw_classes.items()
        }
    return new_cls


COLORS2HSV = {
    "red": HSVColor(0),
    "green": HSVColor(120 / 360),
    "blue": HSVColor(240 / 360),
    "yellow": HSVColor(60 / 360),
    "white": HSVColor(60 / 360, s=0, v=1.0),
    "gray": HSVColor(30 / 360, s=0, v=0.44),
    # Standard amount
    "cyan": HSVColor(180 / 360),
    "orange": HSVColor(30 / 360),
    "purple": HSVColor(273 / 360),
    "magenta": HSVColor(300 / 360),
}


COLORS = ["red", "blue", "green", "yellow", "white", "gray", "cyan", "orange", "purple", "magenta"]

if "SHAPEWORLD_N_COLORS" in os.environ:
    N_COLORS = int(os.environ['SHAPEWORLD_N_COLORS'])
    assert N_COLORS < len(COLORS), f"{N_COLORS} requested, have {len(COLORS)}"
else:
    N_COLORS = 6  # original setting
COLORS = COLORS[:N_COLORS]

AGGDRAW_COLORS = {
    c: color_factory(hsv_c, {"brush": aggdraw.Brush, "pen": aggdraw.Pen})
    for c, hsv_c in COLORS2HSV.items()
}

COLOR_VARS = exprvars('c', len(COLORS))
C2V = dict(zip(COLORS, COLOR_VARS))
V2C = dict(zip(COLOR_VARS, COLORS))
ONEHOT_VAR = expr.OneHot(*COLOR_VARS)


# FIXME - this should be OOP?
def random(colors=None):
    if colors is None:
        return np.random.choice(COLORS)
    else:
        return np.random.choice(colors)


def new(existing_color, colors=None):
    if colors is None:
        colors = COLORS
    if len(colors) == 1 and colors[0] == existing_color:
        raise RuntimeError("No additional colors to generate")
    new_c = existing_color
    while new_c == existing_color:
        new_c = np.random.choice(colors)
    return new_c
