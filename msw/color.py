import aggdraw
import numpy as np
from pyeda.boolalg.bfarray import exprvars
from pyeda.boolalg import expr

COLORS = ["red", "blue", "green", "yellow", "white", "gray"]
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}

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
