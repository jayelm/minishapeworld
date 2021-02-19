import aggdraw
import numpy as np

COLORS = ["red", "blue", "green", "yellow", "white", "gray"]
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


# FIXME - this should be OOP?
def random(colors=None):
    if colors is None:
        return np.random.choice(COLORS)
    else:
        return np.random.choice(colors)


def new_color(existing_color, colors=None):
    if colors is None:
        colors = COLORS
    if len(colors) == 1 and colors[0] == existing_color:
        raise RuntimeError("No additional colors to generate")
    new_c = existing_color
    while new_c == existing_color:
        new_c = np.random.choice(colors)
    return new_c
