"""
Shape implementations
"""

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from pyeda.boolalg.bfarray import exprvars
from pyeda.boolalg import expr
import math
import os

from . import color
from . import constants as C


def rand_size():
    return np.random.randint(C.SIZE_MIN, C.SIZE_MAX)


def rand_size_2():
    """Slightly bigger."""
    return np.random.randint(C.SIZE_MIN + 2, C.SIZE_MAX + 2)


def rand_size_3():
    """Slightly bigger, with a bigger minimum"""
    return np.random.randint(C.SIZE_MIN + 4, C.SIZE_MAX + 2)


def rand_pos():
    return np.random.randint(C.X_MIN, C.X_MAX)


class Shape:
    def __init__(
        self,
        x=None,
        y=None,
        relation=None,
        relation_dir=None,
        color_=None,
        max_rotation=90,
        color_variance=None,
    ):
        if color_ is None:
            raise NotImplementedError("Must specify color")
        self.color = color_
        if x is not None or y is not None:
            assert x is not None and y is not None
            assert relation is None and relation_dir is None
            self.x = x
            self.y = y
        elif relation is None and relation_dir is None:
            self.x = rand_pos()
            self.y = rand_pos()
        else:
            # Generate on 3/4 of image according to relation dir
            if relation == 0:
                # x matters - y is totally random
                self.y = rand_pos()
                if relation_dir == 0:
                    # Place right 3/4 of screen, so second shape
                    # can be placed LEFT
                    self.x = np.random.randint(C.X_MIN_34, C.X_MAX)
                else:
                    # Place left 3/4
                    self.x = np.random.randint(C.X_MIN, C.X_MAX_34)
            else:
                # y matters - x is totally random
                self.x = rand_pos()
                if relation_dir == 0:
                    # Place top 3/4 of screen, so second shape can be placed
                    # BELOW
                    # NOTE: Remember coords for y travel in opp dir
                    self.y = np.random.randint(C.X_MIN, C.X_MAX_34)
                else:
                    self.y = np.random.randint(C.X_MIN_34, C.X_MAX)
        self.rotation = np.random.randint(max_rotation)
        self.color_variance = color_variance
        self.init_shape()

    def draw(self, image):
        acol = color.AGGDRAW_COLORS[self.color](self.color_variance)
        image.draw.polygon(self.coords, acol["brush"], acol["pen"])

    def intersects(self, oth):
        return self.shape.intersects(oth.shape)

    def left(self, oth):
        return self.x < oth.x

    def right(self, oth):
        return self.x > oth.x

    def above(self, oth):
        return self.y < oth.y

    def below(self, oth):
        return self.y > oth.y

    @property
    def name(self):
        return type(self).__name__.lower()

    def __str__(self):
        return f"<{self.color} {self.name} at ({self.x}, {self.y})>"

    def __repr__(self):
        return self.__str__()

    def json(self):
        return {
            "shape": self.name,
            "color": self.color,
            "pos": {"x": self.x, "y": self.y},
            # "rot": self.rotation,  # I don't think I ever use this
        }


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= C.SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > C.SIZE_MIN, "{} {}".format(smaller, self.dx)
            self.dy = np.random.randint(C.SIZE_MIN, smaller)
        else:
            self.dy = np.random.randint(bigger, C.SIZE_MAX)
        if np.random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape_ = Point(self.x, self.y).buffer(1)
        shape_ = affinity.scale(shape_, self.dx, self.dy)
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = np.round(np.array(self.shape.boundary).astype(np.int))
        self.coords = np.unique(self.coords, axis=0).flatten()


class Circle(Ellipse):
    def init_shape(self):
        self.r = rand_size()
        self.shape = Point(self.x, self.y).buffer(self.r)
        self.coords = [int(x) for x in self.shape.bounds]

    def draw(self, image):
        acol = color.AGGDRAW_COLORS[self.color](self.color_variance)
        image.draw.ellipse(self.coords, acol["brush"])


class Semicircle(Circle):
    def draw(self, image):
        start_deg = np.random.randint(180)
        end_deg = start_deg + 180
        acol = color.AGGDRAW_COLORS[self.color](self.color_variance)
        image.draw.pieslice(self.coords, start_deg, end_deg, acol["brush"], acol["pen"])


class Rectangle(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size_2()
        bigger = int(self.dx * min_skew)
        if bigger >= C.SIZE_MAX:
            smaller = int(self.dx / min_skew)
            self.dy = np.random.randint(C.SIZE_MIN, smaller)
        else:
            self.dy = np.random.randint(bigger, C.SIZE_MAX)
        if np.random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape_ = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        # Get coords
        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size_2()
        shape_ = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        # Get coords
        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


class Cross(Square):
    def init_shape(self):
        """
        corners are:
        a = (self.x, self.y)
        b = (self.x + self.size, self.y)
        c = (self.x, self.y + self.size)
        d = (self.x + self.size, self.y + self.size)

        width of cross is, let's imagine, 3. (must be less than self.size)
        then distance from a to corner is

        j = (a[0], a[1] + 3)
        k = (a[0] + 3, a[1] + 3)
        1 = (a[0] + 3, a[1])

        2 = (b[0] - 3, b[1])
        3 = (b[0] - 3, b[1] + 3)
        4 = (b[0], b[1] + 3)

        5 = (d[0], d[1] - 3)
        6 = (d[0] - 3, d[1] - 3)
        7 = (d[0] - 3, d[1])

        8 = (c[0] + 3, c[1])
        9 = (c[0] + 3, c[1] - 3)
        i = (c[0], c[1] - 3)

        a 1 . 2 b
        j k . 3 4
        . . . . .
        i 9 . 6 5
        c 8 . 7 d
        """
        self.size = rand_size_3()
        self.cross_width = 3
        assert self.cross_width < self.size, f"cross width too wide: {self.cross_width} < {self.size}"

        a = (self.x, self.y)
        b = (self.x + self.size, self.y)
        c = (self.x, self.y + self.size)
        d = (self.x + self.size, self.y + self.size)

        coords = [
            (a[0], a[1] + self.cross_width),
            (a[0] + self.cross_width, a[1] + self.cross_width),
            (a[0] + self.cross_width, a[1]),

            (b[0] - self.cross_width, b[1]),
            (b[0] - self.cross_width, b[1] + self.cross_width),
            (b[0], b[1] + self.cross_width),

            (d[0], d[1] - self.cross_width),
            (d[0] - self.cross_width, d[1] - self.cross_width),
            (d[0] - self.cross_width, d[1]),

            (c[0] + self.cross_width, c[1]),
            (c[0] + self.cross_width, c[1] - self.cross_width),
            (c[0], c[1] - self.cross_width),
        ]

        shape_ = Polygon(coords)

        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


class Triangle(Shape):
    """
    Equilateral triangles
    """

    def init_shape(self):
        self.size = rand_size_2()
        # X and Y are the center of the shape and size is the length of one
        # side. Assume one point lies directly above the center. Then:
        # https://math.stackexchange.com/questions/1344690/is-it-possible-to-find-the-vertices-of-an-equilateral-triangle-given-its-center
        shape_ = Polygon(
            [
                (self.x, self.y + np.sqrt(3) * self.size / 3),
                (self.x - self.size / 2, self.y - np.sqrt(3) * self.size / 6),
                (self.x + self.size / 2, self.y - np.sqrt(3) * self.size / 6),
            ]
        )
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


def _equilateral_polygon_vertices(x, y, n_vertices, size):
    vertices = []
    for i in range(n_vertices):
        v = (
            x + size * (math.cos(2 * math.pi * i / n_vertices)),
            y + size * (math.sin(2 * math.pi * i / n_vertices))
        )
        vertices.append(v)
    return vertices


class Pentagon(Shape):
    def init_shape(self):
        self.size = rand_size_2()
        shape_ = Polygon(
            _equilateral_polygon_vertices(self.x, self.y, 5, self.size)
        )
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


class Hexagon(Shape):
    def init_shape(self):
        self.size = rand_size_2()
        shape_ = Polygon(
            _equilateral_polygon_vertices(self.x, self.y, 6, self.size)
        )
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = (
            np.round(np.array(self.shape.exterior.coords)[:-1].flatten())
            .astype(np.int)
            .tolist()
        )


def random(shapes=None):
    if shapes is None:
        return np.random.choice(SHAPES)
    else:
        return np.random.choice(shapes)


def new(existing_shape, shapes=None):
    if shapes is None:
        shapes = SHAPES
    if len(shapes) == 1 and shapes[0] == existing_shape:
        raise RuntimeError("No additional shapes to generate")
    new_s = existing_shape
    while new_s == existing_shape:
        new_s = np.random.choice(shapes)
    return new_s


SHAPE_IMPLS = {
    "circle": Circle,
    "ellipse": Ellipse,
    "square": Square,
    "rectangle": Rectangle,
    "triangle": Triangle,
    "semicircle": Semicircle,
    "pentagon": Pentagon,
    "cross": Cross,
    "hexagon": Hexagon,
}


SHAPES = ["circle", "square", "rectangle", "ellipse", "triangle", "semicircle", "pentagon", "cross", "hexagon"]

if "SHAPEWORLD_N_SHAPES" in os.environ:
    N_SHAPES = int(os.environ['SHAPEWORLD_N_SHAPES'])
    assert N_SHAPES < len(SHAPES), f"{N_SHAPES} requested, have {len(SHAPES)}"
else:
    N_SHAPES = 5  # original setting
SHAPES = SHAPES[:N_SHAPES]


SHAPE_VARS = exprvars('s', len(SHAPES))
S2V = dict(zip(SHAPES, SHAPE_VARS))
V2S = dict(zip(SHAPE_VARS, SHAPES))
ONEHOT_VAR = expr.OneHot(*SHAPE_VARS)
