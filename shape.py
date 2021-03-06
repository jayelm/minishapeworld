"""
Shape implementations
"""

import numpy as np
from numpy import random

import color
import constants as c

from shapely import affinity
from shapely.geometry import Point, box, Polygon


def rand_size():
    return random.randint(c.SIZE_MIN, c.SIZE_MAX)


def rand_size_2():
    """Slightly bigger."""
    return random.randint(c.SIZE_MIN + 2, c.SIZE_MAX + 2)


def rand_pos():
    return random.randint(c.X_MIN, c.X_MAX)


class Shape:
    def __init__(self,
                 x=None,
                 y=None,
                 relation=None,
                 relation_dir=None,
                 color_=None,
                 max_rotation=90):
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
                    self.x = random.randint(c.X_MIN_34, c.X_MAX)
                else:
                    # Place left 3/4
                    self.x = random.randint(c.X_MIN, c.X_MAX_34)
            else:
                # y matters - x is totally random
                self.x = rand_pos()
                if relation_dir == 0:
                    # Place top 3/4 of screen, so second shape can be placed
                    # BELOW
                    # NOTE: Remember coords for y travel in opp dir
                    self.y = random.randint(c.X_MIN, c.X_MAX_34)
                else:
                    self.y = random.randint(c.X_MIN_34, c.X_MAX)
        self.rotation = random.randint(max_rotation)
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, color.PENS[self.color])

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
            'shape': self.name,
            'color': self.color,
            'pos': {
                'x': self.x,
                'y': self.y
            },
            'rotation': self.rotation
        }


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= c.SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > c.SIZE_MIN, ("{} {}".format(smaller, self.dx))
            self.dy = random.randint(c.SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, c.SIZE_MAX)
        if random.random() < 0.5:
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
        image.draw.ellipse(self.coords, color.BRUSHES[self.color])


class Rectangle(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size_2()
        bigger = int(self.dx * min_skew)
        if bigger >= c.SIZE_MAX:
            smaller = int(self.dx / min_skew)
            self.dy = random.randint(c.SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, c.SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape_ = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, color.BRUSHES[self.color],
                           color.PENS[self.color])


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size_2()
        shape_ = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()


class Triangle(Shape):
    """
    Equilateral triangles
    """

    def init_shape(self):
        self.size = rand_size_2()
        # X and Y are the center of the shape and size is the length of one
        # side. Assume one point lies directly above the center. Then:
        # https://math.stackexchange.com/questions/1344690/is-it-possible-to-find-the-vertices-of-an-equilateral-triangle-given-its-center
        shape_ = Polygon([
            (self.x, self.y + np.sqrt(3) * self.size / 3),
            (self.x - self.size / 2, self.y - np.sqrt(3) * self.size / 6),
            (self.x + self.size / 2, self.y - np.sqrt(3) * self.size / 6),
        ])
        # Rotation
        shape_ = affinity.rotate(shape_, self.rotation)
        self.shape = shape_

        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, color.BRUSHES[self.color],
                           color.PENS[self.color])


SHAPES = ['circle', 'square', 'rectangle', 'ellipse', 'triangle']
SHAPE_IMPLS = {
    'circle': Circle,
    'ellipse': Ellipse,
    'square': Square,
    'rectangle': Rectangle,
    'triangle': Triangle
    # TODO: semicircle
}
