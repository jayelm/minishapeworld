"""
Config implementations
"""

from enum import Enum
from collections import namedtuple


SpatialConfig = namedtuple('SpatialConfig', ['shapes', 'relation', 'dir'])
SingleConfig = namedtuple('SingleConfig', ['shape', 'color'])


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
