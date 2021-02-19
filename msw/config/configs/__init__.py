from .single import SingleConfig
from .spatial import SpatialConfig
from .logical import LogicalConfig


CONFIGS = {
    'single': SingleConfig,
    'spatial': SpatialConfig,
    'logical': LogicalConfig,
}
