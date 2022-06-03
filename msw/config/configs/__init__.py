from .logical import LogicalConfig
from .single import SingleConfig
from .spatial import SpatialConfig

CONFIGS = {
    "single": SingleConfig,
    "spatial": SpatialConfig,
    "logical": LogicalConfig,
}
