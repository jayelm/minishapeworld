
from . import configbase
from collections import namedtuple


_LogicalConfigBase = namedtuple('LogicalConfig', ['formula'])


class LogicalConfig(configbase._ConfigBase, _LogicalConfigBase):
    pass
