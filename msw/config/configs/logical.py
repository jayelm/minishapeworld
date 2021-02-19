from collections import namedtuple

from . import configbase

_LogicalConfigBase = namedtuple("LogicalConfig", ["formula"])


class LogicalConfig(configbase._ConfigBase, _LogicalConfigBase):
    pass
