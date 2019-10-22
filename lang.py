"""
Converting configs to language.
"""

import config

def fmt_config(cfg):
    if isinstance(cfg, config.SingleConfig):
        return _fmt_config_single(cfg)
    elif isinstance(cfg, config.SpatialConfig):
        return _fmt_config_spatial(cfg)
    else:
        raise NotImplementedError(type(cfg))


def _fmt_config_single(cfg):
    color_, shape_ = cfg
    shape_txt = 'shape'
    color_txt = ''
    if shape_ is not None:
        shape_txt = shape_
    if color_ is not None:
        color_txt = color_ + ' '
    return '{}{}'.format(color_txt, shape_txt)


def _fmt_config_spatial(cfg):
    (s1, s2), relation, relation_dir = cfg
    if relation == 0:
        if relation_dir == 0:
            rel_txt = 'left'
        else:
            rel_txt = 'right'
    else:
        if relation_dir == 0:
            rel_txt = 'below'
        else:
            rel_txt = 'above'
    if s1[0] is None:
        s1_0_txt = ''
    else:
        s1_0_txt = s1[0]
    if s1[1] is None:
        s1_1_txt = 'shape'
    else:
        s1_1_txt = s1[1]
    if s2[0] is None:
        s2_0_txt = ''
    else:
        s2_0_txt = s2[0]
    if s2[1] is None:
        s2_1_txt = 'shape'
    else:
        s2_1_txt = s2[1]
    return '{} {} {} {} {}'.format(s1_0_txt, s1_1_txt, rel_txt, s2_0_txt,
                                   s2_1_txt)
