import config
import shape
import lang


def test_matching():
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 0)
    s1 = shape.SHAPE_IMPLS['square'](color_='red', x=30, y=30)
    s2 = shape.SHAPE_IMPLS['square'](color_='blue', x=40, y=30)
    assert config.matches(cfg.shapes[0], s1)
    assert config.matches(cfg.shapes[1], s2)
    assert not config.matches(cfg.shapes[1], s1)
    assert not config.matches(cfg.shapes[0], s2)

    assert config.matches((None, None), s1)
    assert config.matches((None, None), s2)

    assert config.matches(('red', None), s1)
    assert not config.matches(('red', None), s2)

    assert config.matches((None, 'square'), s1)
    assert config.matches((None, 'square'), s2)


def test_relation_lang():
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 0)
    assert lang.fmt_config(cfg) == 'red square left blue square'

    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 1)
    assert lang.fmt_config(cfg) == 'red square right blue square'

    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 1, 0)
    assert lang.fmt_config(cfg) == 'red square below blue square'

    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 1, 1)
    assert lang.fmt_config(cfg) == 'red square above blue square'


def test_relation():
    s1 = shape.SHAPE_IMPLS['square'](color_='red', x=30, y=30)
    # S2 is below - higher coordinates
    s2 = shape.SHAPE_IMPLS['square'](color_='blue', x=40, y=40)

    assert s1.left(s2)
    assert not s1.left(s1)
    assert not s1.right(s2)
    assert not s2.right(s2)

    assert s1.above(s2)
    assert s2.below(s1)
    assert not s1.below(s2)
    assert not s2.above(s1)

    # left
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 0)
    assert config.has_relation(s1, s2, cfg.relation, cfg.dir)
    assert not config.has_relation(s2, s1, cfg.relation, cfg.dir)

    # right
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 1)
    assert config.has_relation(s2, s1, cfg.relation, cfg.dir)
    assert not config.has_relation(s1, s2, cfg.relation, cfg.dir)

    # below
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 1, 0)
    assert config.has_relation(s2, s1, cfg.relation, cfg.dir)
    assert not config.has_relation(s1, s2, cfg.relation, cfg.dir)

    # above
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 1, 1)
    assert config.has_relation(s1, s2, cfg.relation, cfg.dir)
    assert not config.has_relation(s2, s1, cfg.relation, cfg.dir)


def test_matches_shapes():
    cfg = config.SpatialConfig((('red', None), ('blue', 'square')), 0, 0)

    s1 = shape.SHAPE_IMPLS['square'](color_='red')
    assert cfg.matches_shapes(s1) == [0]

    s2 = shape.SHAPE_IMPLS['square'](color_='blue')
    assert cfg.matches_shapes(s2) == [1]

    cfg = config.SpatialConfig((('red', None), (None, 'square')), 0, 0)

    s1 = shape.SHAPE_IMPLS['square'](color_='red')
    assert cfg.matches_shapes(s1) == [0, 1]

    s2 = shape.SHAPE_IMPLS['square'](color_='blue')
    assert cfg.matches_shapes(s2) == [1]

    s2 = shape.SHAPE_IMPLS['triangle'](color_='blue')
    assert cfg.matches_shapes(s2) == []


def test_does_not_validate_1():
    cfg = config.SpatialConfig((('red', 'square'), ('blue', 'square')), 0, 0)
    s1 = shape.SHAPE_IMPLS['square'](color_='red', x=30, y=30)
    s2 = shape.SHAPE_IMPLS['square'](color_='blue', x=40, y=30)

    s3 = shape.SHAPE_IMPLS['triangle'](color_='blue')

    assert not cfg.does_not_validate([s1], s2)
    assert not cfg.does_not_validate([s2], s1)

    assert not cfg.does_not_validate([s1, s2], s1)
    assert not cfg.does_not_validate([s1, s2], s2)

    assert cfg.does_not_validate([s1, s2], s3)
    assert not cfg.does_not_validate([s1, s3], s2)
    assert not cfg.does_not_validate([s2, s3, s1], s1)


def test_does_not_validate_2():
    cfg = config.SpatialConfig((('red', None), ('blue', None)), 0, 0)
    s1 = shape.SHAPE_IMPLS['square'](color_='red')
    s1.x = 30
    s1.y = 30
    s2 = shape.SHAPE_IMPLS['square'](color_='blue')
    s2.x = 40
    s2.y = 30

    assert not cfg.does_not_validate([s1], s2)

    assert not cfg.does_not_validate([s1], s2)
    assert not cfg.does_not_validate([s2], s1)

    s3 = shape.SHAPE_IMPLS['triangle'](color_='red', x=10, y=10)
    assert not cfg.does_not_validate([s1, s2], s3)

    s4 = shape.SHAPE_IMPLS['triangle'](color_='red', x=39, y=10)
    assert not cfg.does_not_validate([s1, s2], s4)

    s5 = shape.SHAPE_IMPLS['triangle'](color_='red', x=41, y=10)
    assert cfg.does_not_validate([s1, s2], s5)


def test_does_not_validate_3():
    # i.e. above
    cfg = config.SpatialConfig((('green', None), (None, 'triangle')), 1, 1)
    s1 = shape.SHAPE_IMPLS['ellipse'](color_='green', x=0, y=20)
    s2 = shape.SHAPE_IMPLS['triangle'](color_='white', x=0, y=30)

    assert not cfg.does_not_validate([s2], s1)
    assert not cfg.does_not_validate([s1], s2)

    assert not cfg.does_not_validate([s1, s2], s1)
    assert not cfg.does_not_validate([s1, s2], s2)

    s3 = shape.SHAPE_IMPLS['triangle'](color_='red', x=0, y=31)
    assert not cfg.does_not_validate([s1, s2], s3)
    s4 = shape.SHAPE_IMPLS['triangle'](color_='red', x=0, y=18)
    assert cfg.does_not_validate([s1, s2], s4)

    s5 = shape.SHAPE_IMPLS['square'](color_='green', x=0, y=25)
    assert not cfg.does_not_validate([s1, s2], s5)
    s6 = shape.SHAPE_IMPLS['circle'](color_='green', x=0, y=35)
    assert cfg.does_not_validate([s1, s2], s6)
