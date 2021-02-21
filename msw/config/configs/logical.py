from collections import namedtuple
import itertools

from pyeda.boolalg import expr
from pyeda.boolalg.expr import _LITS
import numpy as np

from ... import color
from ... import constants as C
from ... import shape
from .. import spec
from .. import util as config_util
from . import configbase


_LogicalConfigBase = namedtuple("LogicalConfig", ["formula"])


def onehot_f(f):
    return expr.And(f, color.ONEHOT_VAR, shape.ONEHOT_VAR)


def expr_is_valid(f):
    f_onehot = onehot_f(f)
    return not (
        f.is_one() or
        f.is_zero() or
        not expr_is_satisfiable(f_onehot) or
        expr_is_vacuous(f_onehot)
    )


def expr_is_satisfiable(f):
    return f.satisfy_one() is not None


def expr_is_vacuous(f):
    """
    Vacuous if every shape applies to this.
    """
    return all(
        expr_is_satisfiable(expr.And(f, spc.to_expr()))
        for spc in spec.ShapeSpec.enumerate_both()
    )


def satisfies(spc, f):
    """
    Return True if the spec (which must be BOTH) satisfies formula f.
    """
    assert spc.spec_type == spec.ShapeSpecTypes.BOTH, "must pass in both type"
    spc_expr = spc.to_expr()
    # FIXME - do we need onehot here?
    return expr.And(spc_expr, onehot_f(f)).satisfy_one() is not None


class LogicalConfig(configbase._ConfigBase, _LogicalConfigBase):
    # Formulas are boolean expressions over one hot binary variables
    # that indicate:
    # s[1..7] is shapes, c[1..7] is colors (get the pyeda versions of both then
    # do exhaustive enumeration)
    def __init__(self, *args, **kwargs):
        self.formula_dnf = self.formula.to_dnf()
        self.formula_onehot = onehot_f(self.formula)
        self.pos_assignments = [spc for spc in spec.ShapeSpec.enumerate_both() if satisfies(spc, self.formula_onehot)]

        self.neg_formula = expr.Not(self.formula)
        self.neg_formula_onehot = onehot_f(self.formula)
        self.neg_assignments = [spc for spc in spec.ShapeSpec.enumerate_both() if satisfies(spc, self.neg_formula_onehot)]

    def __hash__(self):
        return hash(str(self.formula_dnf))

    def __eq__(self, other):
        return str(self.formula_dnf) == str(other.formula_dnf)

    @classmethod
    def random(cls, max_formula_len=2):
        flen = np.random.randint(max_formula_len) + 1
        exprs = []
        for _ in range(flen):
            x = spec.ShapeSpec.random().to_expr()
            # Potentially negate
            if np.random.random() < 0.5:
                x = expr.Not(x)
            exprs.append(x)
        # Combine formulas
        formula = exprs.pop()
        while exprs:
            # FIRST SORT OUT ORSvsANDS
            raise NotImplementedError
            if np.random.random() < 0.5:
                op = expr.And
            else:
                op = expr.Or
            formula = op(formula, exprs.pop(), simplify=False)
        if formula.is_one() or not expr_is_satisfiable(f):
            # Try again
            return cls.random(max_formula_len=max_formula_len)
        return cls(formula)

    @classmethod
    def enumerate(cls):
        """
        Enumerate through possible tags. Because we have a small length (max of
        2) we can specifically list out the formulas we want:
        # LENGTH 1
        1. A color/not a color
        2. A shape/not a shape

        # LENGTH 2 (i.e. a single shapecolor)
        # I think this is just ORs across 
        3. A shapecolor/not a shapecolor (i.e. shape and color)
        6. A shape or a color/not a (shape or a color)
        7. A shape or not a color
        8. A shape and not a color
        9. A color or not a shape
        10. A shape and not a color
        # This is just shapes and colors possibly NOTed. that's it?
        # since (not a shape and not a color) -> not (shape or color)
        # and (not a color and not a shape) -> 
        # ORs across colors only work...not ANDs. (Can check satisfiable via
        # onehot.)

        # Maybe leave out the more complicated ones for now? This feels hard
        # enough already.
        """
        configs = set()

        # NOTE - there may be an issue here if we for example reach Or(x,
        # And(x, y)) which isn't simplified to x according to pyeda. But we
        # always start from the bottom up.

        # Length 1
        for spc in spec.ShapeSpec.enumerate_color():
            configs.add(cls(spc.to_expr()))
            configs.add(cls(expr.Not(spc.to_expr())))

        for spc in spec.ShapeSpec.enumerate_shape():
            configs.add(cls(spc.to_expr()))
            configs.add(cls(expr.Not(spc.to_expr())))

        # Length 2 - AND
        primitives = [
            p
            for p in color.COLOR_VARS + shape.SHAPE_VARS
        ]
        primitives += [expr.Not(p) for p in primitives]
        primitives_with_onehots = [
            (p, onehot_f(p))
            for p in primitives
        ]
        combos = itertools.combinations(primitives_with_onehots, 2)
        for (x1, x1_onehot), (x2, x2_onehot) in combos:
            for op in [expr.And, expr.Or]:
                f = op(x1, x2)
                f_onehot = onehot_f(f)
                if (
                    expr_is_valid(f) and
                    not (f_onehot.equivalent(x1_onehot)) and
                    not (f_onehot.equivalent(x2_onehot))
                ):
                    configs.add(cls(f))

        return list(configs)

    def instantiate(self, label, **kwargs):
        """
        TODO - do we actually invalidate here (hard negatives? or just find a negative satisfier)
        """
        pass

    def __str__(self):
        f_ast = self.formula.to_ast()
        f_str = LogicalConfig._ast_to_str(f_ast)
        return f"LogicalConfig(formula={f_str})"

    @staticmethod
    def _ast_to_str(ast):
        if len(ast) > 2:
            assert len(ast) == 3, "Op with > 2 args"
            op_name = ast[0]
            l_str = LogicalConfig._ast_to_str(ast[1])
            r_str = LogicalConfig._ast_to_str(ast[2])
            return f"( {l_str} {op_name} {r_str} )"
        elif ast[0] == 'lit':
            assert len(ast) == 2
            if ast[1] < 0:  # Negative
                maybe_not = "not "
                lit = _LITS[-ast[1]]
            else:
                maybe_not = ""
                lit = _LITS[ast[1]]
            if lit.name == 'c':
                lit_name = color.V2C[lit]
            else:
                lit_name = shape.V2S[lit]
            return f"{maybe_not}{lit_name}"
        else:
            assert len(ast) == 2
            op_name = ast[0]
            arg_str = LogicalConfig._ast_to_str(ast[1])
            return f"( {op_name} {arg_str} )"
