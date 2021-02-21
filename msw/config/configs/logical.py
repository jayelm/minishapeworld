from collections import namedtuple

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


def expr_is_satisfiable(f):
    return onehot_f(f).satisfy_one() is not None


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
        self.pos_assignments = [spc for spc in spec.ShapeSpec.enumerate_both() if satisfies(spc, self.formula)]

        self.neg_formula = expr.Not(self.formula)
        self.neg_assignments = [spc for spc in spec.ShapeSpec.enumerate_both() if satisfies(spc, self.neg_formula)]

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
    def enumerate(cls, max_formula_len=2):
        configs = set()

        # NOTE - there may be an issue here if we for example reach Or(x,
        # And(x, y)) which isn't simplified to x according to pyeda. But we
        # always start from the bottom up.

        def search(f, f_len, max_formula_len):
            # If the formula is not satisfiable or vacuously true, terminate
            if f is not None and (f.is_one() or not expr_is_satisfiable(f)):
                return

            # Max formula length - terminate
            if f_len == max_formula_len:
                c = cls(f)
                if c not in configs:
                    configs.add(c)
            else:
                # Search over all possible variables and ops
                for spc in spec.ShapeSpec.enumerate():
                    x = spc.to_expr()
                    for maybe_neg_x in [x, expr.Not(x)]:
                        if f is None:
                            assert f_len == 0
                            search(maybe_neg_x, f_len + 1, max_formula_len)
                        else:
                            for op in [expr.And, expr.Or]:
                                new_f = op(f, maybe_neg_x, simplify=False)
                                # Make sure this new formula defines a new formula
                                if not onehot_f(new_f).equivalent(onehot_f(f)):
                                    search(new_f, f_len + 1, max_formula_len)

        for mfl in range(1, max_formula_len + 1):
            search(None, 0, mfl)

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
