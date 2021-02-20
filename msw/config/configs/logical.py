from collections import namedtuple

from pyeda.boolalg import expr
import numpy as np

from ... import color
from ... import constants as C
from ... import shape
from .. import spec
from .. import util as config_util
from . import configbase


_LogicalConfigBase = namedtuple("LogicalConfig", ["formula"])


def expr_is_constant(f):
    return f == expr.Zero or f == expr.One


class LogicalConfig(configbase._ConfigBase, _LogicalConfigBase):
    # Formulas are boolean expressions over one hot binary variables
    # that indicate:
    # s[1..7] is shapes, c[1..7] is colors (get the pyeda versions of both then
    # do exhaustive enumeration)
    def __init__(self, *args, **kwargs):
        self.formula_dnf = self.formula.to_dnf()

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
            if np.random.random() < 0.5:
                op = expr.And
            else:
                op = expr.Or
            formula = op(formula, exprs.pop())
        return cls(formula)

    #  def __eq__(self, other):

    @classmethod
    def enumerate(cls, max_formula_len=2):
        configs = set()

        # NOTE - there may be an issue here if we for example reach Or(x,
        # And(x, y)) which isn't simplified to x according to pyeda. But we
        # always start from the bottom up.

        def search(f, f_len, max_formula_len):
            # If the formula is not satisfiable or vacuously true, terminate
            if f is not None and expr_is_constant(f):
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
                            # TODO - if we want to preserve the conjunctive nature, maybe don't simplify here?
                            for op in [expr.And, expr.Or]:
                                new_f = op(f, maybe_neg_x)
                                search(new_f, f_len + 1, max_formula_len)

        for mfl in range(1, max_formula_len + 1):
            search(None, 0, mfl)

        return configs
