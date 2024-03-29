from collections import namedtuple, defaultdict
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


def contains_shape(config):
    config_str = config.formula_to_str()
    config_toks = config_str.split(" ")
    return any(ctok in shape.SHAPES for ctok in config_toks)


def oversample(configs, strategy="singleton_half"):
    """
    Tested options in this function:
    (1) sample singleton shapes up to half of all concepts
    (2) sample singleton shapes up to 1/4 of all concepts
    (3) sample anything containing shape to be twice as many as anything not
        containing shape (note that we even have more colors!!)
    """
    if strategy == "singleton_half":
        return _oversample(configs, special_criterion=is_singleton_shape, special_ratio=1)
    elif strategy == "singleton_quarter":
        return _oversample(configs, special_criterion=is_singleton_shape, special_ratio=0.5)
    elif strategy == "any_2x":
        return _oversample(configs, special_criterion=contains_shape, special_ratio=2)
    elif strategy == "even":
        return _oversample_even(configs)
    else:
        raise ValueError(f"Unknown strategy {strategy}")


def _oversample_even(configs):
    config_types = defaultdict(list)
    for config in configs:
        config_str = config.formula_to_str()
        config_lf = concept_to_lf(config_str)
        config_type = lf_to_config_type(config_lf)
        config_types[config_type].append(config)
    # Get maximum length, multiply by 10 to reduce potential issues with finite
    # oversampling
    max_n_cfgs = max(len(cfgs) for cfgs in config_types.values()) * 10

    even_config_types = defaultdict(list)
    for config_type, configs in config_types.items():
        while len(even_config_types[config_type]) < max_n_cfgs:
            i = np.random.choice(len(configs))
            even_config_types[config_type].append(configs[i])

    flattened_even_configs = []
    for configs in even_config_types.values():
        flattened_even_configs.extend(configs)

    return flattened_even_configs


def _concept_to_lf(concept):
    """
    Parse concept. Since fixed recursion we don't have to worry about
    precedence
    """
    op = ""
    if "or" in concept:
        op = "or"
    elif "and" in concept:
        op = "and"

    if op:
        op_index = concept.index(op)
        left = concept[:op_index]
        right = concept[op_index + 1:]
        return (
            op,
            _concept_to_lf(left),
            _concept_to_lf(right)
        )

    if "not" in concept:
        assert len(concept) == 2, f"unable to parse {concept}"
        return ("not", (concept[1], ))

    assert len(concept) == 1, f"unable to parse {concept}"
    return (concept[0], )


def concept_to_lf(concept, split=True):
    if split:
        concept = concept.split(" ")
    return _concept_to_lf(concept)


def lf_to_config_type(concept):
    # XXX: what's the right way to categorize config type? As just "ors",
    # "ands", and "singletons"? Maybe the right way is how I do metadata i.e.
    # ignore shape/color?
    if len(concept) == 1:  # f
        return (feature_type(concept[0]), )
    elif len(concept) == 2:  # not(f)
        assert concept[0] == "not"
        #  return ("not", lf_to_config_type(concept[1]))
        # Ignore nots?
        return lf_to_config_type(concept[1])
    elif len(concept) == 3:
        return (concept[0], lf_to_config_type(concept[1]), lf_to_config_type(concept[2]))


def feature_type(feat):
    if feat in shape.SHAPES:
        return "shape"
    elif feat in color.COLORS:
        return "color"
    else:
        raise ValueError(f"Unknown feature {feat}")


def is_singleton_shape(config):
    if not config.disjunction and not config.conjunction:
        # Unary
        config_str = config.formula_to_str()
        if config_str.startswith("not "):
            config_str = config_str[4:]
        return config_str in shape.SHAPES
    return False


def _oversample(configs, special_criterion=is_singleton_shape, special_ratio=1):
    special_configs = []
    other_configs = []
    for config in configs:
        if special_criterion(config):
            special_configs.append(config)
        else:
            other_configs.append(config)

    special_configs_rep = []
    target_len = int(special_ratio * len(other_configs))
    while len(special_configs_rep) < target_len:
        i = np.random.choice(len(special_configs))
        special_configs_rep.append(special_configs[i])

    return special_configs_rep + other_configs


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
    return expr.And(spc_expr, f).satisfy_one() is not None


class LogicalConfig(configbase._ConfigBase, _LogicalConfigBase):
    # Formulas are boolean expressions over one hot binary variables
    # that indicate:
    # s[1..7] is shapes, c[1..7] is colors (get the pyeda versions of both then
    # do exhaustive enumeration)
    def __init__(self, *args, **kwargs):
        self.formula_dnf = self.formula.to_dnf()
        self.formula_onehot = onehot_f(self.formula)

        self.pos_assignments = []
        self.neg_assignments = []
        for spc in spec.ShapeSpec.enumerate_both():
            if satisfies(spc, self.formula_onehot):
                self.pos_assignments.append(spc)
            else:
                self.neg_assignments.append(spc)

        self.disjunction = isinstance(self.formula, expr.OrOp)
        self.conjunction = isinstance(self.formula, expr.AndOp)
        # Get assignments by which part they satisfy
        self.left_assignments = []
        self.right_assignments = []
        self.both_assignments = []
        self.neither_assignments = []
        if self.conjunction or self.disjunction:
            left_formula, right_formula = self.formula.xs

            only_left = onehot_f(expr.And(left_formula, expr.Not(right_formula)))
            only_right = onehot_f(expr.And(right_formula, expr.Not(left_formula)))

            for spc in spec.ShapeSpec.enumerate_both():
                if satisfies(spc, only_left):
                    self.left_assignments.append(spc)
                elif satisfies(spc, only_right):
                    self.right_assignments.append(spc)
                elif satisfies(spc, self.formula_onehot):  # Both
                    self.both_assignments.append(spc)
                else:
                    self.neither_assignments.append(spc)

            if self.disjunction:
                assert set(self.neither_assignments) == set(self.neg_assignments)
                assert set(self.left_assignments + self.right_assignments + self.both_assignments) == set(self.pos_assignments)
            else:
                assert set(self.both_assignments) == set(self.pos_assignments)
                assert set(self.left_assignments + self.right_assignments + self.neither_assignments) == set(self.neg_assignments)

    def __hash__(self):
        return hash(str(self.formula_dnf))

    def __eq__(self, other):
        return str(self.formula_dnf) == str(other.formula_dnf)

    @classmethod
    def random(cls, max_formula_len=2, ops={"and", "or", "not"}):
        flen = np.random.randint(max_formula_len) + 1
        exprs = []
        for _ in range(flen):
            x = spec.ShapeSpec.random().to_expr()
            # Potentially negate
            if "not" in ops:
                if np.random.random() < 0.5:
                    x = expr.Not(x)
            exprs.append(x)
        # Combine formulas
        formula = exprs.pop()
        while exprs:
            # FIRST SORT OUT ORSvsANDS
            # ALSO USE OPS
            raise NotImplementedError
            if np.random.random() < 0.5:
                op = expr.And
            else:
                op = expr.Or
            formula = op(formula, exprs.pop(), simplify=False)
        if formula.is_one() or not expr_is_satisfiable(formula):
            # Try again
            return cls.random(max_formula_len=max_formula_len)
        return cls(formula)

    @classmethod
    def enumerate(cls, min_formula_len=1, max_formula_len=2, ops={"and", "or", "not"}):
        """
        Enumerate through possible logical captions.
        They are either length 1 or 2 conjunctions/disjunctions of shapes or
        colors, filtering out ones that are vacuous givne that shapes and
        colors are onehot (e.g. "not green or not blue" is vacuous)
        """
        configs = set()

        assert min_formula_len in range(1, 3), "Invalid min formula length"
        assert max_formula_len in range(1, 3), "Invalid max formula length"
        assert max_formula_len >= min_formula_len, "max must be >= min"

        # Length 1
        if min_formula_len <= 1:
            for spc in spec.ShapeSpec.enumerate_color():
                configs.add(cls(spc.to_expr()))
                if "not" in ops:
                    configs.add(cls(expr.Not(spc.to_expr())))

            for spc in spec.ShapeSpec.enumerate_shape():
                configs.add(cls(spc.to_expr()))
                if "not" in ops:
                    configs.add(cls(expr.Not(spc.to_expr())))

        # Length 2 - AND
        if max_formula_len >= 2:
            primitives = [
                p
                for p in color.COLOR_VARS + shape.SHAPE_VARS
            ]
            if "not" in ops:
                primitives += [expr.Not(p) for p in primitives]
            primitives_with_onehots = [
                (p, onehot_f(p))
                for p in primitives
            ]
            combos = itertools.combinations(primitives_with_onehots, 2)
            for (x1, x1_onehot), (x2, x2_onehot) in combos:
                allowed_ops = []
                if "or" in ops:
                    allowed_ops.append(expr.Or)
                if "and" in ops:
                    allowed_ops.append(expr.And)

                for op in allowed_ops:
                    f = op(x1, x2)
                    f_onehot = onehot_f(f)
                    if (
                        expr_is_valid(f) and
                        not (f_onehot.equivalent(x1_onehot)) and
                        not (f_onehot.equivalent(x2_onehot))
                    ):
                        # Expression should be different from the basic conjunctions.
                        configs.add(cls(f))

        return list(configs)

    def instantiate(self, label, shape_kwargs=None, **kwargs):
        """
        for OR, POSITIVES are sampled equally from
        left_assn, right_assn, both_assn
        For AND, NEGATIVES are sampled equally from
        left_assn, right_assn, neither_assn
        """
        if label:
            if self.disjunction:
                options = [
                    self.left_assignments, self.right_assignments,
                    self.both_assignments
                ]
                # Sometimes can be empty (e.g. square or rectangle - no
                # satisfiers of both]
                options = [o for o in options if o]
                assns = options[np.random.choice(len(options))]
            else:
                assns = self.pos_assignments
        else:
            if self.conjunction:
                options = [
                    self.left_assignments, self.right_assignments,
                    self.neither_assignments,
                ]
                options = [o for o in options if o]
                assns = options[np.random.choice(len(options))]
            else:
                assns = self.neg_assignments

        spc = assns[np.random.choice(len(assns))]
        s = self.add_shape(spc, shape_kwargs=shape_kwargs)
        return self, [s]

    def __str__(self):
        f_str = self.formula_to_str()
        return f"LogicalConfig(formula={f_str})"

    def formula_to_str(self):
        f_ast = self.formula.to_ast()
        f_str = LogicalConfig._ast_to_str(f_ast)
        return f_str

    @staticmethod
    def _ast_to_str(ast):
        if len(ast) > 2:
            assert len(ast) == 3, "Op with > 2 args"
            op_name = ast[0]
            l_str = LogicalConfig._ast_to_str(ast[1])
            r_str = LogicalConfig._ast_to_str(ast[2])
            return f"{l_str} {op_name} {r_str}"
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
            return f"{op_name} {arg_str}"

    def json(self):
        if self.disjunction:
            op = "disjunction"
        elif self.conjunction:
            op = "conjunction"
        else:
            op = "literal"

        return {
            "type": "logical",
            "pos": self.pos_assignments,
            "neg": self.neg_assignments,
            "op": op,
            "concept": self.formula_to_str(),
        }

    def format(self, lang_type="standard"):
        return self.formula_to_str()
