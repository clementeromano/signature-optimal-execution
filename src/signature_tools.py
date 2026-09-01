"""Algebra and numerical utilities for signature-based execution controls.

The central object in this module is :class:`SigPoly`. It represents a finite
linear functional on a path signature. Multiplication uses the shuffle
product, while :meth:`SigPoly.tensor` concatenates words. Together these
operations express the execution objectives used in the project notebook.

Words follow the ordering returned by ``esig.sigkeys``: the empty word first,
then words ordered by level and lexicographically within each level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import esig
import iisignature
import numpy as np
import sympy
from scipy.optimize import minimize


Word = Tuple[int, ...]


class SigPoly:
    """A truncated linear functional on a path signature.

    Parameters
    ----------
    dimension:
        Dimension of the underlying path.
    truncate:
        Highest signature level retained during algebraic operations.
    coefficients:
        Coefficients in signature-key order, including the empty word.
    zero, is_zero:
        Optional zero value and predicate. These allow the coefficients to be
        numeric values or symbolic expressions.
    """

    def __init__(
        self,
        dimension: int,
        truncate: int,
        coefficients: list,
        *,
        zero: Any = 0,
        is_zero: Callable[[Any], bool] | None = None,
    ) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(truncate, int) or truncate <= 0:
            raise ValueError("truncate must be a positive integer")
        if not isinstance(coefficients, list):
            raise TypeError("coefficients must be a list")

        self.dimension = dimension
        self.level = len2level(dimension, len(coefficients))
        if truncate < self.level:
            raise ValueError("truncate cannot be lower than the coefficient level")

        self.coefficients = coefficients
        self.truncate = truncate
        self.zero = zero
        self.is_zero = is_zero if is_zero is not None else lambda value: value == zero

    def set_level(self, truncate: int) -> None:
        """Set the truncation level, preserving the historical method name."""
        if not isinstance(truncate, int) or truncate < self.level:
            raise ValueError("truncate cannot be lower than the coefficient level")
        self.truncate = truncate

    def __add__(self, other: object) -> "SigPoly":
        if other == 0:  # Allows sum([p, q, r]).
            return self
        if not isinstance(other, SigPoly):
            return NotImplemented
        self._check_dimension(other, "add")

        length = max(len(self.coefficients), len(other.coefficients))
        coefficients = []
        for index in range(length):
            left = self.coefficients[index] if index < len(self.coefficients) else self.zero
            right = other.coefficients[index] if index < len(other.coefficients) else other.zero
            coefficients.append(left + right)

        return SigPoly(
            self.dimension,
            min(self.truncate, other.truncate),
            coefficients,
            zero=self.zero,
            is_zero=self.is_zero,
        )

    def __radd__(self, other: object) -> "SigPoly":
        return self.__add__(other)

    def __sub__(self, other: object) -> "SigPoly":
        if other == 0:
            return self
        if not isinstance(other, SigPoly):
            return NotImplemented
        self._check_dimension(other, "subtract")

        length = max(len(self.coefficients), len(other.coefficients))
        coefficients = []
        for index in range(length):
            left = self.coefficients[index] if index < len(self.coefficients) else self.zero
            right = other.coefficients[index] if index < len(other.coefficients) else other.zero
            coefficients.append(left - right)

        return SigPoly(
            self.dimension,
            min(self.truncate, other.truncate),
            coefficients,
            zero=self.zero,
            is_zero=self.is_zero,
        )

    def scale(self, scalar: Any) -> "SigPoly":
        """Return this functional multiplied by ``scalar``."""
        return SigPoly(
            self.dimension,
            self.truncate,
            [scalar * coefficient for coefficient in self.coefficients],
            zero=self.zero,
            is_zero=self.is_zero,
        )

    def __rmul__(self, scalar: Any) -> "SigPoly":
        return self.scale(scalar)

    def __mul__(self, other: object) -> "SigPoly":
        """Multiply two signature functionals using the shuffle product."""
        if not isinstance(other, SigPoly):
            return NotImplemented
        self._check_dimension(other, "multiply")

        new_truncate = min(self.truncate, other.truncate)
        new_level = min(self.level + other.level, new_truncate)
        idx2word, word2idx = create_dict_sigs(self.dimension, new_level)
        coefficients = [self.zero for _ in idx2word]

        for left_index, left_coefficient in enumerate(self.coefficients):
            if self.is_zero(left_coefficient) or left_index >= len(idx2word):
                continue
            left_word = idx2word[left_index]

            for right_index, right_coefficient in enumerate(other.coefficients):
                if other.is_zero(right_coefficient) or right_index >= len(idx2word):
                    continue
                right_word = idx2word[right_index]
                if len(left_word) + len(right_word) > new_level:
                    continue

                product = left_coefficient * right_coefficient
                for word, multiplicity in word_shuffle_prod(left_word, right_word).items():
                    index = word2idx[word]
                    coefficients[index] += multiplicity * product

        return SigPoly(
            self.dimension,
            new_truncate,
            coefficients,
            zero=self.zero,
            is_zero=self.is_zero,
        )

    def tensor(self, other: object) -> "SigPoly":
        """Concatenate the words of two signature functionals."""
        if not isinstance(other, SigPoly):
            return NotImplemented
        self._check_dimension(other, "take a tensor product of")

        new_truncate = min(self.truncate, other.truncate)
        new_level = min(self.level + other.level, new_truncate)
        idx2word, word2idx = create_dict_sigs(self.dimension, new_level)
        coefficients = [self.zero for _ in idx2word]

        for left_index, left_coefficient in enumerate(self.coefficients):
            if self.is_zero(left_coefficient) or left_index >= len(idx2word):
                continue
            left_word = idx2word[left_index]

            for right_index, right_coefficient in enumerate(other.coefficients):
                if other.is_zero(right_coefficient) or right_index >= len(idx2word):
                    continue
                right_word = idx2word[right_index]
                if len(left_word) + len(right_word) > new_level:
                    continue

                word = left_word + right_word
                coefficients[word2idx[word]] += left_coefficient * right_coefficient

        return SigPoly(
            self.dimension,
            new_truncate,
            coefficients,
            zero=self.zero,
            is_zero=self.is_zero,
        )

    def inner(self, other: Sequence[Any] | np.ndarray | "SigPoly") -> Any:
        """Evaluate the coefficient-wise pairing with another signature."""
        if isinstance(other, SigPoly):
            values = other.coefficients
        elif isinstance(other, np.ndarray):
            values = other.reshape(-1)
        elif isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            values = other
        else:
            raise TypeError("other must be a SigPoly or one-dimensional sequence")

        length = min(len(self.coefficients), len(values))
        if length == 0:
            return self.zero
        return sum(
            (self.coefficients[index] * values[index] for index in range(length)),
            start=self.zero,
        )

    def _check_dimension(self, other: "SigPoly", operation: str) -> None:
        if self.dimension != other.dimension:
            raise ValueError(f"cannot {operation}: dimensions differ")

    def __str__(self) -> str:
        return (
            f"SigPoly(dimension={self.dimension}, truncate={self.truncate}, "
            f"level={self.level}, coefficients={len(self.coefficients)})"
        )

    def __repr__(self) -> str:
        return (
            f"SigPoly(dimension={self.dimension!r}, truncate={self.truncate!r}, "
            f"level={self.level!r}, coefficients_len={len(self.coefficients)!r})"
        )


@dataclass
class QuadraticSolution:
    """Result returned by :func:`solve_quadratic_objective`."""

    variables: tuple[sympy.Symbol, ...]
    coefficients: np.ndarray
    objective_value: float
    regularized_objective_value: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    sense: str
    ridge: float

    @property
    def coefficient_map(self) -> dict[sympy.Symbol, float]:
        """Map each symbolic coefficient to its optimized numeric value."""
        return dict(zip(self.variables, self.coefficients))


def solve_quadratic_objective(
    expression: sympy.Expr,
    variables: Sequence[sympy.Symbol] | None = None,
    *,
    sense: str = "min",
    ridge: float = 0.0,
    curvature_tolerance: float = 1e-9,
) -> QuadraticSolution:
    """Solve an unconstrained convex or concave quadratic objective.

    ``sense='min'`` requires a positive-definite Hessian; ``sense='max'``
    requires a negative-definite one. A small ridge can resolve numerically
    flat directions: it is added for minimization and subtracted for
    maximization. ``objective_value`` uses the original expression.

    ``curvature_tolerance`` is an absolute margin around zero. The Hessian's
    smallest eigenvalue must exceed it for a minimum, and its largest
    eigenvalue must be below its negative for a maximum.
    """
    if sense not in {"min", "max"}:
        raise ValueError("sense must be either 'min' or 'max'")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if curvature_tolerance < 0:
        raise ValueError("curvature_tolerance must be non-negative")

    if variables is None:
        variables = sorted(expression.free_symbols, key=_symbol_sort_key)
    variables = tuple(variables)
    if not variables:
        raise ValueError("the objective contains no variables")

    regularization_sign = 1.0 if sense == "min" else -1.0
    regularized = expression + regularization_sign * sympy.Float(ridge / 2) * sum(
        variable**2 for variable in variables
    )

    hessian_symbolic = sympy.hessian(regularized, variables)
    if any(entry.free_symbols for entry in hessian_symbolic):
        raise ValueError("the objective is not quadratic in the selected variables")

    hessian = np.asarray(hessian_symbolic, dtype=float)
    zero = {variable: 0 for variable in variables}
    gradient_at_zero = np.asarray(
        [sympy.diff(regularized, variable).subs(zero) for variable in variables],
        dtype=float,
    )

    eigenvalues = np.linalg.eigvalsh(hessian)
    if sense == "min" and eigenvalues[0] <= curvature_tolerance:
        raise ValueError(
            "minimization requires a positive-definite Hessian; "
            f"smallest eigenvalue is {eigenvalues[0]:.3e}"
        )
    if sense == "max" and eigenvalues[-1] >= -curvature_tolerance:
        raise ValueError(
            "maximization requires a negative-definite Hessian; "
            f"largest eigenvalue is {eigenvalues[-1]:.3e}"
        )

    try:
        coefficients = -np.linalg.solve(hessian, gradient_at_zero)
    except np.linalg.LinAlgError as error:
        raise ValueError("the Hessian is singular; try a small ridge") from error

    substitutions = dict(zip(variables, coefficients))
    objective_value = float(sympy.N(expression.subs(substitutions)))
    regularized_value = float(sympy.N(regularized.subs(substitutions)))

    return QuadraticSolution(
        variables=variables,
        coefficients=coefficients,
        objective_value=objective_value,
        regularized_objective_value=regularized_value,
        min_eigenvalue=float(eigenvalues[0]),
        max_eigenvalue=float(eigenvalues[-1]),
        condition_number=float(np.linalg.cond(hessian)),
        sense=sense,
        ridge=ridge,
    )


def _symbol_sort_key(symbol: sympy.Symbol) -> tuple[str, int | str]:
    """Sort symbols such as x2 before x10 while handling general names."""
    name = symbol.name
    prefix = name.rstrip("0123456789")
    suffix = name[len(prefix) :]
    return prefix, int(suffix) if suffix else name


def extrema_on_constraint(
    F: sympy.Expr,
    G: sympy.Expr,
    vars: Sequence[sympy.Symbol] | None = None,
    n_starts: int = 40,
    seed: int = 0,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Numerically find candidate extrema of ``F`` subject to ``G = 0``.

    Multiple SLSQP starts are used, and duplicate feasible candidates are
    removed. On non-compact feasible sets the returned candidates need not be
    global extrema.
    """
    if vars is None:
        vars = sorted(F.free_symbols.union(G.free_symbols), key=_symbol_sort_key)
    vars = list(vars)
    n_variables = len(vars)

    gradient_F = [sympy.diff(F, variable) for variable in vars]
    gradient_G = [sympy.diff(G, variable) for variable in vars]
    F_numeric = sympy.lambdify(vars, F, "numpy")
    G_numeric = sympy.lambdify(vars, G, "numpy")
    gradient_F_numeric = sympy.lambdify(vars, gradient_F, "numpy")
    gradient_G_numeric = sympy.lambdify(vars, gradient_G, "numpy")

    def objective(point: np.ndarray) -> float:
        return float(F_numeric(*point))

    def constraint_value(point: np.ndarray) -> float:
        return float(G_numeric(*point))

    def objective_jacobian(point: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_F_numeric(*point), dtype=float).reshape(-1)

    def constraint_jacobian(point: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_G_numeric(*point), dtype=float).reshape(-1)

    constraint = {
        "type": "eq",
        "fun": constraint_value,
        "jac": constraint_jacobian,
    }
    rng = np.random.default_rng(seed)
    starts = [np.zeros(n_variables), np.ones(n_variables), -np.ones(n_variables)]
    for scale in (0.1, 0.5, 1.0, 2.0):
        starts.extend(
            scale * rng.standard_normal(n_variables)
            for _ in range(max(1, n_starts // 4))
        )

    def deduplicate(
        points: list[tuple[np.ndarray, float]], atol: float = 1e-6
    ) -> list[tuple[np.ndarray, float]]:
        unique = []
        for point, value in points:
            if not any(
                np.allclose(point, previous, atol=atol, rtol=0)
                for previous, _ in unique
            ):
                unique.append((point, value))
        return unique

    def solve_one(sign: int) -> tuple[dict[str, Any] | None, list]:
        found = []
        for start in starts:
            result = minimize(
                fun=lambda point: sign * objective(point),
                x0=start,
                jac=lambda point: sign * objective_jacobian(point),
                constraints=[constraint],
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 3000, "disp": False},
            )
            if result.success and abs(constraint_value(result.x)) < tol:
                found.append((result.x.copy(), objective(result.x)))

        found = deduplicate(found)
        if not found:
            return None, []
        selector = min if sign == 1 else max
        best = selector(found, key=lambda candidate: candidate[1])
        return {"point": best[0], "value": best[1]}, found

    minimum, min_candidates = solve_one(sign=1)
    maximum, max_candidates = solve_one(sign=-1)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
    }


def constrained_extrema(
    F: sympy.Expr,
    G: sympy.Expr,
    vars: Sequence[sympy.Symbol] | None = None,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Find symbolic candidate extrema of ``F`` subject to ``G = 0``.

    This routine is intended for problems with a finite candidate set. It
    checks regular Lagrange-multiplier solutions and singular constraint
    points.
    """
    if vars is None:
        vars = sorted(F.free_symbols.union(G.free_symbols), key=_symbol_sort_key)
    vars = list(vars)
    lagrange_multiplier = sympy.symbols("_lambda", real=True)

    def is_real_expression(expression: sympy.Expr) -> bool:
        expression = sympy.simplify(expression)
        if expression.is_real is True:
            return True
        try:
            return abs(float(sympy.N(sympy.im(expression), 40))) < tol
        except (TypeError, ValueError):
            return False

    def feasible(point: dict[sympy.Symbol, sympy.Expr]) -> bool:
        try:
            return abs(float(sympy.N(G.subs(point), 40))) < tol
        except (TypeError, ValueError):
            return False

    def make_entry(point: dict[sympy.Symbol, sympy.Expr], source: str) -> dict:
        exact_point = tuple(sympy.simplify(point[variable]) for variable in vars)
        value = sympy.simplify(F.subs(point))
        return {
            "point": exact_point,
            "value": value,
            "point_numeric": tuple(
                float(sympy.N(sympy.re(component), 30)) for component in exact_point
            ),
            "value_numeric": float(sympy.N(sympy.re(value), 30)),
            "source": source,
        }

    candidates = []
    seen = set()
    lagrange_equations = [
        sympy.diff(F, variable)
        - lagrange_multiplier * sympy.diff(G, variable)
        for variable in vars
    ] + [G]
    lagrange_solutions = sympy.solve(
        lagrange_equations,
        vars + [lagrange_multiplier],
        dict=True,
    )
    for solution in lagrange_solutions:
        point = {
            variable: sympy.simplify(solution[variable])
            for variable in vars
            if variable in solution
        }
        if len(point) != len(vars):
            continue
        if all(is_real_expression(point[variable]) for variable in vars) and feasible(point):
            key = tuple(sympy.srepr(point[variable]) for variable in vars)
            if key not in seen:
                seen.add(key)
                candidates.append(make_entry(point, "lagrange"))

    singular_equations = [G] + [sympy.diff(G, variable) for variable in vars]
    for solution in sympy.solve(singular_equations, vars, dict=True):
        point = {
            variable: sympy.simplify(solution[variable])
            for variable in vars
            if variable in solution
        }
        if len(point) != len(vars):
            continue
        if all(is_real_expression(point[variable]) for variable in vars) and feasible(point):
            key = tuple(sympy.srepr(point[variable]) for variable in vars)
            if key not in seen:
                seen.add(key)
                candidates.append(make_entry(point, "singular"))

    candidates.sort(key=lambda candidate: candidate["value_numeric"])
    return {
        "variables": vars,
        "candidates": candidates,
        "minimum": candidates[0] if candidates else None,
        "maximum": candidates[-1] if candidates else None,
    }


def create_dict_sigs(dimension: int, level: int) -> tuple[dict[int, Word], dict[Word, int]]:
    """Return index/word lookup tables in ``esig`` signature order."""
    if dimension <= 0 or level < 0:
        raise ValueError("dimension must be positive and level non-negative")

    word2idx: dict[Word, int] = {}
    idx2word: dict[int, Word] = {}
    current_index = 0
    current_word: list[int] = []
    current_letter = 0
    for character in esig.sigkeys(dimension, level):
        if character == "(":
            current_word = []
            current_letter = 0
        elif character.isdigit():
            current_letter = 10 * current_letter + int(character)
        elif character == ",":
            current_word.append(current_letter)
            current_letter = 0
        elif character == ")":
            if current_letter != 0:
                current_word.append(current_letter)
            word = tuple(current_word)
            idx2word[current_index] = word
            word2idx[word] = current_index
            current_index += 1
    return idx2word, word2idx


def word_shuffle_prod(w1: Word, w2: Word) -> Dict[Word, int]:
    """Return every shuffle of two words together with its multiplicity."""
    w1 = tuple(w1)
    w2 = tuple(w2)
    memo: Dict[tuple[int, int], Dict[Word, int]] = {}

    def recurse(left_index: int, right_index: int) -> Dict[Word, int]:
        key = (left_index, right_index)
        if key in memo:
            return memo[key]
        if left_index == len(w1):
            memo[key] = {w2[right_index:]: 1}
            return memo[key]
        if right_index == len(w2):
            memo[key] = {w1[left_index:]: 1}
            return memo[key]

        output: Dict[Word, int] = {}
        for tail, multiplicity in recurse(left_index + 1, right_index).items():
            word = (w1[left_index],) + tail
            output[word] = output.get(word, 0) + multiplicity
        for tail, multiplicity in recurse(left_index, right_index + 1).items():
            word = (w2[right_index],) + tail
            output[word] = output.get(word, 0) + multiplicity
        memo[key] = output
        return output

    return recurse(0, 0)


def create_variable(dimension: int, level: int, truncate_level: int) -> SigPoly:
    """Create symbolic coefficients for all words through ``level``."""
    if level < 0:
        raise ValueError("level must be non-negative")
    if truncate_level < level:
        raise ValueError("truncate_level cannot be lower than level")

    length = sum(dimension**word_length for word_length in range(level + 1))
    coefficients = list(sympy.symbols(f"x0:{length}"))
    return SigPoly(
        dimension,
        truncate_level,
        coefficients,
        zero=sympy.Integer(0),
    )


def len2level(dimension: int, length: int) -> int:
    """Convert a complete coefficient-vector length to its signature level."""
    if dimension <= 0 or length <= 0:
        raise ValueError("dimension and length must be positive")
    total = 1
    level = 0
    while total < length:
        level += 1
        total += dimension**level
    if total != length:
        raise ValueError(
            f"{length} coefficients do not form complete levels in dimension {dimension}"
        )
    return level


def empirical_expected_signature(paths: np.ndarray, level: int) -> np.ndarray:
    """Estimate an expected signature and prepend the empty-word value."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_observations, dimension)")
    if level <= 0:
        raise ValueError("level must be positive")
    signatures = iisignature.sig(paths, level)
    return np.hstack(([1.0], np.mean(signatures, axis=0)))


def simulate_paths_arithmetic_brownian(
    m: int = 100,
    S0: float = 100.0,
    sigma: float = 1.0,
    seed: int = 42,
    T: float = 1.0,
    n: int = 252,
) -> np.ndarray:
    """Simulate time-augmented arithmetic Brownian price paths."""
    _validate_simulation_inputs(m, T, n)
    rng = np.random.default_rng(seed)
    dt = T / n
    increments = sigma * np.sqrt(dt) * rng.standard_normal((m, n))
    prices = np.cumsum(
        np.concatenate([S0 * np.ones((m, 1)), increments], axis=1),
        axis=1,
    )
    return _augment_with_time(prices, T)


def simulate_paths_geometric_brownian(
    m: int = 100,
    S0: float = 100.0,
    mu: float = 0.1,
    sigma: float = 1.0,
    seed: int = 42,
    T: float = 1.0,
    n: int = 252,
) -> np.ndarray:
    """Simulate time-augmented geometric Brownian price paths."""
    _validate_simulation_inputs(m, T, n)
    if S0 <= 0:
        raise ValueError("S0 must be positive for geometric Brownian motion")

    rng = np.random.default_rng(seed)
    dt = T / n
    brownian_increments = np.sqrt(dt) * rng.standard_normal((m, n))
    brownian_paths = np.cumsum(
        np.concatenate([np.zeros((m, 1)), brownian_increments], axis=1),
        axis=1,
    )
    time = np.linspace(0.0, T, n + 1)
    prices = S0 * np.exp(
        sigma * brownian_paths + (mu - 0.5 * sigma**2) * time[None, :]
    )
    return _augment_with_time(prices, T)


def simulate_paths_heston(
    m: int = 100,
    S0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    seed: int = 42,
    T: float = 1.0,
    n: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths with full-truncation Euler variance updates."""
    _validate_simulation_inputs(m, T, n)
    if S0 <= 0 or v0 < 0 or xi < 0 or not -1 <= rho <= 1:
        raise ValueError("invalid Heston parameters")

    rng = np.random.default_rng(seed)
    dt = T / n
    sqrt_dt = np.sqrt(dt)
    first_shocks = rng.standard_normal((m, n))
    second_shocks = rng.standard_normal((m, n))
    price_brownian = sqrt_dt * first_shocks
    variance_shock = xi * sqrt_dt * (
        rho * first_shocks + np.sqrt(1.0 - rho**2) * second_shocks
    )

    prices = np.zeros((m, n + 1), dtype=float)
    variances = np.zeros((m, n + 1), dtype=float)
    prices[:, 0] = S0
    variances[:, 0] = v0

    for step in range(n):
        current_variance = np.maximum(variances[:, step], 0.0)
        prices[:, step + 1] = prices[:, step] * np.exp(
            (mu - 0.5 * current_variance) * dt
            + np.sqrt(current_variance) * price_brownian[:, step]
        )
        variances[:, step + 1] = np.maximum(
            0.0,
            variances[:, step]
            + kappa * (theta - current_variance) * dt
            + np.sqrt(current_variance) * variance_shock[:, step],
        )

    return _augment_with_time(prices, T), variances


def _validate_simulation_inputs(m: int, T: float, n: int) -> None:
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if T <= 0:
        raise ValueError("T must be positive")


def _augment_with_time(prices: np.ndarray, T: float) -> np.ndarray:
    time = np.linspace(0.0, T, prices.shape[1])
    time_paths = np.broadcast_to(time, prices.shape)
    return np.stack([time_paths, prices], axis=2)


def expected_signature_time_augmented_brownian(
    sigma: float = 0.25,
    T: float = 1.0,
    level: int = 3,
) -> SigPoly:
    """Return Fawcett's expected signature for ``(t, sigma * W_t)``.

    The expected signature is the tensor exponential of
    ``T * (e_time + sigma**2 / 2 * e_price tensor e_price)``.
    """
    if sigma < 0 or T < 0 or level <= 0:
        raise ValueError("sigma and T must be non-negative and level positive")

    price_letter = SigPoly(2, level, [0, 0, 1])
    time_letter = SigPoly(2, level, [0, 1, 0])
    generator = T * (
        time_letter + (sigma**2 / 2.0) * price_letter.tensor(price_letter)
    )
    expected_signature = SigPoly(2, level, [1, 0, 0])
    tensor_power = generator
    for order in range(1, level + 1):
        expected_signature = expected_signature + tensor_power
        tensor_power = (1.0 / (order + 1)) * tensor_power.tensor(generator)
    return expected_signature


def exp_sigs_fawc(sigma: float = 0.25, T: float = 1.0, level: int = 3) -> SigPoly:
    """Backward-compatible alias for the descriptively named helper above."""
    return expected_signature_time_augmented_brownian(sigma=sigma, T=T, level=level)
