import iisignature
import esig
import sympy
from typing import Tuple, Dict
from numbers import Number
import numpy as np


class SigPoly:
    def __init__(self, dimension: int, truncate: int, coefficients: list, *, zero=0, is_zero=None):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive int")
        if not isinstance(truncate, int) or truncate <= 0:
            raise ValueError("level must be a positive int")
        if not isinstance(coefficients, list):
            raise TypeError("coefficients must be a list")

        self.dimension = dimension
        self.level = len2level(dimension,len(coefficients))
        self.coefficients = coefficients
        self.truncate = truncate

        self.zero = zero
        self.is_zero = is_zero if is_zero is not None else (lambda x: x == zero)

    def set_level(self, truncate):
        self.truncate = truncate

    def __add__(self, other):
        if other == 0:   # so sum([p,q,r]) works
            return self
        if not isinstance(other, SigPoly):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("cannot add: different dimension")

        if len(self.coefficients) <= len(other.coefficients):
            out_coeffs = []
            for i in range(len(self.coefficients)):
                out_coeffs.append(self.coefficients[i] + other.coefficients[i])
            for i in range(len(self.coefficients),len(other.coefficients)):
                out_coeffs.append(other.coefficients[i])
        elif len(self.coefficients) > len(other.coefficients):
            out_coeffs = []
            for i in range(len(other.coefficients)):
                out_coeffs.append(self.coefficients[i] + other.coefficients[i])
            for i in range(len(other.coefficients),len(self.coefficients)):
                out_coeffs.append(self.coefficients[i])

        return SigPoly(self.dimension, min(self.truncate, other.truncate), out_coeffs, zero = self.zero)

    def __radd__(self, other):
        # allows 0 + SigPoly and sum([...])
        return self.__add__(other)

    def __sub__(self, other):
        if other == 0:   # so sum([p,q,r]) works
            return self
        if not isinstance(other, SigPoly):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("cannot subtract: different dimension")

        if len(self.coefficients) <= len(other.coefficients):
            out_coeffs = []
            for i in range(len(self.coefficients)):
                out_coeffs.append(self.coefficients[i] - other.coefficients[i])
            for i in range(len(self.coefficients),len(other.coefficients)):
                out_coeffs.append(-other.coefficients[i])
        elif len(self.coefficients) > len(other.coefficients):
            out_coeffs = []
            for i in range(len(other.coefficients)):
                out_coeffs.append(self.coefficients[i] - other.coefficients[i])
            for i in range(len(other.coefficients),len(self.coefficients)):
                out_coeffs.append(self.coefficients[i])

        return SigPoly(self.dimension, min(self.truncate, other.truncate), out_coeffs,  zero = self.zero)

    def scale(self, c):
        return SigPoly(
            self.dimension,
            self.truncate,
            [c * a for a in self.coefficients],
            zero=self.zero,
            is_zero=self.is_zero
        )

    def __rmul__(self, c):
        # scalar * SigPoly
        return self.scale(c)


    def __mul__(self, other: "SigPoly") -> "SigPoly":
        if not isinstance(other, SigPoly):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("cannot multiply: different dimensions")

        d = self.dimension

        new_truncate = min(self.truncate,other.truncate)
        new_level = min(self.level + other.level ,new_truncate)
        idx2word, word2idx = create_dict_sigs(d, new_level)
        new_length = len(idx2word)

        # IMPORTANT: initialize with the right zero element, not integer 0
        out = [self.zero for _ in range(len(idx2word))]

        for i, ai in enumerate(self.coefficients):
            if self.is_zero(ai):
                continue
            if i >= new_length:
                continue
            w1 = idx2word[i]

            for j, bj in enumerate(other.coefficients):
                if j >= new_length:
                    continue
                if other.is_zero(bj) if hasattr(other, "is_zero") else (bj == other.zero if hasattr(other, "zero") else bj == 0):
                    continue

                w2 = idx2word[j]
                if len(w1) + len(w2) > new_level:
                    continue
                mult_fact = ai * bj  # works for polynomials etc.

                sh = word_shuffle_prod(w1, w2)
                for w, mult in sh.items():
                    out[word2idx[w]] = out[word2idx[w]] + mult_fact * mult
        return SigPoly(d, new_truncate, out, zero=self.zero, is_zero=self.is_zero)


    def __str__(self):
        return f"SigPoly(d={self.dimension}, truncate_level={self.truncate}, level={self.level} coeffs={self.coefficients})"

    def __repr__(self):
        # usually a more “debug” / unambiguous representation
        return f"SigPoly(dimension={self.dimension!r}, truncate_level={self.truncate!r}, level={self.level}, coefficients_len={len(self.coefficients)!r})"



    def tensor(self, other: "SigPoly") -> "SigPoly":
        if not isinstance(other, SigPoly):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("cannot multiply: different dimensions")

        d = self.dimension

        new_truncate = min(self.truncate,other.truncate)
        new_level = min(self.level + other.level ,new_truncate)
        idx2word, word2idx = create_dict_sigs(d, new_level)
        new_length = len(idx2word)

        # IMPORTANT: initialize with the right zero element, not integer 0
        out = [self.zero for _ in range(len(idx2word))]

        for i, ai in enumerate(self.coefficients):
            if self.is_zero(ai):
                continue
            if i >= new_length:
                continue
            w1 = idx2word[i]

            for j, bj in enumerate(other.coefficients):
                if j >= new_length:
                    continue
                if other.is_zero(bj) if hasattr(other, "is_zero") else (bj == other.zero if hasattr(other, "zero") else bj == 0):
                    continue


                w2 = idx2word[j]
                if len(w1) + len(w2) > new_level:
                    continue
                mult_fact = ai * bj  # works for polynomials etc.

                w1ow2 = w1 + w2

                out[word2idx[w1ow2]] = out[word2idx[w1ow2]] + mult_fact
        return SigPoly(d, new_truncate, out, zero=self.zero, is_zero=self.is_zero)

    def inner(self, other: "SigPoly"):
        if isinstance(self, list):
            L1 = self
        elif isinstance(self, SigPoly):
            L1 = self.coefficients
            l1 = len(L1)
        else:
            raise ValueError("Not implemented")

        if isinstance(other, list):
            L2 = other
        elif isinstance(other, np.ndarray):
            L2 = other
            l2 = other.shape[0]
        elif isinstance(other, SigPoly):
            L2 = other.coefficients
            l2 = len(L2)
        else:
            raise ValueError("Not implemented")


        out = L1[0] * L2[0]

        for i in range(1,min(l1,l2)):
            out += L1[i] * L2[i]
        return out


import sympy as sympy
import numpy as np
from scipy.optimize import minimize

def extrema_on_constraint(F, G, vars=None, n_starts=40, seed=0, tol=1e-8):
    """
    Numerically find min/max of F subject to G = 0.

    Parameters
    ----------
    F, G : sympy expressions
        Objective and constraint.
    vars : list of sympy symbols
        Variables, in the order you want.
    n_starts : int
        Number of random starting points.
    seed : int
        RNG seed.
    tol : float
        Tolerance for checking feasibility.

    Returns
    -------
    dict with keys:
        "minimum", "maximum", "min_candidates", "max_candidates"
    """

    if vars is None:
        vars = sorted(list(F.free_symbols.union(G.free_symbols)), key=lambda s: s.sort_key())

    n = len(vars)

    gradF = [sympy.diff(F, v) for v in vars]
    gradG = [sympy.diff(G, v) for v in vars]

    F_num = sympy.lambdify(vars, F, "numpy")
    G_num = sympy.lambdify(vars, G, "numpy")
    gradF_num = sympy.lambdify(vars, gradF, "numpy")
    gradG_num = sympy.lambdify(vars, gradG, "numpy")

    def f(x):
        return float(F_num(*x))

    def g(x):
        return float(G_num(*x))

    def jf(x):
        return np.array(gradF_num(*x), dtype=float).reshape(-1)

    def jg(x):
        return np.array(gradG_num(*x), dtype=float).reshape(-1)

    constraint = {
        "type": "eq",
        "fun": g,
        "jac": jg
    }

    rng = np.random.default_rng(seed)

    starts = [np.zeros(n), np.ones(n), -np.ones(n)]
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(max(1, n_starts // 4)):
            starts.append(scale * rng.standard_normal(n))

    def deduplicate(points, atol=1e-6):
        out = []
        for p, val in points:
            if not any(np.allclose(p, q, atol=atol, rtol=0) for q, _ in out):
                out.append((p, val))
        return out

    def solve_one(sign=1):
        found = []

        for x0 in starts:
            res = minimize(
                fun=lambda x: sign * f(x),
                x0=x0,
                jac=lambda x: sign * jf(x),
                constraints=[constraint],
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 3000, "disp": False}
            )

            if res.success and abs(g(res.x)) < tol:
                found.append((res.x.copy(), f(res.x)))

        found = deduplicate(found)

        if not found:
            return None, []

        if sign == 1:
            best = min(found, key=lambda t: t[1])
        else:
            best = max(found, key=lambda t: t[1])

        return {
            "point": best[0],
            "value": best[1]
        }, found

    minimum, min_candidates = solve_one(sign=1)
    maximum, max_candidates = solve_one(sign=-1)

    return {
        "minimum": minimum,
        "maximum": maximum,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates
    }



def constrained_extrema(F, G, vars=None, tol=1e-10):
    """
    Find extrema of the polynomial F subject to the constraint G = 0.

    Parameters
    ----------
    F : sympy expression
        Objective polynomial.
    G : sympy expression
        Constraint polynomial, defining the set {x : G(x) = 0}.
    vars : list of sympy symbols, optional
        Variables, in the desired order. If None, uses the sorted free symbols of F and G.
    tol : float
        Numerical tolerance for realness / feasibility checks.

    Returns
    -------
    result : dict
        {
            "variables": [...],
            "candidates": [
                {
                    "point": (... exact sympy tuple ...),
                    "value": exact sympy value,
                    "point_numeric": (... floats ...),
                    "value_numeric": float,
                    "source": "lagrange" or "singular"
                },
                ...
            ],
            "minimum": {...} or None,
            "maximum": {...} or None
        }

    Notes
    -----
    - This is meant for cases where the candidate set is finite.
    - If the feasible set is noncompact, a global min/max may fail to exist.
    - If SymPy cannot solve the system symbolically, you may need a numerical fallback.
    """

    if vars is None:
        vars = sorted(F.free_symbols.union(G.free_symbols), key=lambda s: s.sort_key())
    vars = list(vars)

    lam = sympy.symbols("_lambda", real=True)

    def is_real_expr(expr):
        expr = sympy.simplify(expr)
        if expr.is_real is True:
            return True
        im_part = sympy.N(sympy.im(expr), 40)
        try:
            return abs(float(im_part)) < tol
        except Exception:
            return False

    def feasible(point_dict):
        val = sympy.N(G.subs(point_dict), 40)
        try:
            return abs(float(val)) < tol
        except Exception:
            return False

    def to_numeric_tuple(point):
        out = []
        for p in point:
            p_num = sympy.N(sympy.re(p), 30)
            out.append(float(p_num))
        return tuple(out)

    def make_entry(point_dict, source):
        point = tuple(sympy.simplify(point_dict[v]) for v in vars)
        val = sympy.simplify(F.subs(point_dict))
        return {
            "point": point,
            "value": val,
            "point_numeric": to_numeric_tuple(point),
            "value_numeric": float(sympy.N(sympy.re(val), 30)),
            "source": source,
        }

    candidates = []
    seen = set()

    # 1) Regular Lagrange multiplier candidates
    lagrange_eqs = [sympy.diff(F, v) - lam * sympy.diff(G, v) for v in vars] + [G]
    lagrange_solutions = sympy.solve(lagrange_eqs, vars + [lam], dict=True)

    for sol in lagrange_solutions:
        point_dict = {v: sympy.simplify(sol[v]) for v in vars if v in sol}
        if len(point_dict) != len(vars):
            continue
        if all(is_real_expr(point_dict[v]) for v in vars) and feasible(point_dict):
            key = tuple(sympy.srepr(point_dict[v]) for v in vars)
            if key not in seen:
                seen.add(key)
                candidates.append(make_entry(point_dict, source="lagrange"))

    # 2) Singular feasible points: G = 0 and grad G = 0
    singular_eqs = [G] + [sympy.diff(G, v) for v in vars]
    singular_solutions = sympy.solve(singular_eqs, vars, dict=True)

    for sol in singular_solutions:
        point_dict = {v: sympy.simplify(sol[v]) for v in vars if v in sol}
        if len(point_dict) != len(vars):
            continue
        if all(is_real_expr(point_dict[v]) for v in vars) and feasible(point_dict):
            key = tuple(sympy.srepr(point_dict[v]) for v in vars)
            if key not in seen:
                seen.add(key)
                candidates.append(make_entry(point_dict, source="singular"))

    candidates.sort(key=lambda c: c["value_numeric"])

    return {
        "variables": vars,
        "candidates": candidates,
        "minimum": candidates[0] if candidates else None,
        "maximum": candidates[-1] if candidates else None,
    }




def create_dict_sigs(dimension,level):
    word2idx = {}
    idx2word = {}

    convert = esig.sigkeys(dimension,level)
    cur_idx = 0
    for i in range(len(convert)):
        if convert[i] == "(":
            new_word = []
            new_letter = int(0)
        elif convert[i] in {'0','1','2','3','4','5','6','7','8','9'}:
            new_letter = 10 * new_letter + int(convert[i])
        elif convert[i] == ",":
            new_word.append(new_letter)
            new_letter = int(0)
        elif convert[i] == ")":
            if new_letter != 0:
                new_word.append(new_letter)
                new_letter = int(0)
            idx2word[cur_idx] = tuple(new_word)
            word2idx[tuple(new_word)] = cur_idx
            cur_idx += 1

    return idx2word,word2idx


Word = Tuple[int, ...]


def word_shuffle_prod(w1: Word, w2: Word) -> Dict[Word, int]:
    """
    Shuffle product of two words w1 and w2.

    Returns a dict mapping each shuffled word to its multiplicity (non-negative int).
    Missing key means coefficient 0.
    """
    w1 = tuple(w1)
    w2 = tuple(w2)

    memo: Dict[tuple[int, int], Dict[Word, int]] = {}

    def rec(i: int, j: int) -> Dict[Word, int]:
        # i letters already taken from w1, j from w2
        key = (i, j)
        if key in memo:
            return memo[key]

        # if one word is exhausted, only one shuffle remains
        if i == len(w1):
            out = {w2[j:]: 1}
            memo[key] = out
            return out
        if j == len(w2):
            out = {w1[i:]: 1}
            memo[key] = out
            return out

        out: Dict[Word, int] = {}

        # take next from w1
        for tail, c in rec(i + 1, j).items():
            word = (w1[i],) + tail
            out[word] = out.get(word, 0) + c

        # take next from w2
        for tail, c in rec(i, j + 1).items():
            word = (w2[j],) + tail
            out[word] = out.get(word, 0) + c

        memo[key] = out
        return out

    return rec(0, 0)


def create_variable(dimension , level, truncate_level):
    if truncate_level < level:
      raise("truncation level can't be lower than coefficients list level")
    N,z = 1,1
    for i in range(level):
        z = z*dimension
        N += z
    N = int(N)
    xs = sympy.symbols(f"x0:{N}")   # tuple (x0, x1, ..., x_{N-1})
    coeffs = list(xs)
    ell = SigPoly(dimension, truncate_level, coeffs, zero=sympy.Integer(0))

    return ell


def len2level(dimension, length):
    s = 1
    level = 0
    while(length > s):
        s = 1 + dimension*s
        level += 1
    return level





def simulate_paths_arithmetic_brownian(m = 100, S0 = 100.0, sigma = 1, seed = 42, T = 1.0, n = 252):
    rng = np.random.default_rng(seed)
    dt = T / n
    sqdt = np.sqrt(dt)

    # Brownian increments (m paths, n steps)
    dB = sigma * sqdt * rng.standard_normal((m, n))
    dB_aug = np.concatenate([S0* np.ones((m,1)) , dB] , axis = 1)

    paths = np.cumsum(dB_aug, axis=1)

    t = np.linspace(0.0, T, n + 1)            # (n+1,)
    t_full = np.broadcast_to(t, (m, n + 1))   # (m, n+1) same for all paths

    full_paths = np.stack([t_full, paths], axis=2)  # (m, n+1, 2)

    return full_paths



def simulate_paths_geometric_brownian(m = 100, S0 = 100.0, mu = 0.1, sigma = 1.0, seed = 42, T = 1.0, n = 252):
    rng = np.random.default_rng(seed)
    dt = T / n
    sqdt = np.sqrt(dt)

    # Brownian increments (m paths, n steps)
    dB = sigma * sqdt * rng.standard_normal((m, n))
    dB_aug = np.concatenate([np.zeros((m,1)) , dB] , axis = 1)

    paths = np.cumsum(dB_aug, axis=1)

    t = np.linspace(0.0, T, n + 1)            # (n+1,)
    t_full = np.broadcast_to(t, (m, n + 1))   # (m, n+1) same for all paths

    paths = paths + (mu - sigma**2 / 2)*t_full
    paths = S0*np.exp(paths)

    full_paths = np.stack([t_full, paths], axis=2)  # (m, n+1, 2)

    return full_paths




def simulate_paths_heston(m = 100, S0 = 100.0, v0 = 0.04, mu = 0.05, kappa = 2.0, theta=0.04, xi=0.3, rho =-0.7, seed = 42, T = 1.0, n = 252):
    rng = np.random.default_rng(seed)
    dt = T / n
    sqdt = np.sqrt(dt)

    Z1 = rng.standard_normal((m, n))
    Z2 = rng.standard_normal((m, n))
    dW1 = Z1 * sqdt
    dW2 = rho * xi * dW1 + xi * sqdt * np.sqrt(1.0 - rho ** 2) * Z2

    S = np.zeros((m, n + 1), dtype=np.float64)
    v = np.zeros((m, n + 1), dtype=np.float64)

    S[:, 0] = S0
    v[:, 0] = v0

    for k in range(n):
        # Log-Euler update for price: guarantees S > 0
        S[:, k + 1] = S[:, k]*np.exp( (mu - 0.5 * v[:,k]) * dt + np.sqrt(v[:,k]) *dW1[:,k]   )

        v[:, k + 1] = np.maximum(0.0, v[:,k] + kappa * (theta - v[:, k]) * dt + np.sqrt(v[:, k]) * dW2[:, k])


    t = np.linspace(0.0, T, n + 1)            # (n+1,)
    t_full = np.broadcast_to(t, (m, n + 1))   # (m, n+1) same for all paths


    full_paths = np.stack([t_full, S], axis=2)  # (m, n+1, 2)

    return full_paths,v



def exp_sigs_fawc( sigma = 0.25, T = 1, level = 3):

  x = SigPoly(2,level,[0,0,1])
  xx = x.tensor(x)
  t = SigPoly(2,level,[0,1,0])
  X = T*(t + (sigma**2 / 2) * (xx))
  i = 1
  expected_sigs_fawcett = SigPoly(2,level,[1,0,0])

  Xpow = X

  while i <= level:
    expected_sigs_fawcett = expected_sigs_fawcett + Xpow

    i += 1
    Xpow = (1/i)*(Xpow.tensor(X))
  return expected_sigs_fawcett





