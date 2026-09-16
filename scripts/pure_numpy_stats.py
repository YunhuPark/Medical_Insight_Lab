"""Pure numpy/pandas fallback for scipy.stats / sklearn primitives.

Written 2026-09-16 to re-run Section 3 of paper_draft.md on a machine where an
application-control policy blocks scipy.sparse/scipy.linalg/sklearn's compiled
DLLs from loading (numpy and pandas load fine; only scipy/statsmodels/sklearn
are affected). If scipy/statsmodels/sklearn work in your environment, prefer
those directly -- this module exists only to route around that specific block,
not because these implementations are preferable in general.

Each function is checked against textbook values and hand-traced examples in
test_pure_numpy_stats.py. Notably: wilcoxon() defaults to correction=False to
match scipy.stats.wilcoxon's default (unlike mannwhitneyu, whose scipy default
is use_continuity=True, which this module also matches).
"""
import math
import numpy as np

# ---------- normal distribution ----------
_SQRT2 = math.sqrt(2.0)

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT2))

def norm_sf(x):
    return 1.0 - norm_cdf(x)

def _norm_ppf_scalar(pi):
    """Peter Acklam's rational approximation, refined by one Halley step."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if pi < p_low:
        q = math.sqrt(-2 * math.log(pi))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif pi <= p_high:
        q = pi - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - pi))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # one Halley refinement step for extra precision
    e = 0.5 * math.erfc(-x / _SQRT2) - pi
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return x

def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's algorithm + Halley refinement)."""
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(p).astype(float)
    out = np.empty_like(p)
    for i, pi in enumerate(p):
        if pi <= 0:
            out[i] = -np.inf
        elif pi >= 1:
            out[i] = np.inf
        else:
            out[i] = _norm_ppf_scalar(pi)
    return out[0] if scalar else out

# ---------- t distribution two-sided survival function ----------
def _betacf(a, b, x, itmax=200, eps=1e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

def _betainc_reg(a, b, x):
    """Regularized incomplete beta I_x(a,b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x)
    else:
        return 1.0 - (math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) / b) * _betacf(b, a, 1 - x)

def t_sf_two_sided(t_abs, dof):
    """two-sided p-value = P(|T| > t_abs) for Student-t with `dof` df."""
    t_abs = float(abs(t_abs))
    dof = float(dof)
    x = dof / (dof + t_abs * t_abs)
    p_one_tail_upper_half = 0.5 * _betainc_reg(dof / 2.0, 0.5, x)
    return 2.0 * p_one_tail_upper_half

# ---------- Mann-Whitney U (two-sided, normal approx with tie correction) ----------
def mannwhitneyu(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind='mergesort')
    ranks = np.empty(len(combined))
    sorted_vals = combined[order]
    i = 0
    tie_term = 0.0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    N = n1 + n2
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie_term / (N * (N - 1))))
    if sigma == 0:
        return U1, 1.0
    z_cc = (abs(U1 - mu) - 0.5) / sigma
    p = 2 * float(norm_sf(z_cc))
    p = min(p, 1.0)
    return U1, p

# ---------- Wilcoxon signed-rank test (paired, normal approximation) ----------
# NOTE: scipy.stats.wilcoxon defaults to correction=False (no continuity
# correction) for its normal approximation, unlike mannwhitneyu (which
# defaults use_continuity=True). Matching that default matters: psm_analysis.py
# calls stats.wilcoxon(...) with no explicit `correction` argument.
def wilcoxon(x, y, correction=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = len(d)
    absd = np.abs(d)
    order = np.argsort(absd, kind='mergesort')
    ranks = np.empty(n)
    sorted_abs = absd[order]
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    signed_ranks = ranks * np.sign(d)
    W = signed_ranks[signed_ranks > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if sigma == 0:
        return W, 1.0
    diff = abs(W - mu)
    if correction:
        diff = max(diff - 0.5, 0.0)
    z = diff / sigma
    p = 2 * float(norm_sf(z))
    p = min(p, 1.0)
    return W, p

# ---------- Standardize ----------
def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd

# ---------- Logistic regression via Newton-Raphson (IRLS) with L2 (matches sklearn C=1.0) ----------
def logistic_regression_fit(X, y, C=1.0, add_intercept=True, max_iter=200, tol=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if add_intercept:
        Xd = np.hstack([np.ones((n, 1)), X])
    else:
        Xd = X
    k = Xd.shape[1]
    beta = np.zeros(k)
    lam = 1.0 / C
    reg = np.full(k, lam)
    if add_intercept:
        reg[0] = 0.0
    for _ in range(max_iter):
        z = Xd @ beta
        z = np.clip(z, -35, 35)
        p_hat = 1.0 / (1.0 + np.exp(-z))
        W = p_hat * (1 - p_hat)
        W = np.clip(W, 1e-10, None)
        grad = Xd.T @ (y - p_hat) - reg * beta
        H = -(Xd.T * W) @ Xd - np.diag(reg)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta

def logistic_regression_predict_proba(X, beta, add_intercept=True):
    X = np.asarray(X, dtype=float)
    if add_intercept:
        X = np.hstack([np.ones((len(X), 1)), X])
    z = np.clip(X @ beta, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

# ---------- OLS via lstsq with classical SEs ----------
def ols_fit(X, y, add_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if add_intercept:
        Xd = np.hstack([np.ones((n, 1)), X])
    else:
        Xd = X
    k = Xd.shape[1]
    beta, _, rank, _ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    resid = y - yhat
    dof = n - rank
    s2 = float(resid @ resid) / dof
    XtX_pinv = np.linalg.pinv(Xd.T @ Xd)
    cov = s2 * XtX_pinv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_vals = np.array([t_sf_two_sided(t, dof) for t in t_stats])
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n_params = (1 if add_intercept else 0)
    r2_adj = 1 - (1 - r2) * (n - n_params) / dof if dof > 0 else r2
    return {
        'beta': beta, 'se': se, 't': t_stats, 'p': p_vals,
        'r2': r2, 'r2_adj': r2_adj, 'n': n, 'k': k, 'dof': dof, 'resid': resid,
    }
