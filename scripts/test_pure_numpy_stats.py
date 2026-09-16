import sys, math
import os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pure_numpy_stats as P

fail = []

def check(name, got, expect, tol):
    if isinstance(expect, bool):
        ok = bool(got) == expect
    else:
        ok = abs(float(got) - float(expect)) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} expect={expect!r} (tol={tol})")
    if not ok:
        fail.append(name)

print("=== norm_cdf / norm_sf (reference: standard normal table) ===")
check("norm_cdf(0)", float(P.norm_cdf(0.0)), 0.5, 1e-9)
check("norm_cdf(1.96)", float(P.norm_cdf(1.96)), 0.9750021, 1e-6)
check("norm_cdf(-1.96)", float(P.norm_cdf(-1.96)), 0.0249979, 1e-6)
check("norm_cdf(2.576)", float(P.norm_cdf(2.576)), 0.995003, 1e-5)
check("norm_sf(1.6449)", float(P.norm_sf(1.6449)), 0.05, 1e-4)

print("\n=== norm_ppf (inverse check) ===")
check("norm_ppf(0.975)", float(P.norm_ppf(0.975)), 1.959964, 1e-4)
check("norm_ppf(0.5)", float(P.norm_ppf(0.5)), 0.0, 1e-6)
check("norm_ppf(0.05)", float(P.norm_ppf(0.05)), -1.644854, 1e-4)
# round trip
for q in [0.001, 0.1, 0.4, 0.6, 0.9, 0.999]:
    x = float(P.norm_ppf(q))
    back = float(P.norm_cdf(x))
    check(f"roundtrip ppf->cdf at q={q}", back, q, 1e-6)

print("\n=== t_sf_two_sided (reference: known Student-t two-sided p-values) ===")
# t=2.0, df=10 -> two-sided p ~ 0.0734 (standard table)
check("t_sf(2.0, df=10)", P.t_sf_two_sided(2.0, 10), 0.0734, 2e-3)
# t=1.0, df=30 -> two-sided p ~ 0.3253
check("t_sf(1.0, df=30)", P.t_sf_two_sided(1.0, 30), 0.3253, 2e-3)
# large df should approach normal: t=1.96, df=100000
p_large = P.t_sf_two_sided(1.96, 100000)
p_norm = 2 * float(P.norm_sf(1.96))
check("t_sf(1.96, df=100000) ~ normal", p_large, p_norm, 1e-3)
# t=0.074 (matches the paper's parallel-trends stat), large df -> should be ~0.94
p_tiny = P.t_sf_two_sided(0.074, 500)
check("t_sf(0.074, df=500) ~ 0.94", p_tiny, 0.9411, 5e-3)

print("\n=== mannwhitneyu (reference: textbook example, scipy-verified values) ===")
# classic example: two small samples with a known U
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
U, p = P.mannwhitneyu(a, b)
check("mwu fully separated U", U, 0.0, 1e-9)
# symmetric/identical distributions -> p should be large (no significant difference)
rng = np.random.default_rng(0)
x = rng.normal(0, 1, 2000)
y = rng.normal(0, 1, 2000)
U2, p2 = P.mannwhitneyu(x, y)
print(f"  [INFO] identical-dist large-sample p (should be > 0.05 typically): p={p2:.4f}")
# shifted distributions -> p should be tiny
y2 = rng.normal(2, 1, 2000)
U3, p3 = P.mannwhitneyu(x, y2)
check("mwu clearly-shifted p is tiny", p3 < 1e-10, True, 0)

print("\n=== wilcoxon signed-rank (hand-traced tiny example, no ties) ===")
# d = x - y = [3, -1, 2, 5, -4]; |d| ranks (ascending): 1->r1, 2->r2, 3->r3, 4->r4, 5->r5
# signed ranks: +3->+r3, -1->-r1, +2->+r2, +5->+r5, -4->-r4
# W+ (sum of positive signed ranks) = r3+r2+r5 = 3+2+5 = 10; W- = r1+r4 = 1+4 = 5
# sanity: W+ + W- must equal n(n+1)/2 = 5*6/2 = 15
xw = np.array([13.0, 9.0, 12.0, 15.0, 6.0])
yw = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
Wstat, pw = P.wilcoxon(xw, yw)  # correction=False, matching scipy's default
check("wilcoxon hand-traced W+ = 10", Wstat, 10.0, 1e-9)
mu_check = 5 * 6 / 4.0
sigma_check = math.sqrt(5 * 6 * 11 / 24.0)
z_check = abs(Wstat - mu_check) / sigma_check
p_expected = 2 * float(P.norm_sf(z_check))
check("wilcoxon p matches hand-derived normal-approx formula (no correction)", pw, p_expected, 1e-9)
Wstat_c, pw_c = P.wilcoxon(xw, yw, correction=True)
z_check_c = (abs(Wstat - mu_check) - 0.5) / sigma_check
p_expected_c = 2 * float(P.norm_sf(z_check_c))
check("wilcoxon p with correction=True matches formula", pw_c, p_expected_c, 1e-9)
check("continuity correction makes p larger (less significant)", pw_c >= pw, True, 0)
# all-positive differences -> W should equal n(n+1)/2 (every rank positive)
d_allpos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
zeros = np.zeros(6)
Wpos, ppos = P.wilcoxon(d_allpos, zeros)
check("wilcoxon all-positive-diff W = n(n+1)/2", Wpos, 21.0, 1e-9)
check("wilcoxon all-positive-diff p is small", ppos < 0.05, True, 0)

print("\n=== logistic regression (reference: perfectly separable + known coef sign) ===")
rng = np.random.default_rng(1)
n = 2000
x1 = rng.normal(0, 1, n)
true_beta = np.array([-0.3, 1.2])  # intercept, slope
z = true_beta[0] + true_beta[1] * x1
p_true = 1 / (1 + np.exp(-z))
y = (rng.uniform(0, 1, n) < p_true).astype(float)
X = x1.reshape(-1, 1)
beta_hat = P.logistic_regression_fit(X, y, C=1e6, max_iter=200)  # near-unregularized to compare to true beta
print(f"  [INFO] recovered beta={beta_hat}, true={true_beta} (n={n}, should be reasonably close)")
check("logreg recovers slope sign/magnitude roughly", abs(beta_hat[1] - true_beta[1]) < 0.15, True, 0)
check("logreg recovers intercept roughly", abs(beta_hat[0] - true_beta[0]) < 0.15, True, 0)

print("\n=== OLS (reference: exact synthetic linear model) ===")
rng = np.random.default_rng(2)
n = 500
X = rng.normal(0, 1, (n, 2))
true_beta_ols = np.array([1.5, -2.0, 0.5])  # intercept, b1, b2
noise = rng.normal(0, 0.01, n)  # tiny noise so recovery should be near-exact
y = true_beta_ols[0] + X @ true_beta_ols[1:] + noise
fit = P.ols_fit(X, y)
print(f"  [INFO] recovered beta={fit['beta']}, true={true_beta_ols}")
check("ols beta[0] (intercept)", fit['beta'][0], true_beta_ols[0], 0.01)
check("ols beta[1]", fit['beta'][1], true_beta_ols[1], 0.01)
check("ols beta[2]", fit['beta'][2], true_beta_ols[2], 0.01)
check("ols r2 near 1 for near-noiseless fit", fit['r2'] > 0.999, True, 0)
# null case: y independent of X -> p-values should NOT be tiny (sanity, not exact)
y_null = rng.normal(0, 1, n)
fit_null = P.ols_fit(X, y_null)
print(f"  [INFO] null-model p-values (should mostly be > 0.01): {fit_null['p']}")

print("\n" + "=" * 50)
if fail:
    print(f"FAILURES: {fail}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
