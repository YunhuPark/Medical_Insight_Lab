"""
Re-runs the Medical_Insight_Lab pipeline for paper_draft.md Section 3, for
environments where scipy/statsmodels/sklearn cannot load (see
pure_numpy_stats.py for why this was needed on 2026-09-16). If those
libraries work for you, just re-run the actual analysis scripts listed
below directly -- this script exists as a fallback, not a replacement.

Uses the analysis scripts own exact methodology (FEAR_KEYWORDS, reference
date, covariates, filters, random seed 42) with scipy/statsmodels/sklearn
calls replaced by pure_numpy_stats.py.

Usage: python scripts/rerun_section3_no_deps.py

Mirrors:
  src/analysis/psm_analysis.py        -> PSM ATT (Section 3.4)
  src/analysis/fixed_effects.py       -> channel fixed effects (Section 3.5)
  src/analysis/covid_experiment.py    -> DiD (Section 3.6)
  src/analysis/effect_size_power.py   -> Hedge's g, SMD, parallel trends,
                                          Rosenbaum bounds (Sections 3.4/3.6)
  src/analysis/type_fear_interaction.py (partially, via effect_size_power's
                                          Type x Fear loop) -> Section 3.7
"""
import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pure_numpy_stats as P
from datetime import datetime, timezone

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "4_medical_full_dataset.csv")

FEAR_KEYWORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

def load_base():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Year'] = df['Published_At'].dt.year
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['VPD'] = df['Views'] / df['Age_Days']
    df['Log_Views'] = np.log1p(df['Views'])
    df['Log_VPD'] = np.log1p(df['VPD'])
    df['Log_Age'] = np.log1p(df['Age_Days'])
    title = df['Title'].astype(str)
    df['Fear_Score'] = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Has_Fear'] = (df['Fear_Score'] > 0).astype(int)
    df['Title_Length'] = title.str.len()
    df['Has_Number'] = title.str.contains(r'\d').astype(int)
    df['Is_Shorts'] = (df['Duration_Sec'] <= 60).astype(int)
    ch_med = df.groupby('Channel')['Views'].median()
    df['Ch_Med_Views'] = df['Channel'].map(ch_med)
    df['Log_Ch_Med'] = np.log1p(df['Ch_Med_Views'])
    if 'Medical_Score' in df.columns:
        df['Medical_Score'] = df['Medical_Score'].fillna(0.5)
    else:
        df['Medical_Score'] = 0.5
    return df

def skewness(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=0)
    return (np.sum((x - m) ** 3) / n) / (s ** 3)

print("=" * 70)
print("SECTION 3.1: DESCRIPTIVE STATISTICS")
print("=" * 70)
df = load_base()
n_total = len(df)
views = df['Views'].values
med = np.median(views)
q1, q3 = np.percentile(views, [25, 75])
skew = skewness(views)
fear_pct = df['Has_Fear'].mean() * 100
fear_n = int(df['Has_Fear'].sum())
type_counts = df['Type'].value_counts(normalize=True) * 100 if 'Type' in df.columns else None
shorts_pct = df['Is_Shorts'].mean() * 100

print(f"n videos                 = {n_total:,}")
print(f"median views             = {med:,.0f}  (IQR: {q1:,.0f}-{q3:,.0f})")
print(f"skewness (views)         = {skew:.2f}")
print(f"fear-keyword videos      = {fear_pct:.1f}% (n={fear_n:,})")
if type_counts is not None:
    print(f"channel type breakdown   = {dict(type_counts.round(1))}")
print(f"shorts (<=60s)           = {shorts_pct:.1f}%")

print()
print("=" * 70)
print("SECTION 3.2: NAIVE COMPARISON (H1)")
print("=" * 70)
f = df[df['Has_Fear'] == 1]
nf = df[df['Has_Fear'] == 0]
mf, mnf = f['Views'].median(), nf['Views'].median()
U, p_naive = P.mannwhitneyu(f['Views'].values, nf['Views'].values)
print(f"median views  fear={mf:,.0f}  nofear={mnf:,.0f}  ratio={mf/mnf:.4f}")
print(f"Mann-Whitney U={U:.0f}  p={p_naive:.4g}")

print()
print("=" * 70)
print("SECTION 3.3: AGE-ADJUSTED COMPARISON, VPD (H2)")
print("=" * 70)
mvf, mvnf = f['VPD'].median(), nf['VPD'].median()
Uv, p_vpd = P.mannwhitneyu(f['VPD'].values, nf['VPD'].values)
print(f"median VPD  fear={mvf:.2f}  nofear={mvnf:.2f}  ratio={mvf/mvnf:.4f}")
print(f"Mann-Whitney U={Uv:.0f}  p={p_vpd:.4g}")

print()
print("=" * 70)
print("SECTION 3.4: PSM (H3) -- mirrors src/analysis/psm_analysis.py exactly")
print("=" * 70)

def run_psm(df):
    df = df.copy()
    kw_dummies = pd.get_dummies(df['Keyword'], prefix='kw', drop_first=True)
    covariates = ['Log_Ch_Med', 'Log_Age', 'Title_Length', 'Has_Number', 'Medical_Score']
    X_cov = pd.concat([df[covariates], kw_dummies], axis=1).astype(float).fillna(0)
    T = df['Has_Fear'].values.astype(float)

    X_scaled = P.standardize(X_cov.values)
    beta = P.logistic_regression_fit(X_scaled, T, C=1.0, max_iter=300)
    ps = P.logistic_regression_predict_proba(X_scaled, beta)
    df['Propensity_Score'] = ps

    treated_idx = df.index[df['Has_Fear'] == 1].tolist()
    control_idx = df.index[df['Has_Fear'] == 0].tolist()
    pos = {idx: i for i, idx in enumerate(df.index)}
    ps_treated = ps[[pos[i] for i in treated_idx]]
    ps_control = ps[[pos[i] for i in control_idx]]

    logit_ps = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
    caliper_logit = 0.2 * logit_ps.std()
    logit_treated = logit_ps[[pos[i] for i in treated_idx]]
    logit_control = logit_ps[[pos[i] for i in control_idx]]

    # 1-NN matching on a 1D scale = for each treated point, nearest control point.
    order = np.argsort(logit_control)
    sorted_control = logit_control[order]
    sorted_control_idx = np.array(control_idx)[order]
    insert_pos = np.searchsorted(sorted_control, logit_treated)
    matched_control_idx, distances = [], []
    for t_i, ip in enumerate(insert_pos):
        candidates = []
        if ip > 0:
            candidates.append(ip - 1)
        if ip < len(sorted_control):
            candidates.append(ip)
        best = min(candidates, key=lambda c: abs(sorted_control[c] - logit_treated[t_i]))
        matched_control_idx.append(sorted_control_idx[best])
        distances.append(abs(sorted_control[best] - logit_treated[t_i]))
    distances = np.array(distances)

    valid_pairs = [(t, c) for t, c, d in zip(treated_idx, matched_control_idx, distances) if d <= caliper_logit]
    print(f"PSM matched pairs: {len(valid_pairs)} (total treated: {len(treated_idx)}, logit caliper={caliper_logit:.4f})")

    t_idx = [p[0] for p in valid_pairs]
    c_idx = [p[1] for p in valid_pairs]

    treated_views = df.loc[t_idx, 'Views']
    control_views = df.loc[c_idx, 'Views']
    ATT_ratio = treated_views.median() / control_views.median()
    W, p_psm = P.wilcoxon(df.loc[t_idx, 'Log_Views'].values, df.loc[c_idx, 'Log_Views'].values)

    rng = np.random.default_rng(42)
    t_arr, c_arr = treated_views.values, control_views.values
    boot_ratios = []
    for _ in range(2000):
        bt = rng.choice(t_arr, size=len(t_arr), replace=True)
        bc = rng.choice(c_arr, size=len(c_arr), replace=True)
        if np.median(bc) > 0:
            boot_ratios.append(np.median(bt) / np.median(bc))
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])

    print(f"treated median views: {treated_views.median():,.0f}   control median views: {control_views.median():,.0f}")
    print(f"ATT ratio: {ATT_ratio:.4f}   Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Wilcoxon signed-rank W={W:.0f}  p={p_psm:.4g}")

    def smd_fn(a, b):
        pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        return abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

    print("\ncovariate balance (SMD):")
    key_covs = ['Log_Ch_Med', 'Log_Age', 'Title_Length', 'Has_Number']
    for cov in key_covs:
        pre = smd_fn(df[df['Has_Fear'] == 1][cov], df[df['Has_Fear'] == 0][cov])
        post = smd_fn(df.loc[t_idx, cov], df.loc[c_idx, cov])
        print(f"  {cov:<16} pre={pre:.4f}  post={post:.4f}")

    return df, t_idx, c_idx, ATT_ratio, ci_lo, ci_hi, p_psm

df_psm, t_idx, c_idx, ATT_ratio, ci_lo, ci_hi, p_psm = run_psm(df)

print()
print("=" * 70)
print("SECTION 3.5: CHANNEL FIXED EFFECTS (H4) -- mirrors fixed_effects.py")
print("=" * 70)

def run_fixed_effects(df_full):
    df = df_full.copy()
    df['Solution_Score'] = 0  # not needed for Fear_Score coefficient; keep column count minimal
    SOLUTION_KEYWORDS = ['방법','치료','완치','개선','해결','예방','관리','치유','회복','낫는','좋아지','극복']
    title = df['Title'].astype(str)
    df['Solution_Score'] = title.apply(lambda x: sum(1 for kw in SOLUTION_KEYWORDS if kw in x))
    df['Has_Question'] = title.str.contains(r'\?|？').astype(int)
    df['Is_Long'] = (df['Duration_Sec'] > 600).astype(int)

    ch_counts = df['Channel'].value_counts()
    valid_channels = ch_counts[ch_counts >= 5].index
    df_fe = df[df['Channel'].isin(valid_channels)].copy()
    print(f"fixed-effects sample: {len(df_fe)} videos / {df_fe['Channel'].nunique()} channels")

    feat = ['Fear_Score', 'Solution_Score', 'Title_Length', 'Has_Number',
            'Has_Question', 'Medical_Score', 'Is_Long', 'Log_Age']
    X1 = df_fe[feat].astype(float).values
    m1 = P.ols_fit(X1, df_fe['Log_Views'].values)

    ch_dummies = pd.get_dummies(df_fe['Channel'], prefix='ch', drop_first=True).astype(float)
    X2 = pd.concat([df_fe[feat].astype(float).reset_index(drop=True),
                     ch_dummies.reset_index(drop=True)], axis=1).values
    m2 = P.ols_fit(X2, df_fe['Log_Views'].values)

    df_fe['Log_VPD'] = np.log1p(df_fe['Views'] / df_fe['Age_Days'])
    m3 = P.ols_fit(X2, df_fe['Log_VPD'].values)

    fear_col = 0  # Fear_Score is column 0 in `feat`, and beta[0] is intercept -> Fear_Score coef is beta index 1
    for label, m in [("OLS, no channel control", m1),
                      ("channel FE (Log_Views)", m2),
                      ("channel FE (Log_VPD)", m3)]:
        coef = m['beta'][1]  # index 0 = intercept, index 1 = Fear_Score
        pval = m['p'][1]
        pct = (np.exp(coef) - 1) * 100
        print(f"  {label:<26} R2={m['r2']:.4f} adjR2={m['r2_adj']:.4f}  Fear_Score: {pct:+.1f}%  p={pval:.4g}")

    return m1, m2, m3

m1, m2, m3 = run_fixed_effects(df)

print()
print("=" * 70)
print("SECTION 3.6: COVID-19 DiD -- mirrors covid_experiment.py exactly")
print("=" * 70)

def run_did(df_full):
    df = df_full.copy()
    df['Post_COVID'] = (df['Year'] >= 2020).astype(int)
    df_main = df[(df['Year'] >= 2015) & (df['Year'] <= 2025)]

    groups = {
        'pre_nf':  df_main[(df_main['Post_COVID'] == 0) & (df_main['Has_Fear'] == 0)]['Log_VPD'],
        'pre_f':   df_main[(df_main['Post_COVID'] == 0) & (df_main['Has_Fear'] == 1)]['Log_VPD'],
        'post_nf': df_main[(df_main['Post_COVID'] == 1) & (df_main['Has_Fear'] == 0)]['Log_VPD'],
        'post_f':  df_main[(df_main['Post_COVID'] == 1) & (df_main['Has_Fear'] == 1)]['Log_VPD'],
    }
    for k, v in groups.items():
        print(f"  n_{k} = {len(v)}")
    means = {k: v.mean() for k, v in groups.items()}
    did = (means['post_f'] - means['post_nf']) - (means['pre_f'] - means['pre_nf'])
    print(f"DiD estimate (log scale): {did:+.4f}   -> {(np.exp(did)-1)*100:+.2f}%")

    rng = np.random.default_rng(42)
    pre_f, pre_nf = groups['pre_f'].values, groups['pre_nf'].values
    post_f, post_nf = groups['post_f'].values, groups['post_nf'].values
    boots = []
    for _ in range(2000):
        d = (rng.choice(post_f, len(post_f), True).mean() - rng.choice(post_nf, len(post_nf), True).mean()) - \
            (rng.choice(pre_f, len(pre_f), True).mean() - rng.choice(pre_nf, len(pre_nf), True).mean())
        boots.append(d)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    print(f"Bootstrap 95% CI (log scale): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"Bootstrap 95% CI (pct effect): [{(np.exp(ci_lo)-1)*100:+.1f}%, {(np.exp(ci_hi)-1)*100:+.1f}%]")
    return did, ci_lo, ci_hi

did, did_ci_lo, did_ci_hi = run_did(df)

print()
print("=" * 70)
print("SECTION 3.4b/3.6b: effect_size_power.py -- Hedge's g, SMD(v2 spec),")
print("parallel trends test, Rosenbaum bounds")
print("=" * 70)

def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    pooled_var = ((n1-1)*np.var(x,ddof=1) + (n2-1)*np.var(y,ddof=1)) / (n1+n2-2)
    d = (np.mean(x)-np.mean(y)) / np.sqrt(pooled_var)
    cf = 1 - 3/(4*(n1+n2-2)-1)
    return d*cf

def rank_biserial_r(x, y):
    U, _ = P.mannwhitneyu(x, y)
    return 1 - (2*U)/(len(x)*len(y))

def mw_power(n1, n2, effect_r, alpha=0.05):
    z_alpha = float(P.norm_ppf(1 - alpha/2))
    se = math.sqrt((n1+n2+1)/(3*n1*n2))
    ncp = abs(effect_r)/se if se > 0 else 0
    power = 1 - float(P.norm_cdf(z_alpha - ncp))
    return min(power, 1.0)

def smd_fn(x_treated, x_control):
    pooled_std = np.sqrt((np.var(x_treated,ddof=1) + np.var(x_control,ddof=1))/2)
    return 0.0 if pooled_std == 0 else (np.mean(x_treated)-np.mean(x_control))/pooled_std

df_reg = df[df['Is_Shorts'] == 0].copy()
fear_lv, nofear_lv = df_reg[df_reg['Has_Fear']==1]['Log_Views'], df_reg[df_reg['Has_Fear']==0]['Log_Views']
fear_vpd, nofear_vpd = df_reg[df_reg['Has_Fear']==1]['Log_VPD'], df_reg[df_reg['Has_Fear']==0]['Log_VPD']

print("\n[Effect sizes]")
for name, x, y in [("naive (Log_Views)", fear_lv, nofear_lv), ("age-adjusted (Log_VPD)", fear_vpd, nofear_vpd)]:
    _, p_mw = P.mannwhitneyu(x.values, y.values)
    g = cohens_d(x.values, y.values)
    r = rank_biserial_r(x.values, y.values)
    pwr = mw_power(len(x), len(y), r)
    print(f"  {name}: g={g:+.4f}, r={r:+.4f}, power={pwr:.3f}, p={p_mw:.4g}")

if 'Type' in df_reg.columns:
    for t in ['General', 'Medical Pro']:
        sub = df_reg[df_reg['Type']==t]
        f_, nf_ = sub[sub['Has_Fear']==1]['Log_Views'], sub[sub['Has_Fear']==0]['Log_Views']
        if len(f_) >= 5 and len(nf_) >= 5:
            _, p_mw = P.mannwhitneyu(f_.values, nf_.values)
            g = cohens_d(f_.values, nf_.values); r = rank_biserial_r(f_.values, nf_.values)
            ratio = np.exp(f_.median()) / np.exp(nf_.median())
            print(f"  Type x Fear [{t}] (n_fear={len(f_)}, n_nofear={len(nf_)}): "
                  f"median-Views ratio~{ratio:.3f}, g={g:+.4f}, p={p_mw:.4g}")

print("\n[PSM for SMD -- effect_size_power.py's separate spec: excludes Shorts, raw-ps caliper 0.05*std]")
def run_psm_for_smd(df_full):
    dfp = df_full[df_full['Is_Shorts']==0].copy()
    covariates = ['Log_Ch_Med','Log_Age','Title_Length','Has_Number','Medical_Score']
    kw_dummies = pd.get_dummies(dfp['Keyword'], prefix='kw', drop_first=True)
    X_cov = pd.concat([dfp[covariates], kw_dummies], axis=1).astype(float).fillna(0)
    T = dfp['Has_Fear'].values.astype(float)
    Xs = P.standardize(X_cov.values)
    beta = P.logistic_regression_fit(Xs, T, C=1.0, max_iter=300)
    ps = P.logistic_regression_predict_proba(Xs, beta)
    dfp = dfp.copy(); dfp['PS'] = ps
    treated_idx = dfp.index[dfp['Has_Fear']==1].tolist()
    control_idx = dfp.index[dfp['Has_Fear']==0].tolist()
    pos = {idx:i for i,idx in enumerate(dfp.index)}
    ps_t = ps[[pos[i] for i in treated_idx]]
    ps_c = ps[[pos[i] for i in control_idx]]
    order = np.argsort(ps_c); sorted_c = ps_c[order]; sorted_c_idx = np.array(control_idx)[order]
    ip = np.searchsorted(sorted_c, ps_t)
    matched_c, dists = [], []
    for ti, p_ in enumerate(ip):
        cands = [x for x in (p_-1, p_) if 0 <= x < len(sorted_c)]
        best = min(cands, key=lambda c: abs(sorted_c[c]-ps_t[ti]))
        matched_c.append(sorted_c_idx[best]); dists.append(abs(sorted_c[best]-ps_t[ti]))
    dists = np.array(dists)
    caliper = 0.05*ps.std()
    valid = [(t,c) for t,c,d in zip(treated_idx, matched_c, dists) if d <= caliper]
    print(f"  matched pairs: {len(valid)} (treated total: {len(treated_idx)}, caliper={caliper:.4f})")
    t_idx = [p[0] for p in valid]; c_idx = [p[1] for p in valid]
    for cov in covariates:
        pre_t, pre_c = dfp.loc[dfp['Has_Fear']==1, cov].dropna(), dfp.loc[dfp['Has_Fear']==0, cov].dropna()
        post_t, post_c = dfp.loc[t_idx, cov].dropna(), dfp.loc[c_idx, cov].dropna()
        smd_pre, smd_post = smd_fn(pre_t.values, pre_c.values), smd_fn(post_t.values, post_c.values)
        print(f"  {cov:<16} SMD {smd_pre:+.4f} -> {smd_post:+.4f}")
    return dfp, t_idx, c_idx

dfp2, t2_idx, c2_idx = run_psm_for_smd(df)

print("\n[Parallel trends test, pre-COVID 2015-2019, excludes Shorts]")
def parallel_trends_test(df_full):
    pre = df_full[(df_full['Year']>=2015)&(df_full['Year']<=2019)&(df_full['Is_Shorts']==0)].copy()
    pre['Year_c'] = pre['Year'] - 2017
    pre['Interaction'] = pre['Year_c'] * pre['Has_Fear']
    X = pre[['Year_c','Has_Fear','Interaction']].values.astype(float)
    y = pre['Log_VPD'].values
    fit = P.ols_fit(X, y)
    print(f"  n_pre = {len(pre)}")
    print(f"  interaction coef = {fit['beta'][3]:+.6f}  t={fit['t'][3]:+.4f}  p={fit['p'][3]:.4g}")
    return fit, pre

pt_fit, pre_df = parallel_trends_test(df)

print("\n[Rosenbaum sensitivity bounds -- on the PSM-for-SMD matched pairs]")
def rosenbaum_bounds(treated_log, control_log, gammas=None):
    if gammas is None:
        gammas = [1.0, 1.05, 1.10, 1.25, 1.5, 2.0, 3.0]
    n = len(treated_log)
    diff = treated_log - control_log
    T_plus = int(np.sum(diff > 0))
    out = []
    for gamma in gammas:
        e_min = n/(1+gamma)
        var = n*gamma/(1+gamma)**2
        z_left = (T_plus-e_min)/math.sqrt(var) if var>0 else 0
        p_upper = min(2*float(P.norm_cdf(z_left)), 1.0)
        out.append((gamma, p_upper, T_plus, n))
    return out

t_lv = dfp2.loc[t2_idx, 'Log_Views'].values
c_lv = dfp2.loc[c2_idx, 'Log_Views'].values
for gamma, p_upper, T_plus, n in rosenbaum_bounds(t_lv, c_lv):
    print(f"  Gamma={gamma:.2f}: p_upper={p_upper:.4f}  (T+={T_plus}/{n})  {'still sig' if p_upper<0.05 else '<- flips here'}")
