"""
SCI-level statistical reporting
- Cohen's d / Hedge's g + rank-biserial r for all main comparisons
- Post-hoc statistical power (Mann-Whitney approximation)
- PSM covariate balance: Standardized Mean Differences (SMD) before/after matching
- Parallel trends assumption test for DiD (pre-COVID period regression)
- Rosenbaum sensitivity bounds (Gamma sensitivity)
"""
import os, sys, warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime, timezone

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')

FEAR_KEYWORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

# ── 효과크기 함수들 ─────────────────────────────────────────

def cohens_d(x, y):
    """Hedge's g (bias-corrected Cohen's d for unequal n)"""
    n1, n2 = len(x), len(y)
    pooled_var = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)
    # Hedge's correction factor
    cf = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    return d * cf

def rank_biserial_r(x, y):
    """Rank-biserial correlation — effect size for Mann-Whitney U"""
    U, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    return 1 - (2 * U) / (len(x) * len(y))

def interpret_d(d):
    a = abs(d)
    if a < 0.2:  return 'negligible'
    elif a < 0.5: return 'small'
    elif a < 0.8: return 'medium'
    else:         return 'large'

def mw_power(n1, n2, effect_r, alpha=0.05):
    """
    Post-hoc power for Mann-Whitney via normal approximation.
    Uses the relationship between rank-biserial r and the standardized effect.
    """
    # Convert r to z for power calculation
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
    # Expected U under H1
    ncp = abs(effect_r) / se if se > 0 else 0
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    return min(power, 1.0)

def smd(x_treated, x_control):
    """Standardized Mean Difference for covariate balance check"""
    pooled_std = np.sqrt((np.var(x_treated, ddof=1) + np.var(x_control, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x_treated) - np.mean(x_control)) / pooled_std

# ── 데이터 준비 ─────────────────────────────────────────────

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Year']     = df['Published_At'].dt.year
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['VPD']      = df['Views'] / df['Age_Days']
    df['Log_VPD']  = np.log1p(df['VPD'])
    df['Log_Views']= np.log1p(df['Views'])

    title = df['Title'].astype(str)
    df['Fear_Score']   = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Has_Fear']     = (df['Fear_Score'] > 0).astype(int)
    df['Title_Length'] = title.str.len()
    df['Has_Number']   = title.str.contains(r'\d').astype(int)
    df['Is_Shorts']    = (df['Duration_Sec'] <= 60).astype(int)

    ch_med = df.groupby('Channel')['Views'].median()
    df['Ch_Med_Views'] = df['Channel'].map(ch_med)
    df['Log_Ch_Med']   = np.log1p(df['Ch_Med_Views'])
    df['Log_Age']      = np.log1p(df['Age_Days'])

    df['Medical_Score'] = df['Medical_Score'].fillna(0.5)
    return df

# ── PSM 재실행 (SMD 계산용) ──────────────────────────────────

def run_psm_for_smd(df):
    df_psm = df[df['Is_Shorts'] == 0].copy()
    covariates = ['Log_Ch_Med', 'Log_Age', 'Title_Length', 'Has_Number', 'Medical_Score']
    kw_dummies = pd.get_dummies(df_psm['Keyword'], prefix='kw', drop_first=True)
    X_cov = pd.concat([df_psm[covariates], kw_dummies], axis=1).astype(float).fillna(0)
    T = df_psm['Has_Fear']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cov)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    df_psm = df_psm.copy()
    df_psm['PS'] = ps

    treated_idx = df_psm[df_psm['Has_Fear']==1].index.tolist()
    control_idx = df_psm[df_psm['Has_Fear']==0].index.tolist()
    ps_t = ps[df_psm.index.get_indexer(treated_idx)]
    ps_c = ps[df_psm.index.get_indexer(control_idx)]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ps_c.reshape(-1,1))
    distances, matched = nn.kneighbors(ps_t.reshape(-1,1))
    matched_c = [control_idx[i] for i in matched.flatten()]
    caliper = 0.05 * ps.std()
    valid = [(t, c) for t, c, d in zip(treated_idx, matched_c, distances.flatten()) if d <= caliper]

    t_idx = [p[0] for p in valid]
    c_idx = [p[1] for p in valid]
    return df_psm, t_idx, c_idx, covariates

# ── 평행추세 검정 ─────────────────────────────────────────────

def parallel_trends_test(df):
    """
    Formal parallel trends test:
    Regress Log_VPD on Year × Has_Fear for pre-COVID period (2015-2019).
    H0: interaction coefficient = 0 (parallel trends hold)
    """
    pre = df[(df['Year'] >= 2015) & (df['Year'] <= 2019) & (df['Is_Shorts'] == 0)].copy()
    pre['Year_c']      = pre['Year'] - 2017  # center
    pre['Interaction'] = pre['Year_c'] * pre['Has_Fear']

    X = pre[['Year_c', 'Has_Fear', 'Interaction']].values
    X = np.hstack([np.ones((len(X), 1)), X])
    y = pre['Log_VPD'].values

    # OLS
    try:
        beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        n, k = len(y), X.shape[1]
        s2 = np.dot(residuals, residuals) / (n - k)
        cov_beta = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        t_stats = beta / se
        p_vals  = 2 * stats.t.sf(np.abs(t_stats), df=n - k)
        return {
            'coef_interaction': beta[3],
            'se_interaction':   se[3],
            't_stat':           t_stats[3],
            'p_value':          p_vals[3],
            'n_pre':            len(pre),
            'pre_df':           pre,
            'beta':             beta
        }
    except Exception:
        return None

# ── 로젠바움 감도 분석 (단순화) ────────────────────────────────

def rosenbaum_bounds(treated_log, control_log, gammas=None):
    """
    Rosenbaum sensitivity analysis (sign-test approximation).
    For a reverse effect (treated < control), computes the maximum
    two-sided p-value achievable under Gamma-level hidden bias.

    Interpretation: Gamma_crit is the smallest Gamma at which a hidden
    confounder could overturn the result. Gamma=1.0 means no hidden bias.
    Gamma=1.5 means confounder raises treatment odds by 50%.
    """
    if gammas is None:
        gammas = [1.0, 1.05, 1.10, 1.25, 1.5, 2.0, 3.0]

    n = len(treated_log)
    diff = treated_log - control_log
    T_plus = int(np.sum(diff > 0))

    results = []
    for gamma in gammas:
        # E_min: minimum expected T+ (bias underestimates treatment)
        e_min = n / (1 + gamma)
        var   = n * gamma / (1 + gamma) ** 2
        z_left = (T_plus - e_min) / np.sqrt(var) if var > 0 else 0
        # Upper bound on two-sided p-value (most conservative for reverse effect)
        p_upper = min(2 * stats.norm.cdf(z_left), 1.0)
        results.append({'Gamma': gamma, 'p_upper': round(p_upper, 4),
                        'T_plus': T_plus, 'n': n,
                        'significant': p_upper < 0.05})
    return results

# ── 메인 분석 ────────────────────────────────────────────────

def run_effect_size_analysis():
    print("=" * 60)
    print("SCI-level 통계 보고서")
    print("=" * 60)

    df = prepare_data()
    df_reg = df[df['Is_Shorts'] == 0].copy()

    fear_lv   = df_reg[df_reg['Has_Fear']==1]['Log_Views']
    nofear_lv = df_reg[df_reg['Has_Fear']==0]['Log_Views']
    fear_vpd   = df_reg[df_reg['Has_Fear']==1]['Log_VPD']
    nofear_vpd = df_reg[df_reg['Has_Fear']==0]['Log_VPD']

    # ── 1) 효과크기 테이블 ──────────────────────────────────
    print("\n[1] 효과크기 (Effect Sizes)")
    comparisons = [
        ("단순 비교 (Log_Views)",  fear_lv,  nofear_lv),
        ("나이 보정 (Log_VPD)",    fear_vpd, nofear_vpd),
    ]

    effect_rows = []
    for name, x, y in comparisons:
        _, p_mw = stats.mannwhitneyu(x, y, alternative='two-sided')
        g   = cohens_d(x.values, y.values)
        r   = rank_biserial_r(x.values, y.values)
        pwr = mw_power(len(x), len(y), r)
        interp = interpret_d(g)
        print(f"  {name}: g={g:+.4f} ({interp}), r={r:+.4f}, power={pwr:.3f}, p={p_mw:.4f}")
        effect_rows.append({'비교': name, "Hedge's g": round(g, 4),
                            'rank-r': round(r, 4), '검정력': round(pwr, 3),
                            'p값': round(p_mw, 4), '해석': interp})

    # Type × Fear 효과크기
    for t in ['General', 'Medical Pro']:
        sub = df_reg[df_reg['Type'] == t]
        f_  = sub[sub['Has_Fear']==1]['Log_Views']
        nf_ = sub[sub['Has_Fear']==0]['Log_Views']
        if len(f_) >= 5 and len(nf_) >= 5:
            _, p_mw = stats.mannwhitneyu(f_, nf_, alternative='two-sided')
            g   = cohens_d(f_.values, nf_.values)
            r   = rank_biserial_r(f_.values, nf_.values)
            pwr = mw_power(len(f_), len(nf_), r)
            print(f"  {t}: g={g:+.4f} ({interpret_d(g)}), r={r:+.4f}, power={pwr:.3f}, p={p_mw:.4f}")
            effect_rows.append({'비교': f'{t} (Type×Fear)', "Hedge's g": round(g, 4),
                                'rank-r': round(r, 4), '검정력': round(pwr, 3),
                                'p값': round(p_mw, 4), '해석': interpret_d(g)})
    effects_df = pd.DataFrame(effect_rows)

    # ── 2) PSM SMD 균형 진단 ─────────────────────────────────
    print("\n[2] PSM 공변량 균형 (SMD)")
    df_psm, t_idx, c_idx, covariates = run_psm_for_smd(df)
    smd_rows = []
    for cov in covariates:
        pre_t  = df_psm.loc[df_psm['Has_Fear']==1, cov].dropna()
        pre_c  = df_psm.loc[df_psm['Has_Fear']==0, cov].dropna()
        post_t = df_psm.loc[t_idx, cov].dropna()
        post_c = df_psm.loc[c_idx, cov].dropna()
        smd_pre  = smd(pre_t.values,  pre_c.values)
        smd_post = smd(post_t.values, post_c.values)
        balanced = abs(smd_post) < 0.1
        print(f"  {cov}: SMD {smd_pre:+.4f} → {smd_post:+.4f} ({'✓ 균형' if balanced else '✗ 불균형'})")
        smd_rows.append({'공변량': cov, 'SMD(매칭전)': round(smd_pre, 4),
                         'SMD(매칭후)': round(smd_post, 4),
                         '균형(|SMD|<0.1)': '✓' if balanced else '✗'})
    smd_df = pd.DataFrame(smd_rows)

    # ── 3) 평행추세 검정 ─────────────────────────────────────
    print("\n[3] 평행추세 검정 (DiD 가정)")
    pt = parallel_trends_test(df)
    if pt:
        sig = pt['p_value'] < 0.05
        print(f"  상호작용항 계수: {pt['coef_interaction']:+.6f}")
        print(f"  t={pt['t_stat']:+.4f}, p={pt['p_value']:.4f}")
        print(f"  → {'⚠ 평행추세 기각 (DiD 가정 위반)' if sig else '✓ 평행추세 충족 (DiD 가정 성립)'}")
        print(f"  n_pre-COVID = {pt['n_pre']}개")

    # ── 4) 로젠바움 감도 분석 ─────────────────────────────────
    print("\n[4] 로젠바움 감도 분석 (PSM 강건성)")
    t_lv = df_psm.loc[t_idx, 'Log_Views'].values
    c_lv = df_psm.loc[c_idx, 'Log_Views'].values
    rosenbauml = rosenbaum_bounds(t_lv, c_lv)
    for rb in rosenbauml:
        print(f"  Γ={rb['Gamma']:.2f}: p_upper={rb['p_upper']:.4f} "
              f"{'(여전히 유의)' if rb['significant'] else '← 여기서 결과 뒤집힘'}")

    # ── 시각화 ───────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # [A] 효과크기 비교 (Hedge's g)
    ax_a = fig.add_subplot(gs[0, :2])
    colors_e = ['#e94560' if abs(g) >= 0.2 else '#888888' for g in effects_df["Hedge's g"]]
    bars = ax_a.barh(effects_df['비교'], effects_df["Hedge's g"],
                     color=colors_e, alpha=0.85, edgecolor='white')
    ax_a.axvline(0, color='white', lw=1.5, linestyle='--')
    ax_a.axvline(0.2, color='#27ae60', lw=1, linestyle=':', alpha=0.7)
    ax_a.axvline(-0.2, color='#27ae60', lw=1, linestyle=':', alpha=0.7)
    for bar, (_, row) in zip(bars, effects_df.iterrows()):
        x = bar.get_width()
        g_val = row["Hedge's g"]
        r_val = row['rank-r']
        pw_val = row['검정력']
        label_txt = f"g={g_val:+.3f} | r={r_val:+.3f} | power={pw_val:.2f}"
        ax_a.text(x + 0.005 * np.sign(x) if x != 0 else 0.005,
                  bar.get_y() + bar.get_height()/2,
                  label_txt, va='center', fontsize=8, color='white')
    ax_a.set_xlabel("Hedge's g (효과크기)", fontsize=11)
    ax_a.set_title("효과크기 요약 — Hedge's g\n(녹색 점선: |g|=0.2 임계값 / 빨강: 유의 효과)", fontsize=12, fontweight='bold')
    ax_a.set_facecolor('#1a1a2e')
    ax_a.tick_params(colors='white')
    ax_a.xaxis.label.set_color('white')
    ax_a.title.set_color('white')

    # [B] 검정력 표
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.axis('off')
    tbl_data = [[row['비교'][:20], f"{row['검정력']:.3f}", f"{row['p값']:.4f}"]
                for _, row in effects_df.iterrows()]
    tbl = ax_b.table(cellText=tbl_data,
                     colLabels=['비교', '검정력', 'p값'],
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a3e' if r > 0 else '#0f3460')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333366')
    ax_b.set_title('검정력 요약', fontsize=10, fontweight='bold', color='white')

    # [C] SMD 러브 플롯 (Love Plot)
    ax_c = fig.add_subplot(gs[1, :2])
    y_pos = range(len(smd_df))
    ax_c.scatter(smd_df['SMD(매칭전)'],  list(y_pos), color='#e94560', s=80,
                 label='매칭 전', zorder=5, marker='D')
    ax_c.scatter(smd_df['SMD(매칭후)'], list(y_pos), color='#27ae60', s=80,
                 label='매칭 후', zorder=5, marker='o')
    for i, (pre, post) in enumerate(zip(smd_df['SMD(매칭전)'], smd_df['SMD(매칭후)'])):
        ax_c.plot([pre, post], [i, i], color='gray', lw=1, alpha=0.5)
    ax_c.axvline(0,    color='white', lw=1.5, linestyle='--')
    ax_c.axvline(0.1,  color='#f5a623', lw=1, linestyle=':', alpha=0.8, label='±0.1 임계값')
    ax_c.axvline(-0.1, color='#f5a623', lw=1, linestyle=':', alpha=0.8)
    ax_c.set_yticks(list(y_pos))
    ax_c.set_yticklabels(smd_df['공변량'], fontsize=10, color='white')
    ax_c.set_xlabel('표준화 평균 차이 (SMD)', fontsize=11)
    ax_c.set_title("PSM Love Plot — 공변량 균형 진단\n(|SMD| < 0.1 → 균형 달성)", fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.set_facecolor('#1a1a2e')
    ax_c.tick_params(colors='white')
    ax_c.xaxis.label.set_color('white')
    ax_c.title.set_color('white')

    # [D] SMD 테이블
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.axis('off')
    tbl2_data = [[row['공변량'], f"{row['SMD(매칭전)']:+.3f}",
                  f"{row['SMD(매칭후)']:+.3f}", row['균형(|SMD|<0.1)']]
                 for _, row in smd_df.iterrows()]
    tbl2 = ax_d.table(cellText=tbl2_data,
                      colLabels=['공변량', '전', '후', '균형'],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    for (r, c), cell in tbl2.get_celld().items():
        val = cell.get_text().get_text()
        bg = '#1a3a1a' if val == '✓' else ('#3a1a1a' if val == '✗' else '#1a1a3e' if r > 0 else '#0f3460')
        cell.set_facecolor(bg)
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333366')
    ax_d.set_title('SMD 균형 테이블', fontsize=10, fontweight='bold', color='white')

    # [E] 평행추세 시각화
    ax_e = fig.add_subplot(gs[2, :2])
    if pt:
        pre_df = pt['pre_df']
        for hf, label, col in [(0,'공포 미포함','#4a90d9'), (1,'공포 포함','#e94560')]:
            sub = pre_df[pre_df['Has_Fear']==hf].groupby('Year')['Log_VPD'].mean().reset_index()
            ax_e.plot(sub['Year'], sub['Log_VPD'], marker='o', lw=2.5,
                      color=col, label=label, markersize=8)
            # 회귀선
            if len(sub) >= 2:
                z = np.polyfit(sub['Year'], sub['Log_VPD'], 1)
                p_ = np.poly1d(z)
                ax_e.plot(sub['Year'], p_(sub['Year']), '--', color=col, lw=1.2, alpha=0.6)
        ax_e.set_xlabel('연도 (Pre-COVID: 2015–2019)', fontsize=11)
        ax_e.set_ylabel('평균 log(Views/Day)', fontsize=11)
        sig_label = '⚠ 기각' if pt['p_value'] < 0.05 else '✓ 충족'
        ax_e.set_title(f"DiD 평행추세 가정 검정\n"
                       f"상호작용 계수={pt['coef_interaction']:+.5f}, "
                       f"t={pt['t_stat']:+.3f}, p={pt['p_value']:.4f} → {sig_label}",
                       fontsize=12, fontweight='bold')
        ax_e.legend(fontsize=10)
    ax_e.set_facecolor('#1a1a2e')
    ax_e.tick_params(colors='white')
    ax_e.xaxis.label.set_color('white')
    ax_e.yaxis.label.set_color('white')
    ax_e.title.set_color('white')

    # [F] 로젠바움 감도
    ax_f = fig.add_subplot(gs[2, 2])
    gammas_rb  = [rb['Gamma'] for rb in rosenbauml]
    pvals_rb   = [rb['p_upper'] for rb in rosenbauml]
    bar_cols   = ['#27ae60' if rb['significant'] else '#e94560' for rb in rosenbauml]
    ax_f.bar(gammas_rb, pvals_rb, color=bar_cols, alpha=0.85,
             width=0.18, edgecolor='white')
    ax_f.axhline(0.05, color='#f5a623', lw=2, linestyle='--', label='α=0.05')
    ax_f.set_xlabel('Γ (숨겨진 편향 크기)', fontsize=10)
    ax_f.set_ylabel('p값 상한 (Upper bound)', fontsize=10)
    ax_f.set_title("로젠바움 감도 분석\n(초록=여전히 유의, 빨강=결과 뒤집힘)",
                   fontsize=10, fontweight='bold')
    ax_f.legend(fontsize=9)
    ax_f.set_facecolor('#1a1a2e')
    ax_f.tick_params(colors='white')
    ax_f.xaxis.label.set_color('white')
    ax_f.yaxis.label.set_color('white')
    ax_f.title.set_color('white')

    fig.patch.set_facecolor('#0a0a1a')
    plt.suptitle("SCI-Level Statistical Reporting\nEffect Size · PSM Balance (Love Plot) · Parallel Trends · Rosenbaum Bounds",
                 fontsize=15, fontweight='bold', y=1.01, color='white')

    out_path = os.path.join(SAVE_DIR, 'v2_13_sci_statistical_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"\n저장: {out_path}")

    # CSV 저장
    effects_df.to_csv(os.path.join(SAVE_DIR, 'effect_sizes.csv'), index=False, encoding='utf-8-sig')
    smd_df.to_csv(os.path.join(SAVE_DIR, 'psm_smd_balance.csv'), index=False, encoding='utf-8-sig')
    print("효과크기 CSV, SMD 균형 CSV 저장 완료")

    return effects_df, smd_df, pt, rosenbauml

if __name__ == '__main__':
    run_effect_size_analysis()
