"""
Phase 2-D: 성향점수매칭(PSM) — 인과 추정
처치: Has_Fear (공포 키워드 포함 여부)
공변량: 채널 중앙 조회수, 영상 나이, 제목 길이, 카테고리
→ 채널 규모가 비슷한 그룹끼리만 비교하여 인과 효과 추정
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
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

def run_psm():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)

    title = df['Title'].astype(str)
    df['Fear_Score'] = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Has_Fear']   = (df['Fear_Score'] > 0).astype(int)
    df['Title_Length'] = title.str.len()
    df['Has_Number']   = title.str.contains(r'\d').astype(int)
    df['Log_Views']    = np.log1p(df['Views'])

    # 채널 규모 (로그 중앙 조회수로 proxy)
    ch_med = df.groupby('Channel')['Views'].median()
    df['Ch_Med_Views'] = df['Channel'].map(ch_med)
    df['Log_Ch_Med']   = np.log1p(df['Ch_Med_Views'])

    # 카테고리 더미
    kw_dummies = pd.get_dummies(df['Keyword'], prefix='kw', drop_first=True)

    # PSM 공변량
    covariates = ['Log_Ch_Med', 'Log_Age', 'Title_Length', 'Has_Number', 'Medical_Score']
    df['Log_Age'] = np.log1p(df['Age_Days'])

    X_cov = pd.concat([df[covariates], kw_dummies], axis=1).astype(float)
    X_cov = X_cov.fillna(0)
    T = df['Has_Fear']

    # 성향 점수 추정 (로지스틱 회귀)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cov)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]
    df['Propensity_Score'] = ps

    # ATE 추정: 1:1 최근접 이웃 매칭
    treated_idx   = df[df['Has_Fear']==1].index.tolist()
    control_idx   = df[df['Has_Fear']==0].index.tolist()

    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps_control.reshape(-1,1))
    distances, matched_idx = nn.kneighbors(ps_treated.reshape(-1,1))

    matched_control_idx = [control_idx[i] for i in matched_idx.flatten()]
    caliper = 0.05 * ps.std()  # Caliper = 0.2 SD

    # Caliper 적용: 거리 초과 매칭 제거
    valid_pairs = [(t, c) for t, c, d in zip(treated_idx, matched_control_idx, distances.flatten())
                   if d <= caliper]
    print(f"PSM 매칭 쌍: {len(valid_pairs)}쌍 (전체 처치: {len(treated_idx)}개, caliper={caliper:.4f})")

    t_idx = [p[0] for p in valid_pairs]
    c_idx = [p[1] for p in valid_pairs]

    treated_views = df.loc[t_idx, 'Views']
    control_views = df.loc[c_idx, 'Views']

    # ATT (Average Treatment Effect on Treated)
    ATT_median = treated_views.median() - control_views.median()
    ATT_log    = df.loc[t_idx,'Log_Views'].mean() - df.loc[c_idx,'Log_Views'].mean()
    ATT_ratio  = treated_views.median() / control_views.median() if control_views.median() > 0 else 1
    stat, p_psm = stats.wilcoxon(df.loc[t_idx,'Log_Views'].values,
                                  df.loc[c_idx,'Log_Views'].values)

    # Bootstrap 95% CI for ATT ratio
    rng = np.random.default_rng(42)
    boot_ratios = []
    t_arr = treated_views.values
    c_arr = control_views.values
    for _ in range(2000):
        bt = rng.choice(t_arr, size=len(t_arr), replace=True)
        bc = rng.choice(c_arr, size=len(c_arr), replace=True)
        if np.median(bc) > 0:
            boot_ratios.append(np.median(bt) / np.median(bc))
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])

    print(f"\n[PSM 결과]")
    print(f"  처치군 중앙 조회수: {treated_views.median():,.0f}")
    print(f"  대조군 중앙 조회수: {control_views.median():,.0f}")
    print(f"  ATT (중앙값 차이): {ATT_median:+,.0f}회")
    print(f"  ATT (log 평균 차이): {ATT_log:+.4f}")
    print(f"  배율: {ATT_ratio:.2f}배  Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Wilcoxon p={p_psm:.4f} → {'★ 유의' if p_psm<0.05 else '비유의'}")

    # 매칭 전후 공변량 균형 확인
    pre_balance = abs(df[df['Has_Fear']==1]['Log_Ch_Med'].mean() -
                      df[df['Has_Fear']==0]['Log_Ch_Med'].mean())
    post_balance = abs(df.loc[t_idx,'Log_Ch_Med'].mean() -
                       df.loc[c_idx,'Log_Ch_Med'].mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # [1] 성향점수 분포
    axes[0].hist(ps[df['Has_Fear']==0], bins=40, alpha=0.6,
                 color='steelblue', label='공포 키워드 미포함', density=True)
    axes[0].hist(ps[df['Has_Fear']==1], bins=40, alpha=0.6,
                 color='crimson', label='공포 키워드 포함', density=True)
    axes[0].set_xlabel('성향 점수 (Propensity Score)')
    axes[0].set_ylabel('밀도')
    axes[0].set_title('성향 점수 분포\n(공통 지지 구간 확인)')
    axes[0].legend()

    # [2] 매칭 전후 채널 규모 균형
    labels = ['매칭 전\n(처치군)', '매칭 전\n(대조군)', '매칭 후\n(처치군)', '매칭 후\n(대조군)']
    data_b = [df[df['Has_Fear']==1]['Log_Ch_Med'].values,
               df[df['Has_Fear']==0]['Log_Ch_Med'].values,
               df.loc[t_idx,'Log_Ch_Med'].values,
               df.loc[c_idx,'Log_Ch_Med'].values]
    bp = axes[1].boxplot(data_b, labels=labels, patch_artist=True,
                         medianprops=dict(color='white', lw=2))
    for patch, col in zip(bp['boxes'], ['#e94560','#aaaaaa','#e94560','#aaaaaa']):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    axes[1].set_ylabel('log(채널 중앙 조회수)')
    axes[1].set_title(f'채널 규모 균형\n전:{pre_balance:.3f} → 후:{post_balance:.3f}')

    # [3] PSM 결과
    axes[2].boxplot([np.log1p(control_views), np.log1p(treated_views)],
                    labels=['대조군\n(공포X)', '처치군\n(공포O)'],
                    patch_artist=True,
                    medianprops=dict(color='white', lw=2.5),
                    boxprops=dict(alpha=0.8))
    [p.set_facecolor(c) for p, c in zip(axes[2].patches, ['steelblue','crimson'])]
    axes[2].set_ylabel('log(조회수)')
    axes[2].set_title(f'PSM 매칭 후 조회수 비교\nWilcoxon p={p_psm:.4f}\nATT={ATT_ratio:.2f}배  95%CI=[{ci_lo:.2f},{ci_hi:.2f}]')

    plt.suptitle('성향점수매칭(PSM): 채널 규모 통제 후 공포 키워드 인과 효과 추정',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_04_psm_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_psm()
