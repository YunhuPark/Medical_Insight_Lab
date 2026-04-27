"""
Phase 2-F: COVID-19 자연실험 분석
2020년 이전 vs 이후 의료 콘텐츠 소비 패턴 변화
Difference-in-Differences 구조로 접근
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

PANDEMIC_KEYWORDS = ['코로나','바이러스','감염','면역','폐렴','백신','격리','방역']

def run_covid_analysis():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Year']  = df['Published_At'].dt.year
    df['Month'] = df['Published_At'].dt.month
    df['YM']    = df['Published_At'].dt.to_period('Q')  # 분기별

    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['VPD']      = df['Views'] / df['Age_Days']
    df['Log_VPD']  = np.log1p(df['VPD'])
    df['Log_Views']= np.log1p(df['Views'])

    title = df['Title'].astype(str)
    df['Fear_Score'] = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Has_Fear']   = (df['Fear_Score'] > 0).astype(int)
    df['Post_COVID'] = (df['Year'] >= 2020).astype(int)
    df['Has_Pandemic_KW'] = title.apply(lambda x: any(kw in x for kw in PANDEMIC_KEYWORDS)).astype(int)

    # 2015-2025 분석
    df_main = df[(df['Year'] >= 2015) & (df['Year'] <= 2025)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('COVID-19 자연실험: 의료 콘텐츠 소비 패턴 변화', fontsize=16, fontweight='bold', y=1.01)

    # [1] 연도별 업로드 수
    yr_count = df_main.groupby('Year').size()
    axes[0,0].bar(yr_count.index, yr_count.values, color=['#e94560' if y>=2020 else '#0f3460' for y in yr_count.index],
                  alpha=0.85, edgecolor='white')
    axes[0,0].axvline(2019.5, color='red', lw=2, linestyle='--', label='COVID 시작')
    axes[0,0].set_title('연도별 의료 유튜브 업로드 수', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('연도')
    axes[0,0].set_ylabel('영상 수')
    axes[0,0].legend()

    # [2] 연도별 중앙 Views/Day (바이럴 속도)
    yr_vpd = df_main.groupby('Year')['VPD'].median()
    axes[0,1].plot(yr_vpd.index, yr_vpd.values, marker='o', linewidth=2.5,
                   color='#e94560', markersize=8)
    axes[0,1].fill_between(yr_vpd.index, yr_vpd.values, alpha=0.15, color='#e94560')
    axes[0,1].axvspan(2019.5, 2021.5, alpha=0.12, color='orange', label='팬데믹 기간')
    axes[0,1].set_title('연도별 중앙값 Views/Day\n(바이럴 속도 — 나이 보정)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('연도')
    axes[0,1].set_ylabel('중앙값 VPD')
    axes[0,1].legend()

    # [3] 공포 키워드 비율 변화
    yr_fear = df_main.groupby('Year')['Has_Fear'].mean() * 100
    axes[0,2].bar(yr_fear.index, yr_fear.values,
                  color=['#e94560' if y>=2020 else '#aaaaaa' for y in yr_fear.index],
                  alpha=0.85, edgecolor='white')
    axes[0,2].axvline(2019.5, color='red', lw=2, linestyle='--')
    axes[0,2].set_title('연도별 공포 키워드 포함 영상 비율(%)', fontsize=12, fontweight='bold')
    axes[0,2].set_xlabel('연도')
    axes[0,2].set_ylabel('비율 (%)')

    # [4] DiD 분석: COVID 전후 × 공포키워드 교호작용
    # 집단: Pre/Post × Fear/NoFear
    groups_did = {
        'Pre × 미포함':  df_main[(df_main['Post_COVID']==0) & (df_main['Has_Fear']==0)]['Log_VPD'],
        'Pre × 포함':    df_main[(df_main['Post_COVID']==0) & (df_main['Has_Fear']==1)]['Log_VPD'],
        'Post × 미포함': df_main[(df_main['Post_COVID']==1) & (df_main['Has_Fear']==0)]['Log_VPD'],
        'Post × 포함':   df_main[(df_main['Post_COVID']==1) & (df_main['Has_Fear']==1)]['Log_VPD'],
    }
    means = {k: v.mean() for k, v in groups_did.items()}

    # DiD 추정량
    did = (means['Post × 포함'] - means['Post × 미포함']) - \
          (means['Pre × 포함']  - means['Pre × 미포함'])
    print(f"\nDiD 추정량 (log scale): {did:.4f}")
    print(f"해석: COVID 후 공포키워드의 추가 효과 = {(np.exp(did)-1)*100:+.1f}%")

    # Bootstrap 95% CI for DiD
    rng = np.random.default_rng(42)
    boot_dids = []
    pre_f  = groups_did['Pre × 포함'].values
    pre_nf = groups_did['Pre × 미포함'].values
    post_f  = groups_did['Post × 포함'].values
    post_nf = groups_did['Post × 미포함'].values
    for _ in range(2000):
        d = (rng.choice(post_f,  len(post_f),  True).mean() -
             rng.choice(post_nf, len(post_nf), True).mean()) - \
            (rng.choice(pre_f,   len(pre_f),   True).mean() -
             rng.choice(pre_nf,  len(pre_nf),  True).mean())
        boot_dids.append(d)
    did_ci = np.percentile(boot_dids, [2.5, 97.5])
    did_pct_lo = (np.exp(did_ci[0])-1)*100
    did_pct_hi = (np.exp(did_ci[1])-1)*100
    print(f"Bootstrap 95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
    print(f"  → 효과 크기 95% CI: [{did_pct_lo:+.1f}%, {did_pct_hi:+.1f}%]")

    x_did = [0, 1]
    axes[1,0].plot(x_did, [means['Pre × 미포함'], means['Post × 미포함']],
                   'o--', color='steelblue', lw=2, markersize=8, label='공포 미포함')
    axes[1,0].plot(x_did, [means['Pre × 포함'], means['Post × 포함']],
                   'o-', color='#e94560', lw=2.5, markersize=8, label='공포 포함')
    axes[1,0].set_xticks([0,1])
    axes[1,0].set_xticklabels(['Pre-COVID\n(~2019)', 'Post-COVID\n(2020~)'])
    axes[1,0].set_ylabel('평균 log(Views/Day)')
    axes[1,0].set_title(f'Difference-in-Differences\nDiD={did:+.4f} ({(np.exp(did)-1)*100:+.1f}%)\nBootstrap 95% CI: [{did_pct_lo:+.1f}%, {did_pct_hi:+.1f}%]',
                         fontsize=12, fontweight='bold')
    axes[1,0].legend()

    # [5] 카테고리별 COVID 전후 변화
    kw_pre  = df_main[df_main['Post_COVID']==0].groupby('Keyword')['VPD'].median()
    kw_post = df_main[df_main['Post_COVID']==1].groupby('Keyword')['VPD'].median()
    kw_change = ((kw_post - kw_pre) / (kw_pre + 1) * 100).dropna().sort_values()
    colors_ch = ['#e94560' if v > 0 else '#0f3460' for v in kw_change.values]
    axes[1,1].barh(kw_change.index, kw_change.values, color=colors_ch, alpha=0.85, edgecolor='white')
    axes[1,1].axvline(0, color='black', lw=1.5, linestyle='--')
    axes[1,1].set_xlabel('Views/Day 변화율 (%)')
    axes[1,1].set_title('카테고리별 COVID 전후\nViews/Day 변화율', fontsize=12, fontweight='bold')

    # [6] 분기별 트렌드 (2018~2025)
    df_trend = df[(df['Year'] >= 2018) & (df['Year'] <= 2025)].copy()
    df_trend['Quarter'] = df_trend['Published_At'].dt.to_period('Q').astype(str)
    qt_trend = df_trend.groupby(['Quarter', 'Has_Fear'])['VPD'].median().reset_index()

    for fear_val, label, col in [(0, '공포 미포함', 'steelblue'), (1, '공포 포함', '#e94560')]:
        sub = qt_trend[qt_trend['Has_Fear']==fear_val]
        axes[1,2].plot(range(len(sub)), sub['VPD'], label=label, color=col, lw=2.5, marker='o', markersize=4)

    n_qt = len(qt_trend['Quarter'].unique())
    axes[1,2].set_xticks(range(0, n_qt, 4))
    axes[1,2].set_xticklabels(qt_trend['Quarter'].unique()[::4], rotation=45, fontsize=8)
    axes[1,2].axvspan(qt_trend['Quarter'].unique().tolist().index('2020Q1') if '2020Q1' in qt_trend['Quarter'].unique() else 0,
                       min(qt_trend['Quarter'].unique().tolist().index('2022Q1') if '2022Q1' in qt_trend['Quarter'].unique() else n_qt, n_qt),
                       alpha=0.1, color='orange', label='팬데믹')
    axes[1,2].set_ylabel('중앙값 VPD')
    axes[1,2].set_title('분기별 Views/Day 추이\n(공포 vs 비공포)', fontsize=12, fontweight='bold')
    axes[1,2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_05_covid_experiment.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 검정: COVID 전후 공포키워드 조회수 차이
    pre_fear  = df_main[(df_main['Post_COVID']==0)&(df_main['Has_Fear']==1)]['VPD']
    post_fear = df_main[(df_main['Post_COVID']==1)&(df_main['Has_Fear']==1)]['VPD']
    stat, p_cv = stats.mannwhitneyu(post_fear, pre_fear, alternative='greater')
    print(f"COVID 후 공포영상 VPD 증가: p={p_cv:.4f} → {'유의' if p_cv<0.05 else '비유의'}")

if __name__ == '__main__':
    run_covid_analysis()
