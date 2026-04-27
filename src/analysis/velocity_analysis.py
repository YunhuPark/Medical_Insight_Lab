"""
Phase 2-A: 바이럴 속도 분석
- 영상 나이(업로드 후 경과일) 보정
- Views/Day 기준 재분석 (나이 편향 제거)
- Shorts 분리
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from datetime import datetime, timezone

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')
PALETTE   = {'primary': '#e94560', 'secondary': '#0f3460', 'accent': '#f5a623'}

FEAR_KEYWORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

def load_and_enrich(path):
    df = pd.read_csv(path)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)

    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Age_Days']     = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['Views_Per_Day'] = df['Views'] / df['Age_Days']
    df['Log_VPD']      = np.log1p(df['Views_Per_Day'])

    df['Fear_Score'] = df['Title'].apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in str(x)))
    df['Has_Fear']   = (df['Fear_Score'] > 0).astype(int)

    # Shorts 판별
    title = df['Title'].astype(str)
    desc = df['Description'].fillna('').astype(str) if 'Description' in df.columns else pd.Series([''] * len(df), index=df.index)
    df['Is_Shorts'] = (
        title.str.contains('#short|#shorts', case=False) |
        desc.str.contains('#short|#shorts', case=False) |
        (df['Duration_Sec'] <= 60)
    ).astype(int)

    return df

def run_velocity_analysis():
    df = load_and_enrich(DATA_PATH)
    print(f"Shorts 영상: {df['Is_Shorts'].sum()}개 ({df['Is_Shorts'].mean()*100:.1f}%)")
    print(f"일반 영상: {(df['Is_Shorts']==0).sum()}개")

    df_reg = df[df['Is_Shorts'] == 0].copy()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('바이럴 속도(Views/Day) 분석 — 영상 나이 편향 보정', fontsize=16, fontweight='bold', y=1.01)

    # [1] 조회수 vs 영상 나이
    ax = axes[0, 0]
    sc = ax.scatter(df_reg['Age_Days'], np.log1p(df_reg['Views']),
                    c=df_reg['Has_Fear'], cmap='RdBu_r', alpha=0.4, s=15)
    ax.set_xlabel('영상 나이 (일)')
    ax.set_ylabel('log(조회수)')
    ax.set_title('영상 나이 vs 조회수\n(빨강=공포키워드)')
    r, p = stats.spearmanr(df_reg['Age_Days'], df_reg['Views'])
    ax.text(0.05, 0.95, f'Spearman r={r:.3f}, p={p:.4f}', transform=ax.transAxes,
            va='top', fontsize=9, color='red' if p<0.05 else 'gray')

    # [2] Views/Day 분포
    ax = axes[0, 1]
    ax.hist(df_reg['Log_VPD'], bins=60, color=PALETTE['primary'], alpha=0.8, edgecolor='white', lw=0.3)
    ax.set_title('Views/Day 분포 (로그 변환)')
    ax.set_xlabel('log(Views/Day)')

    # [3] Shorts vs Regular
    ax = axes[0, 2]
    labels = ['일반 영상', 'Shorts']
    short_g = [df[df['Is_Shorts']==0]['Views_Per_Day'], df[df['Is_Shorts']==1]['Views_Per_Day']]
    bp = ax.boxplot([np.log1p(g) for g in short_g], labels=labels, patch_artist=True,
                    medianprops=dict(color='white', lw=2.5))
    colors = [PALETTE['secondary'], PALETTE['primary']]
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    stat_s, p_s = stats.mannwhitneyu(short_g[1], short_g[0], alternative='greater')
    ax.set_title(f'Shorts vs 일반 영상 Views/Day\np={p_s:.4f}')
    ax.set_ylabel('log(Views/Day)')

    # [4] 공포 키워드: 조회수 vs Views/Day 비교
    ax = axes[1, 0]
    groups = [(df_reg[df_reg['Has_Fear']==0], '미포함'), (df_reg[df_reg['Has_Fear']==1], '포함')]
    metrics = ['Views', 'Views_Per_Day']
    ratios = []
    for m in metrics:
        g0 = groups[0][0][m].median()
        g1 = groups[1][0][m].median()
        ratios.append(g1/g0)
    bars = ax.bar(['원본 조회수\n(나이 미보정)', 'Views/Day\n(나이 보정)'], ratios,
                  color=[PALETTE['secondary'], PALETTE['primary']], alpha=0.85, edgecolor='white')
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{r:.2f}배', ha='center', va='bottom', fontweight='bold', fontsize=13)
    ax.axhline(1, color='black', linestyle='--', lw=1.5)
    ax.set_ylabel('공포키워드 포함/미포함 비율')
    ax.set_title('나이 보정 전후 공포 키워드 효과 비교')

    # [5] 키워드별 Views/Day (보정 후)
    ax = axes[1, 1]
    kw_vpd = df_reg.groupby('Keyword')['Views_Per_Day'].median().sort_values(ascending=True)
    colors_kw = plt.cm.Reds(np.linspace(0.3, 0.9, len(kw_vpd)))
    bars = ax.barh(kw_vpd.index, kw_vpd.values, color=colors_kw, edgecolor='white')
    for bar, val in zip(bars, kw_vpd.values):
        ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)
    ax.set_xlabel('중앙값 Views/Day')
    ax.set_title('카테고리별 Views/Day (나이 보정)')

    # [6] 연도별 신규 영상 VPD 트렌드
    ax = axes[1, 2]
    df_reg['Year'] = df_reg['Published_At'].dt.year
    yr_vpd = df_reg.groupby('Year')['Views_Per_Day'].median()
    ax.plot(yr_vpd.index, yr_vpd.values, marker='o', color=PALETTE['primary'], linewidth=2.5, markersize=8)
    ax.fill_between(yr_vpd.index, yr_vpd.values, alpha=0.15, color=PALETTE['primary'])
    ax.set_xlabel('업로드 연도')
    ax.set_ylabel('중앙값 Views/Day')
    ax.set_title('연도별 영상 바이럴 속도 추이\n(신규 영상 경쟁 강도 반영)')
    ax.axvspan(2020, 2021, alpha=0.1, color='red', label='COVID-19')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_01_velocity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 결과 요약 저장
    df.to_csv(os.path.join(BASE_DIR, '../enriched_base.csv'), index=False, encoding='utf-8-sig')
    print(f"enriched_base.csv 저장 완료")

    # 수정된 H1 검정 결과
    fear_vpd    = df_reg[df_reg['Has_Fear']==1]['Views_Per_Day']
    nofear_vpd  = df_reg[df_reg['Has_Fear']==0]['Views_Per_Day']
    stat, p = stats.mannwhitneyu(fear_vpd, nofear_vpd, alternative='greater')
    r_eff = 1 - (2*stat)/(len(fear_vpd)*len(nofear_vpd))
    print(f"\n[보정 후 H1 검정]")
    print(f"  공포 포함 VPD 중앙값: {fear_vpd.median():.1f}")
    print(f"  공포 미포함 VPD 중앙값: {nofear_vpd.median():.1f}")
    print(f"  배율: {fear_vpd.median()/nofear_vpd.median():.2f}배")
    print(f"  p={p:.4f}, r={r_eff:.3f} -> {'채택' if p<0.05 else '기각'}")

if __name__ == '__main__':
    run_velocity_analysis()
