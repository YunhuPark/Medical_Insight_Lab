"""
Shorts vs 일반영상 분석
- YouTube Shorts(60초 이하)는 일반 영상과 알고리즘이 다름
- Shorts 포함 여부가 Fear Effect 측정에 미치는 영향 정량화
- Shorts 전용 조회수 패턴, 카테고리별 Shorts 비율 분석
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

def run():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Year']     = df['Published_At'].dt.year
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['VPD']      = df['Views'] / df['Age_Days']

    title = df['Title'].astype(str)
    df['Has_Fear']   = title.apply(lambda x: int(any(kw in x for kw in FEAR_KEYWORDS)))
    df['Is_Shorts']  = (df['Duration_Sec'] <= 60).astype(int)
    df['Log_Views']  = np.log1p(df['Views'])
    df['Like_Rate']  = df['Likes'] / (df['Views'] + 1)

    shorts = df[df['Is_Shorts']==1]
    normal = df[df['Is_Shorts']==0]

    print(f"Shorts: {len(shorts)}개 ({len(shorts)/len(df)*100:.1f}%)")
    print(f"일반영상: {len(normal)}개 ({len(normal)/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Shorts vs 일반영상 분리 분석\n"데이터의 36.5%가 Shorts — 혼합 분석의 위험성"',
                 fontsize=15, fontweight='bold', y=1.01)

    # [1] 조회수 분포 비교
    ax = axes[0, 0]
    ax.hist(normal['Log_Views'], bins=40, alpha=0.6, color='#0f3460', label=f'일반영상 (n={len(normal)})', density=True)
    ax.hist(shorts['Log_Views'], bins=40, alpha=0.6, color='#e94560', label=f'Shorts (n={len(shorts)})', density=True)
    stat_mw, p_mw = stats.mannwhitneyu(normal['Views'], shorts['Views'], alternative='two-sided')
    ax.set_xlabel('log(조회수)')
    ax.set_ylabel('밀도')
    ax.set_title(f'조회수 분포: 일반 vs Shorts\nMann-Whitney p={p_mw:.4f}')
    ax.legend()

    # [2] 카테고리별 Shorts 비율
    ax = axes[0, 1]
    kw_shorts = df.groupby('Keyword')['Is_Shorts'].mean().sort_values(ascending=False) * 100
    colors_kw = ['#e94560' if v > 50 else '#0f3460' for v in kw_shorts.values]
    bars = ax.barh(kw_shorts.index, kw_shorts.values, color=colors_kw, alpha=0.85, edgecolor='white')
    ax.axvline(df['Is_Shorts'].mean()*100, color='black', lw=1.5, linestyle='--', label=f'평균 {df["Is_Shorts"].mean()*100:.1f}%')
    ax.set_xlabel('Shorts 비율 (%)')
    ax.set_title('카테고리별 Shorts 비율\n(빨강=50% 초과)', fontweight='bold')
    ax.legend(fontsize=9)

    # [3] 연도별 Shorts 비율 증가
    ax = axes[0, 2]
    yr_shorts = df[df['Year'].between(2019,2025)].groupby('Year')['Is_Shorts'].mean() * 100
    ax.plot(yr_shorts.index, yr_shorts.values, marker='o', color='#e94560', lw=2.5, markersize=8)
    ax.fill_between(yr_shorts.index, yr_shorts.values, alpha=0.2, color='#e94560')
    ax.set_xlabel('연도')
    ax.set_ylabel('Shorts 비율 (%)')
    ax.set_title('연도별 Shorts 비율 추이\n(YouTube Shorts 도입 이후 급증)', fontweight='bold')
    ax.grid(alpha=0.3)
    for yr, v in yr_shorts.items():
        ax.text(yr, v+1, f'{v:.0f}%', ha='center', fontsize=9)

    # [4] Fear 효과: 일반 vs Shorts 분리 비교 (핵심)
    ax = axes[1, 0]
    groups_cmp = {
        '일반\n공포X': normal[normal['Has_Fear']==0]['Log_Views'],
        '일반\n공포O': normal[normal['Has_Fear']==1]['Log_Views'],
        'Shorts\n공포X': shorts[shorts['Has_Fear']==0]['Log_Views'],
        'Shorts\n공포O': shorts[shorts['Has_Fear']==1]['Log_Views'],
    }
    bp = ax.boxplot([v.values for v in groups_cmp.values()],
                    labels=list(groups_cmp.keys()),
                    patch_artist=True, medianprops=dict(color='white', lw=2))
    box_colors = ['#0f3460','#27ae60','#888888','#e94560']
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_ylabel('log(조회수)')
    ax.set_title('Fear 효과: 일반영상 vs Shorts 분리\n(4분류 비교)', fontweight='bold')

    # p값 계산 + 표시
    _, p_normal = stats.mannwhitneyu(
        normal[normal['Has_Fear']==1]['Log_Views'],
        normal[normal['Has_Fear']==0]['Log_Views'], alternative='two-sided')
    _, p_shorts = stats.mannwhitneyu(
        shorts[shorts['Has_Fear']==1]['Log_Views'],
        shorts[shorts['Has_Fear']==0]['Log_Views'], alternative='two-sided')
    ax.text(1.5, ax.get_ylim()[1]*0.97,
            f'일반: p={p_normal:.3f}\nShorts: p={p_shorts:.3f}',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    # [5] Shorts 포함/제외 시 전체 Fear 효과 변화
    ax = axes[1, 1]
    scenarios = [
        ('전체 데이터\n(혼합)', df),
        ('일반영상만\n(Shorts 제외)', normal),
        ('Shorts만', shorts),
    ]
    ratios, labels_s, pvals = [], [], []
    for label, sub in scenarios:
        f  = sub[sub['Has_Fear']==1]['Views']
        nf = sub[sub['Has_Fear']==0]['Views']
        if len(f) < 5 or len(nf) < 5:
            continue
        ratio = f.median() / nf.median() if nf.median() > 0 else 1
        _, pv = stats.mannwhitneyu(f, nf, alternative='two-sided')
        ratios.append(ratio); labels_s.append(label); pvals.append(pv)

    colors_sc = ['#27ae60' if p < 0.05 else '#aaaaaa' for p in pvals]
    bars_sc = ax.bar(range(len(ratios)), ratios, color=colors_sc, alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='black', lw=1.5, linestyle='--')
    for i, (r, p) in enumerate(zip(ratios, pvals)):
        sig = f'p={p:.3f}\n★' if p < 0.05 else f'p={p:.3f}\nn.s.'
        ax.text(i, r + 0.03, f'{r:.2f}x\n{sig}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(labels_s)))
    ax.set_xticklabels(labels_s, fontsize=10)
    ax.set_ylabel('공포 포함 / 미포함 조회수 비율')
    ax.set_title('Shorts 포함/제외에 따른\nFear 효과 변화', fontweight='bold')

    # [6] VPD 기준 비교 (나이 보정)
    ax = axes[1, 2]
    vpd_groups = {
        '일반\n공포X': normal[normal['Has_Fear']==0]['VPD'],
        '일반\n공포O': normal[normal['Has_Fear']==1]['VPD'],
        'Shorts\n공포X': shorts[shorts['Has_Fear']==0]['VPD'],
        'Shorts\n공포O': shorts[shorts['Has_Fear']==1]['VPD'],
    }
    medians_vpd = [v.median() for v in vpd_groups.values()]
    vcolors = ['#0f3460','#27ae60','#888888','#e94560']
    bars_vpd = ax.bar(range(4), medians_vpd, color=vcolors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(vpd_groups.keys()), fontsize=10)
    ax.set_ylabel('중앙값 Views/Day (나이 보정)')
    ax.set_title('Views/Day 기준 비교\n(영상 나이 편향 제거)', fontweight='bold')
    for bar, v in zip(bars_vpd, medians_vpd):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_12_shorts_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n[Shorts 분리 분석 요약]")
    print(f"  전체 데이터 Fear 배율: {df[df['Has_Fear']==1]['Views'].median()/df[df['Has_Fear']==0]['Views'].median():.2f}x")
    print(f"  일반영상만 Fear 배율: {normal[normal['Has_Fear']==1]['Views'].median()/normal[normal['Has_Fear']==0]['Views'].median():.2f}x  p={p_normal:.4f}")
    print(f"  Shorts만 Fear 배율:   {shorts[shorts['Has_Fear']==1]['Views'].median()/shorts[shorts['Has_Fear']==0]['Views'].median():.2f}x  p={p_shorts:.4f}")
    print(f"  → Shorts 혼합 시 Fear 효과가 왜곡될 수 있음")

if __name__ == '__main__':
    run()
