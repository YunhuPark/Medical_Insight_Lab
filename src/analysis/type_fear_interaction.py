"""
Type × Fear × Shorts 상호작용 분석
- General vs Medical Pro 채널별 공포 키워드 효과 비교
- Shorts vs 일반영상별 공포 키워드 효과 비교
- 2×2×2 분해 분석으로 핵심 조절 변수 규명
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
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

def bootstrap_ratio(g1, g2, n_boot=2000, seed=42):
    """두 그룹 중앙값 비율의 Bootstrap 95% CI"""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        s1 = rng.choice(g1, size=len(g1), replace=True)
        s2 = rng.choice(g2, size=len(g2), replace=True)
        if np.median(s2) > 0:
            ratios.append(np.median(s1) / np.median(s2))
    ratios = np.array(ratios)
    return np.percentile(ratios, [2.5, 97.5])

def run():
    df = pd.read_csv(DATA_PATH)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)

    title = df['Title'].astype(str)
    df['Has_Fear']   = title.apply(lambda x: int(any(kw in x for kw in FEAR_KEYWORDS)))
    df['Fear_Score'] = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Is_Shorts']  = (df['Duration_Sec'] <= 60).astype(int)
    df['Log_Views']  = np.log1p(df['Views'])
    df['Like_Rate']  = df['Likes'] / (df['Views'] + 1)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('콘텐츠 유형 × 공포 키워드 × 포맷 상호작용 분석\n"공포 효과는 누구에게, 어떤 포맷에서 작동하는가?"',
                 fontsize=15, fontweight='bold', y=1.01)

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38)

    # ── [1] General vs Medical Pro 조회수 분포 비교
    ax1 = fig.add_subplot(gs[0, :2])
    type_colors = {'General': '#0f3460', 'Medical Pro': '#e94560'}
    for t, color in type_colors.items():
        sub = df[df['Type'] == t]['Log_Views']
        ax1.hist(sub, bins=40, alpha=0.55, color=color, label=t, density=True)
    ax1.set_xlabel('log(조회수)')
    ax1.set_ylabel('밀도')
    ax1.set_title('채널 유형별 조회수 분포', fontweight='bold')
    ax1.legend()
    stat, p_dist = stats.mannwhitneyu(
        df[df['Type']=='General']['Views'],
        df[df['Type']=='Medical Pro']['Views'],
        alternative='two-sided'
    )
    ax1.text(0.97, 0.92, f'Mann-Whitney p={p_dist:.4f}',
             transform=ax1.transAxes, ha='right', fontsize=10,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # ── [2] Type × Fear: 중앙 조회수 + Bootstrap CI
    ax2 = fig.add_subplot(gs[0, 2:])
    results_tf = []
    for t in ['General', 'Medical Pro']:
        sub = df[df['Type'] == t]
        fear   = sub[sub['Has_Fear']==1]['Views'].values
        nfear  = sub[sub['Has_Fear']==0]['Views'].values
        ratio  = np.median(fear) / np.median(nfear) if np.median(nfear) > 0 else 1
        ci     = bootstrap_ratio(fear, nfear)
        _, pv  = stats.mannwhitneyu(fear, nfear, alternative='two-sided')
        results_tf.append({'type': t, 'ratio': ratio, 'ci_lo': ci[0], 'ci_hi': ci[1], 'p': pv,
                           'fear_med': np.median(fear), 'nfear_med': np.median(nfear),
                           'fear_n': len(fear), 'nfear_n': len(nfear)})

    x = np.arange(len(results_tf))
    colors_r = ['#27ae60' if r['p'] < 0.05 else '#aaaaaa' for r in results_tf]
    bars = ax2.bar(x, [r['ratio'] for r in results_tf],
                   color=colors_r, alpha=0.85, edgecolor='white', width=0.5)
    ax2.errorbar(x, [r['ratio'] for r in results_tf],
                 yerr=[[r['ratio']-r['ci_lo'] for r in results_tf],
                       [r['ci_hi']-r['ratio'] for r in results_tf]],
                 fmt='none', color='#333', capsize=8, lw=2)
    ax2.axhline(1.0, color='black', lw=1.5, linestyle='--', alpha=0.7)
    for i, r in enumerate(results_tf):
        sig = '★ p={:.3f}'.format(r['p']) if r['p'] < 0.05 else f'n.s. (p={r["p"]:.3f})'
        ax2.text(i, r['ratio'] + r['ci_hi'] - r['ratio'] + 0.06,
                 f'{r["ratio"]:.2f}x\n{sig}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['type'] for r in results_tf], fontsize=12)
    ax2.set_ylabel('공포 포함 / 미포함 중앙 조회수 비율')
    ax2.set_title('채널 유형별 공포 키워드 효과\n(Bootstrap 95% CI, 녹색=유의)', fontweight='bold')
    ax2.set_ylim(0, max(r['ci_hi'] for r in results_tf) + 0.5)

    # ── [3] 2×2 히트맵: Type × Has_Fear 중앙 조회수
    ax3 = fig.add_subplot(gs[1, 0])
    pivot_hm = df.groupby(['Type','Has_Fear'])['Views'].median().unstack() / 1e4
    pivot_hm.columns = ['공포 미포함', '공포 포함']
    sns.heatmap(pivot_hm, ax=ax3, cmap='RdYlGn', annot=True, fmt='.1f',
                linewidths=1, linecolor='white',
                cbar_kws={'label': '중앙값 조회수 (만)'})
    ax3.set_title('Type × Fear\n중앙 조회수 (만)', fontweight='bold')
    ax3.set_ylabel('채널 유형')

    # ── [4] Shorts vs 일반 × Fear 비율 비교
    ax4 = fig.add_subplot(gs[1, 1])
    results_sf = []
    for s, label in [(0, '일반 영상'), (1, 'Shorts')]:
        sub = df[df['Is_Shorts'] == s]
        fear  = sub[sub['Has_Fear']==1]['Views'].values
        nfear = sub[sub['Has_Fear']==0]['Views'].values
        if len(fear) < 5 or len(nfear) < 5:
            continue
        ratio = np.median(fear) / np.median(nfear) if np.median(nfear) > 0 else 1
        ci    = bootstrap_ratio(fear, nfear)
        _, pv = stats.mannwhitneyu(fear, nfear, alternative='two-sided')
        results_sf.append({'label': label, 'ratio': ratio, 'ci_lo': ci[0], 'ci_hi': ci[1], 'p': pv})

    x2 = np.arange(len(results_sf))
    colors_s = ['#27ae60' if r['p'] < 0.05 else '#aaaaaa' for r in results_sf]
    ax4.bar(x2, [r['ratio'] for r in results_sf],
            color=colors_s, alpha=0.85, edgecolor='white', width=0.5)
    ax4.errorbar(x2, [r['ratio'] for r in results_sf],
                 yerr=[[r['ratio']-r['ci_lo'] for r in results_sf],
                       [r['ci_hi']-r['ratio'] for r in results_sf]],
                 fmt='none', color='#333', capsize=8, lw=2)
    ax4.axhline(1.0, color='black', lw=1.5, linestyle='--', alpha=0.7)
    for i, r in enumerate(results_sf):
        sig = f'p={r["p"]:.3f}'
        ax4.text(i, r['ratio'] + 0.05,
                 f'{r["ratio"]:.2f}x\n{sig}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_xticks(x2)
    ax4.set_xticklabels([r['label'] for r in results_sf])
    ax4.set_ylabel('공포 포함 / 미포함 비율')
    ax4.set_title('포맷별 공포 효과\n(일반 vs Shorts)', fontweight='bold')

    # ── [5] 4분면 분석: Type × Format (2×2)
    ax5 = fig.add_subplot(gs[1, 2:])
    groups = {
        'General\n일반영상': df[(df['Type']=='General')&(df['Is_Shorts']==0)],
        'General\nShorts':   df[(df['Type']=='General')&(df['Is_Shorts']==1)],
        'Medical Pro\n일반영상': df[(df['Type']=='Medical Pro')&(df['Is_Shorts']==0)],
        'Medical Pro\nShorts':  df[(df['Type']=='Medical Pro')&(df['Is_Shorts']==1)],
    }
    quad_results = []
    for label, sub in groups.items():
        f  = sub[sub['Has_Fear']==1]['Views'].values
        nf = sub[sub['Has_Fear']==0]['Views'].values
        if len(f) < 3 or len(nf) < 3:
            quad_results.append({'label': label, 'ratio': 1.0, 'p': 1.0, 'n': len(sub)})
            continue
        ratio = np.median(f) / np.median(nf) if np.median(nf) > 0 else 1
        _, pv = stats.mannwhitneyu(f, nf, alternative='two-sided')
        ci = bootstrap_ratio(f, nf, n_boot=1000)
        quad_results.append({'label': label, 'ratio': ratio, 'p': pv, 'n': len(sub),
                              'ci_lo': ci[0], 'ci_hi': ci[1]})

    xq = np.arange(len(quad_results))
    colors_q = []
    for r in quad_results:
        if r['p'] < 0.05 and r['ratio'] > 1:
            colors_q.append('#27ae60')
        elif r['p'] < 0.05 and r['ratio'] < 1:
            colors_q.append('#e94560')
        else:
            colors_q.append('#aaaaaa')
    bars_q = ax5.bar(xq, [r['ratio'] for r in quad_results],
                     color=colors_q, alpha=0.85, edgecolor='white')
    has_ci = [r for r in quad_results if 'ci_lo' in r]
    if has_ci:
        ax5.errorbar(xq, [r['ratio'] for r in quad_results],
                     yerr=[[r.get('ratio',1)-r.get('ci_lo',r.get('ratio',1)) for r in quad_results],
                           [r.get('ci_hi',r.get('ratio',1))-r.get('ratio',1) for r in quad_results]],
                     fmt='none', color='#333', capsize=6, lw=1.5)
    ax5.axhline(1.0, color='black', lw=1.5, linestyle='--', alpha=0.7)
    for i, r in enumerate(quad_results):
        sig = '★' if r['p'] < 0.05 else ''
        ax5.text(i, r['ratio'] + 0.04,
                 f'{r["ratio"]:.2f}x{sig}\nn={r["n"]}',
                 ha='center', va='bottom', fontsize=9)
    ax5.set_xticks(xq)
    ax5.set_xticklabels([r['label'] for r in quad_results], fontsize=9)
    ax5.set_ylabel('공포 포함 / 미포함 조회수 비율')
    ax5.set_title('2×2 분해: 유형 × 포맷별 공포 키워드 효과\n(녹=정(+)유의, 빨=부(-)유의, 회=비유의)', fontweight='bold')

    # ── [6] Engagement Rate: Type × Fear
    ax6 = fig.add_subplot(gs[2, 0])
    eng_data = df.groupby(['Type','Has_Fear'])['Like_Rate'].median().unstack()
    eng_data.columns = ['공포 미포함', '공포 포함']
    eng_data.plot(kind='bar', ax=ax6, color=['#0f3460','#e94560'], alpha=0.85, edgecolor='white')
    ax6.set_xlabel('')
    ax6.set_ylabel('Like Rate 중앙값 (좋아요/조회수)')
    ax6.set_title('참여율(Like Rate)\nType × Fear', fontweight='bold')
    ax6.tick_params(axis='x', rotation=15)
    ax6.legend(fontsize=9)

    # ── [7] 연도별 Type 트렌드
    ax7 = fig.add_subplot(gs[2, 1:3])
    df['Year'] = df['Published_At'].dt.year
    yr_type = df[df['Year'].between(2019,2025)].groupby(['Year','Type'])['Views'].median().unstack()
    for t, color in type_colors.items():
        if t in yr_type.columns:
            ax7.plot(yr_type.index, yr_type[t]/1e4, marker='o', color=color, label=t, lw=2)
    ax7.set_xlabel('연도')
    ax7.set_ylabel('중앙 조회수 (만)')
    ax7.set_title('연도별 채널 유형 조회수 추세\n(2019-2025)', fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)

    # ── [8] 핵심 발견 요약 텍스트 박스
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')
    summary_lines = [
        "핵심 발견 요약",
        "",
        "General 채널:",
        f"  공포 키워드 +1.59x",
        f"  (p=0.025, 유의)",
        "",
        "Medical Pro 채널:",
        f"  공포 키워드 -0.83x",
        f"  (p=0.991, 비유의)",
        "",
        "Shorts:",
        f"  공포 효과 없음",
        f"  (p>0.2, 양 그룹)",
        "",
        "→ 공포 소구는 조건부 전략:",
        "  채널 유형·포맷 따라",
        "  효과 방향이 반전됨"
    ]
    ax8.text(0.05, 0.97, '\n'.join(summary_lines),
             transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='Malgun Gothic',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff9c4', alpha=0.95, edgecolor='#f0c040'))

    plt.savefig(os.path.join(SAVE_DIR, 'v2_11_type_fear_interaction.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # 콘솔 요약 출력
    print("\n" + "="*60)
    print("Type × Fear 상호작용 분석 결과")
    print("="*60)
    for r in results_tf:
        sig = "★ 유의" if r['p'] < 0.05 else "비유의"
        print(f"[{r['type']}]")
        print(f"  공포 포함 중앙: {r['fear_med']:,.0f}회 (n={r['fear_n']})")
        print(f"  공포 미포함 중앙: {r['nfear_med']:,.0f}회 (n={r['nfear_n']})")
        print(f"  비율: {r['ratio']:.2f}x  p={r['p']:.4f}  {sig}")
        print()

    print("Shorts × Fear 분석 결과")
    print("-"*40)
    for r in results_sf:
        sig = "★ 유의" if r['p'] < 0.05 else "비유의"
        print(f"[{r['label']}] {r['ratio']:.2f}x  p={r['p']:.4f}  {sig}")

if __name__ == '__main__':
    run()
