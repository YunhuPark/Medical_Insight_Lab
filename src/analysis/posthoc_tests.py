"""
Phase 2-E: Dunn's Post-hoc Test
Kruskal-Wallis 이후 22개 카테고리 쌍별 다중 비교
Bonferroni 보정 적용
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')

def run_posthoc():
    df = pd.read_csv(DATA_PATH)
    df['Log_Views'] = np.log1p(df['Views'])

    # 1. Kruskal-Wallis 확인
    groups = [df[df['Keyword']==kw]['Views'].values for kw in df['Keyword'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis: H={stat:.2f}, p={p:.2e} → 다중비교 필요")

    # 2. Dunn's Test (Bonferroni 보정)
    dunn = sp.posthoc_dunn(df, val_col='Log_Views', group_col='Keyword', p_adjust='bonferroni')

    # 3. 히트맵 시각화
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # 유의성 마스크 (p < 0.05)
    sig_mask = dunn >= 0.05
    annot_matrix = dunn.copy().applymap(lambda x: f'{x:.3f}' if x < 0.05 else '')

    sns.heatmap(dunn, mask=sig_mask, cmap='Reds_r', ax=axes[0],
                vmin=0, vmax=0.05, annot=False, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'p-value (Bonferroni)'})
    sns.heatmap(dunn, mask=~sig_mask, cmap='Blues', ax=axes[0],
                vmin=0.05, vmax=1, annot=False, alpha=0.2, linewidths=0.3, linecolor='white',
                cbar=False)
    axes[0].set_title("Dunn's Post-hoc Test (p < 0.05 = 빨강, 유의한 차이)\nBonferroni 보정 적용",
                       fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # 카테고리별 중앙값 + 신뢰구간
    kw_stats = df.groupby('Keyword').agg(
        median=('Views', 'median'),
        q25=('Views', lambda x: x.quantile(0.25)),
        q75=('Views', lambda x: x.quantile(0.75)),
        n=('Views', 'count')
    ).reset_index().sort_values('median', ascending=True)

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(kw_stats)))
    bars = axes[1].barh(kw_stats['Keyword'], kw_stats['median']/1e4, color=colors,
                        edgecolor='white', alpha=0.85)
    axes[1].errorbar(kw_stats['median']/1e4, kw_stats['Keyword'],
                     xerr=[(kw_stats['median']-kw_stats['q25'])/1e4,
                           (kw_stats['q75']-kw_stats['median'])/1e4],
                     fmt='none', color='#333', capsize=3, lw=1.5)
    for bar, (_, row) in zip(bars, kw_stats.iterrows()):
        axes[1].text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                     f'{row["median"]/1e4:.1f}만 (n={row["n"]})',
                     va='center', fontsize=9)
    axes[1].set_xlabel('중앙값 조회수 (만회) ± IQR')
    axes[1].set_title('카테고리별 중앙값 조회수\n(오차막대=IQR, 색=순위)', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, kw_stats['median'].max()/1e4 * 1.4)

    plt.suptitle("Dunn's Post-hoc Test: 22개 질환 카테고리 쌍별 통계 비교",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_02_posthoc_dunn.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 유의한 쌍 요약
    n_sig = (dunn < 0.05).sum().sum() // 2
    total = len(dunn) * (len(dunn)-1) // 2
    print(f"\n유의한 쌍: {n_sig}/{total} ({n_sig/total*100:.1f}%)")

    # 가장 극단적 차이 쌍 출력
    pairs = []
    kws = list(dunn.columns)
    for i in range(len(kws)):
        for j in range(i+1, len(kws)):
            pairs.append((kws[i], kws[j], dunn.iloc[i, j]))
    pairs.sort(key=lambda x: x[2])
    print("\n가장 유의한 차이 TOP 5:")
    for a, b, p_val in pairs[:5]:
        ma = df[df['Keyword']==a]['Views'].median()
        mb = df[df['Keyword']==b]['Views'].median()
        print(f"  {a} vs {b}: p={p_val:.4f} | {ma/1e4:.1f}만 vs {mb/1e4:.1f}만")

if __name__ == '__main__':
    run_posthoc()
