"""
구독자 수 데이터 통합 분석
채널 규모(구독자) × 공포 키워드 → 조회수 관계 정밀 분석
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

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
STATS_PATH = os.path.join(BASE_DIR, '../channel_stats.csv')
SAVE_DIR   = os.path.join(BASE_DIR, '../../results')

FEAR_KEYWORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

def run():
    df = pd.read_csv(DATA_PATH)
    ch_stats = pd.read_csv(STATS_PATH)

    # 병합
    ch_stats_valid = ch_stats[ch_stats['Subscribers'].notna()][['Channel','Subscribers','Total_Videos']]
    df = df.merge(ch_stats_valid, on='Channel', how='left')

    title = df['Title'].astype(str)
    df['Has_Fear']   = title.apply(lambda x: int(any(kw in x for kw in FEAR_KEYWORDS)))
    df['Fear_Score'] = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Log_Views']  = np.log1p(df['Views'])
    df['Log_Subs']   = np.log1p(df['Subscribers'].fillna(0))

    df_with_subs = df[df['Subscribers'].notna()].copy()
    print(f"구독자 데이터 보유 영상: {len(df_with_subs)}개")
    print(f"구독자 데이터 없는 영상: {df['Subscribers'].isna().sum()}개")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('채널 구독자 규모 × 공포 키워드 × 조회수 심층 분석', fontsize=15, fontweight='bold', y=1.01)

    if len(df_with_subs) < 10:
        print("구독자 데이터 부족 — 채널 규모 프록시(채널 내 중앙 조회수)로 대체 분석")
        ch_med = df.groupby('Channel')['Views'].median()
        df['Ch_Median'] = df['Channel'].map(ch_med)
        df['Log_Ch'] = np.log1p(df['Ch_Median'])
        df_with_subs = df.copy()
        sub_col = 'Ch_Median'
        sub_label = '채널 중앙 조회수 (구독자 프록시)'
    else:
        sub_col = 'Subscribers'
        sub_label = '구독자 수'

    # [1] 구독자 vs 조회수 산점도
    ax = axes[0, 0]
    scatter = ax.scatter(np.log1p(df_with_subs[sub_col]),
                         df_with_subs['Log_Views'],
                         c=df_with_subs['Has_Fear'], cmap='RdBu_r',
                         alpha=0.4, s=15)
    r, p = stats.spearmanr(np.log1p(df_with_subs[sub_col].fillna(0)),
                           df_with_subs['Log_Views'])
    ax.set_xlabel(f'log({sub_label})')
    ax.set_ylabel('log(조회수)')
    ax.set_title(f'채널 규모 vs 조회수\nSpearman r={r:.3f}, p={p:.4f}')
    plt.colorbar(scatter, ax=ax, label='공포키워드(1=포함)')

    # [2] 채널 규모 구간별 공포 키워드 사용 비율
    ax = axes[0, 1]
    df_with_subs['Ch_Tier'] = pd.qcut(
        np.log1p(df_with_subs[sub_col].fillna(0)),
        q=4, labels=['소형', '중소형', '중형', '대형']
    )
    tier_fear = df_with_subs.groupby('Ch_Tier')['Has_Fear'].mean() * 100
    colors_t = ['#aaaaaa', '#f5a623', '#e94560', '#8b0000']
    bars = ax.bar(tier_fear.index, tier_fear.values, color=colors_t, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, tier_fear.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('채널 규모 (4분위)')
    ax.set_ylabel('공포 키워드 포함 비율 (%)')
    ax.set_title('채널 규모별 공포 키워드 사용 패턴\n대형 채널일수록 공포 제목 많이 쓰는가?')

    # [3] 채널 규모 통제 후 공포 키워드 효과 (티어별)
    ax = axes[0, 2]
    tier_labels = ['소형', '중소형', '중형', '대형']
    fear_ratios, pvalues = [], []
    for tier in tier_labels:
        t_df = df_with_subs[df_with_subs['Ch_Tier'] == tier]
        if len(t_df) < 5:
            fear_ratios.append(1.0); pvalues.append(1.0); continue
        fear_g  = t_df[t_df['Has_Fear']==1]['Views']
        nfear_g = t_df[t_df['Has_Fear']==0]['Views']
        if len(fear_g) < 2 or len(nfear_g) < 2:
            fear_ratios.append(1.0); pvalues.append(1.0); continue
        ratio = fear_g.median() / nfear_g.median() if nfear_g.median() > 0 else 1
        _, p_t = stats.mannwhitneyu(fear_g, nfear_g, alternative='greater')
        fear_ratios.append(ratio); pvalues.append(p_t)

    bar_colors = ['#27ae60' if p < 0.05 else '#aaaaaa' for p in pvalues]
    bars = ax.bar(tier_labels, fear_ratios, color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='black', lw=1.5, linestyle='--')
    for bar, ratio, p in zip(bars, fear_ratios, pvalues):
        sig = '★유의' if p < 0.05 else 'n.s.'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{ratio:.2f}배\n{sig}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('채널 규모 티어')
    ax.set_ylabel('공포 포함 / 미포함 중앙 조회수 비율')
    ax.set_title('채널 규모 통제 후 공포 키워드 효과\n(녹색=유의, 회색=비유의)')

    # [4] 카테고리 × 채널 규모 히트맵
    ax = axes[1, 0]
    if 'Ch_Tier' in df_with_subs.columns:
        pivot = df_with_subs.groupby(['Keyword','Ch_Tier'])['Views'].median().unstack() / 1e4
        if not pivot.empty:
            sns.heatmap(pivot, ax=ax, cmap='Reds', annot=True, fmt='.0f',
                        linewidths=0.5, linecolor='white',
                        cbar_kws={'label': '중앙값 조회수 (만)'})
            ax.set_title('질환 카테고리 × 채널 규모\n중앙값 조회수 (만회)', fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
            ax.tick_params(axis='y', rotation=0)

    # [5] 구독자당 조회수 (채널 효율성)
    ax = axes[1, 1]
    if sub_col == 'Subscribers':
        df_with_subs['Views_Per_Sub'] = df_with_subs['Views'] / (df_with_subs['Subscribers'] + 1)
        eff_by_fear = df_with_subs.groupby('Has_Fear')['Views_Per_Sub'].median()
        ax.bar(['공포 미포함', '공포 포함'], eff_by_fear.values,
               color=['#0f3460','#e94560'], alpha=0.85, edgecolor='white')
        for bar, v in zip(ax.patches, eff_by_fear.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('구독자당 조회수 (효율성)')
        ax.set_title('채널 규모 표준화: 구독자당 조회수\n공포 키워드 순수 효율성')
    else:
        # 채널 중앙 조회수 대비 개별 영상 성과
        df_with_subs['Relative_Perf'] = df_with_subs['Views'] / (df_with_subs[sub_col] + 1)
        rel_by_fear = df_with_subs.groupby('Has_Fear')['Relative_Perf'].median()
        ax.bar(['공포 미포함', '공포 포함'], rel_by_fear.values,
               color=['#0f3460','#e94560'], alpha=0.85, edgecolor='white')
        for bar, v in zip(ax.patches, rel_by_fear.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('채널 중앙 대비 상대 성과')
        ax.set_title('채널 내 상대 성과 비교\n(채널 규모 통제)')

    # [6] 채널 상위 10개 조회수 집중도
    ax = axes[1, 2]
    ch_total = df.groupby('Channel')['Views'].sum().sort_values(ascending=False)
    top10 = ch_total.head(10)
    total_all = ch_total.sum()
    ax.pie([top10.sum(), total_all - top10.sum()],
           labels=[f'상위 10개 채널\n({top10.sum()/total_all*100:.1f}%)',
                   f'나머지 {len(ch_total)-10}개\n({(total_all-top10.sum())/total_all*100:.1f}%)'],
           colors=['#e94560', '#aaaaaa'], autopct='%1.1f%%',
           startangle=90, wedgeprops=dict(edgecolor='white', lw=2))
    ax.set_title('조회수 시장 집중도\n(상위 10개 채널 vs 나머지)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_10_subscriber_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n상위 10개 채널 조회수 집중도: {top10.sum()/total_all*100:.1f}%")

if __name__ == '__main__':
    run()
