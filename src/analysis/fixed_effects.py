"""
Phase 2-C: 채널 고정효과 회귀 (Within-Channel Analysis)
채널 규모 영향을 통제하여 순수 제목 효과 측정
→ "공포 키워드는 채널과 무관하게 조회수를 높이는가?"
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime, timezone
from scipy import stats

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
SOLUTION_KEYWORDS = ['방법','치료','완치','개선','해결','예방','관리','치유','회복','낫는','좋아지','극복']

def run_fixed_effects():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)

    title = df['Title'].astype(str)
    df['Fear_Score']     = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Solution_Score'] = title.apply(lambda x: sum(1 for kw in SOLUTION_KEYWORDS if kw in x))
    df['Has_Fear']       = (df['Fear_Score'] > 0).astype(int)
    df['Has_Number']     = title.str.contains(r'\d').astype(int)
    df['Has_Question']   = title.str.contains(r'\?|？').astype(int)
    df['Title_Length']   = title.str.len()
    df['Is_Long']        = (df['Duration_Sec'] > 600).astype(int)
    df['Log_Views']      = np.log1p(df['Views'])
    df['Log_Age']        = np.log1p(df['Age_Days'])
    df['Is_Shorts']      = (df['Duration_Sec'] <= 60).astype(int)

    # 영상 5개 이상 채널만 (고정효과 의미 있도록)
    ch_counts = df['Channel'].value_counts()
    valid_channels = ch_counts[ch_counts >= 5].index
    df_fe = df[df['Channel'].isin(valid_channels)].copy()
    print(f"고정효과 분석 대상: {len(df_fe)}개 영상 / {df_fe['Channel'].nunique()}개 채널")

    # ── Model 1: 기본 OLS (채널 미통제)
    feat = ['Fear_Score', 'Solution_Score', 'Title_Length', 'Has_Number',
            'Has_Question', 'Medical_Score', 'Is_Long', 'Log_Age']
    X1 = sm.add_constant(df_fe[feat].astype(float))
    m1 = sm.OLS(df_fe['Log_Views'], X1).fit()

    # ── Model 2: 채널 고정효과 (채널 더미 추가)
    ch_dummies = pd.get_dummies(df_fe['Channel'], prefix='ch', drop_first=True).astype(float)
    X2_raw = pd.concat([df_fe[feat].astype(float).reset_index(drop=True),
                        ch_dummies.reset_index(drop=True)], axis=1)
    X2 = sm.add_constant(X2_raw.astype(float))
    m2 = sm.OLS(df_fe['Log_Views'].reset_index(drop=True), X2).fit()

    # ── Model 3: Views/Day 종속변수 (나이 보정)
    df_fe = df_fe.copy()
    df_fe['Log_VPD'] = np.log1p(df_fe['Views'] / df_fe['Age_Days'])
    X3 = sm.add_constant(X2_raw.astype(float))
    m3 = sm.OLS(df_fe['Log_VPD'].reset_index(drop=True), X3).fit()

    # 결과 비교
    print("\n" + "="*70)
    print(f"{'모델':<30} {'R²':>8} {'Adj.R²':>10} {'Fear_Score coef':>16} {'p':>8}")
    print("="*70)
    for label, m in [("OLS (채널 미통제)", m1),
                     ("고정효과 (채널 통제)", m2),
                     ("고정효과 + VPD 종속변수", m3)]:
        coef = m.params.get('Fear_Score', np.nan)
        pval = m.pvalues.get('Fear_Score', np.nan)
        pct  = (np.exp(coef)-1)*100
        sig  = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'n.s.'
        print(f"{label:<30} {m.rsquared:>8.4f} {m.rsquared_adj:>10.4f} {pct:>+13.1f}% {pval:>6.4f} {sig}")

    # 시각화: 세 모델 Fear_Score 계수 비교
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    models = {
        'OLS\n(채널 미통제)': m1,
        '채널 고정효과\n(채널 통제)': m2,
        '채널 고정효과\n+VPD 종속변수': m3
    }
    feat_to_show = ['Fear_Score', 'Solution_Score', 'Title_Length',
                    'Has_Number', 'Has_Question', 'Is_Long']
    feat_kor = {'Fear_Score':'공포 점수','Solution_Score':'해결 점수',
                'Title_Length':'제목 길이','Has_Number':'숫자 포함',
                'Has_Question':'의문문','Is_Long':'장편 영상'}

    x = np.arange(len(feat_to_show))
    width = 0.28
    colors_m = ['#aaaaaa', '#e94560', '#0f3460']

    for i, (label, m) in enumerate(models.items()):
        coefs = [(np.exp(m.params.get(f, 0))-1)*100 for f in feat_to_show]
        pvals = [m.pvalues.get(f, 1) for f in feat_to_show]
        bars = axes[0].bar(x + i*width, coefs, width, label=label,
                           color=colors_m[i], alpha=0.85, edgecolor='white')
        for bar, pv in zip(bars, pvals):
            sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else ''
            if sig:
                axes[0].text(bar.get_x()+bar.get_width()/2,
                             bar.get_height()+(1 if bar.get_height()>=0 else -2),
                             sig, ha='center', va='bottom', fontsize=11, color='#333')

    axes[0].axhline(0, color='black', lw=1, linestyle='--')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([feat_kor.get(f,f) for f in feat_to_show], rotation=15)
    axes[0].set_ylabel('조회수 증감률 (%)')
    axes[0].set_title('채널 고정효과 통제 전후 회귀 계수 비교\n(* p<0.05, ** p<0.01, *** p<0.001)',
                       fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    # R² 비교
    r2_vals = [m.rsquared for m in models.values()]
    adj_r2_vals = [m.rsquared_adj for m in models.values()]
    x2 = np.arange(len(models))
    axes[1].bar(x2-0.2, r2_vals, 0.35, label='R²', color='#e94560', alpha=0.8)
    axes[1].bar(x2+0.2, adj_r2_vals, 0.35, label='Adj. R²', color='#0f3460', alpha=0.8)
    for i, (r2, adj) in enumerate(zip(r2_vals, adj_r2_vals)):
        axes[1].text(i-0.2, r2+0.005, f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[1].text(i+0.2, adj+0.005, f'{adj:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(list(models.keys()), fontsize=10)
    axes[1].set_ylabel('R²')
    axes[1].set_title('모델 설명력(R²) 비교', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.suptitle('채널 고정효과 분석: 채널 규모 통제 후 순수 제목 효과',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_03_fixed_effects.png'), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_fixed_effects()
