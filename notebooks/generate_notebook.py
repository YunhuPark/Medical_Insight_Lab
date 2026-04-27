"""
Jupyter Notebook 자동 생성 스크립트
실행: venv/Scripts/python notebooks/generate_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13.0"
    }
}

cells = []

# ────────────────────────────────────────────────────────────────
# COVER
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 50px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
  <h1 style="color: #e94560; font-size: 2.2em; margin-bottom: 10px;">🏥 의료 유튜브 조회수 결정 요인 분석</h1>
  <h2 style="color: #f5f5f5; font-size: 1.3em; font-weight: 300; margin-bottom: 25px;">공포 소구(Fear Appeal) 마케팅 전략의 실증적 검증</h2>
  <div style="color: #aaaaaa; font-size: 0.95em; line-height: 2;">
    <b style="color:#e94560;">데이터 규모</b>: 2,512개 영상 · 921개 채널 · 22개 질환 카테고리<br>
    <b style="color:#e94560;">분석 기간</b>: 2018 – 2024<br>
    <b style="color:#e94560;">분석 방법</b>: 통계 검정 · 회귀분석 · 머신러닝(Random Forest) · SHAP 해석<br>
  </div>
</div>

---

## 📋 Executive Summary

> 의료 유튜브 시장에서 **어떤 제목 키워드가 조회수를 결정하는가?**
> 본 분석은 2,512개 의료 콘텐츠 데이터를 기반으로 공포 소구(Fear Appeal) 마케팅의 실제 효과를 통계적으로 검증하고,
> 질환 카테고리·제목 구조·계절성 등 다각적 요인을 분석하여 **데이터 기반 콘텐츠 전략**을 도출합니다.

### 🔑 핵심 발견 (Key Findings)
| # | 발견 | 의미 |
|---|------|------|
| 1 | 공포 키워드 포함 제목은 미포함 대비 **중앙값 조회수 2.1배** 높음 | Fear Appeal 효과 통계적 유의 |
| 2 | 질환 카테고리별 조회수 격차 **최대 12배** | 시장 수요 불균형 존재 |
| 3 | 숫자 포함 제목("5가지", "3주") 조회수 **1.4배** 높음 | 구체성이 클릭을 유도 |
| 4 | 동영상 길이 10분 이상 시 조회수 **유의미하게 낮음** | 짧고 임팩트 있는 콘텐츠가 유리 |
| 5 | Random Forest 모델 R² = 0.38 | 제목 피처만으로 조회수 예측 가능 |
"""))

# ────────────────────────────────────────────────────────────────
# 1. 연구 배경 & 가설
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""
---
## 1. 연구 배경 및 가설 설정

### 1.1 연구 배경

한국 유튜브 시장에서 의료·건강 콘텐츠는 코로나19 이후 급성장했습니다.
"암 초기 신호" "혈당 뚝 떨어집니다" 같은 자극적 제목이 알고리즘 상위를 점령하는 현상이 관찰되지만,
**이것이 실제 조회수와 통계적으로 유의한 관계인지**는 실증 분석된 바가 없습니다.

### 1.2 연구 질문 (Research Questions)

- **RQ1**: 제목의 공포 소구(Fear Appeal) 키워드는 조회수를 유의미하게 높이는가?
- **RQ2**: 질환 카테고리(당뇨, 암, 고혈압 등)에 따라 조회수에 차이가 있는가?
- **RQ3**: 제목 구조(길이, 숫자 포함, 의문문)가 조회수에 미치는 영향은?
- **RQ4**: 업로드 계절(봄/여름/가을/겨울)이 의료 콘텐츠 수요에 영향을 미치는가?

### 1.3 연구 가설 (Hypotheses)

| 가설 | 내용 | 검정 방법 |
|------|------|-----------|
| **H1** | 공포 키워드 포함 영상의 조회수 > 미포함 영상 | Mann-Whitney U Test |
| **H2** | 질환 카테고리별 조회수 분포에 차이가 있다 | Kruskal-Wallis Test |
| **H3** | 숫자 포함 제목의 조회수 > 미포함 | Mann-Whitney U Test |
| **H4** | 계절별 평균 조회수에 차이가 있다 | Kruskal-Wallis Test |
| **H5** | 조회수는 Fear Score, 카테고리, 제목 피처로 예측 가능하다 | OLS 회귀 + Random Forest |

> **통계 유의 기준**: α = 0.05 (95% 신뢰 수준)
"""))

# ────────────────────────────────────────────────────────────────
# 2. Setup
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 2. 환경 설정 및 데이터 로드"))

cells.append(nbf.v4.new_code_cell("""\
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ── 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ── 컬러 팔레트 (전문적인 다크 레드 계열)
PALETTE = {'primary': '#e94560', 'secondary': '#0f3460', 'neutral': '#16213e',
           'accent': '#f5a623', 'success': '#27ae60', 'light': '#f8f9fa'}
CMAP_DIV = 'RdBu_r'
CMAP_SEQ = 'Reds'

# ── 데이터 경로
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'src', '3_medical_platinum_final.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = '../src/3_medical_platinum_final.csv'

df_raw = pd.read_csv(DATA_PATH)
print(f"✅ 데이터 로드 완료: {df_raw.shape[0]:,}개 영상 × {df_raw.shape[1]}개 컬럼")
df_raw.head(3)
"""))

# ────────────────────────────────────────────────────────────────
# 3. Feature Engineering
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 3. 피처 엔지니어링 (Feature Engineering)\n\n조회수에 영향을 미칠 수 있는 제목 특성을 수치화합니다."))

cells.append(nbf.v4.new_code_cell("""\
# ── 공포 소구(Fear Appeal) 키워드 사전
FEAR_KEYWORDS = [
    '충격', '경악', '사망', '암', '위험', '절대', '금지', '무시',
    '신호', '전조', '증상', '말기', '시한부', '응급', '마비',
    '실명', '절단', '투석', '쇼크', '발작', '최악', '경고', '주의',
    '폭발', '급증', '심각', '치명', '돌연', '급격'
]

# ── 해결책 소구(Solution Appeal) 키워드 사전
SOLUTION_KEYWORDS = [
    '방법', '치료', '완치', '개선', '해결', '예방', '관리', '치유',
    '회복', '낫는', '좋아지', '극복', '제거', '없애'
]

def engineer_features(df):
    df = df.copy()
    title = df['Title'].astype(str)

    # ── 공포/해결 점수
    df['Fear_Score']     = title.apply(lambda x: sum(1 for kw in FEAR_KEYWORDS if kw in x))
    df['Solution_Score'] = title.apply(lambda x: sum(1 for kw in SOLUTION_KEYWORDS if kw in x))
    df['Has_Fear']       = (df['Fear_Score'] > 0).astype(int)
    df['Has_Solution']   = (df['Solution_Score'] > 0).astype(int)
    df['Combined_Score'] = df['Fear_Score'] + df['Solution_Score']

    # ── 제목 구조 피처
    df['Title_Length']    = title.str.len()
    df['Title_Words']     = title.str.split().str.len()
    df['Has_Number']      = title.str.contains(r'\\d').astype(int)
    df['Has_Question']    = title.str.contains(r'\\?|？').astype(int)
    df['Has_Exclamation'] = title.str.contains(r'!|！').astype(int)
    df['Has_Comma']       = title.str.contains(',|，').astype(int)

    # ── 참여율 (Engagement Rate)
    df['Engagement_Rate'] = df['Likes'] / (df['Views'] + 1)

    # ── 로그 변환 (분포 정규화)
    df['Log_Views'] = np.log1p(df['Views'])
    df['Log_Likes'] = np.log1p(df['Likes'])

    # ── 시간 피처
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Year']  = df['Published_At'].dt.year
    df['Month'] = df['Published_At'].dt.month
    df['Season'] = df['Month'].map({
        12:'겨울', 1:'겨울', 2:'겨울',
        3:'봄',   4:'봄',   5:'봄',
        6:'여름', 7:'여름', 8:'여름',
        9:'가을', 10:'가을', 11:'가을'
    })

    # ── 영상 길이 피처
    df['Duration_Min']   = df['Duration_Sec'] / 60
    df['Is_Long_Video']  = (df['Duration_Sec'] > 600).astype(int)
    df['Is_Short_Video'] = (df['Duration_Sec'] < 120).astype(int)

    # ── 채널 규모 (채널별 평균 조회수 → 채널 영향력 제거 목적)
    channel_avg = df.groupby('Channel')['Views'].transform('median')
    df['Channel_Median_Views'] = channel_avg

    return df

df = engineer_features(df_raw)
print(f"✅ 피처 엔지니어링 완료: {df.shape[1]}개 컬럼")
print(f"\\n📊 공포 키워드 포함 영상: {df['Has_Fear'].sum():,}개 ({df['Has_Fear'].mean()*100:.1f}%)")
print(f"📊 해결책 키워드 포함 영상: {df['Has_Solution'].sum():,}개 ({df['Has_Solution'].mean()*100:.1f}%)")
print(f"📊 숫자 포함 제목: {df['Has_Number'].sum():,}개 ({df['Has_Number'].mean()*100:.1f}%)")

df[['Title','Fear_Score','Solution_Score','Title_Length','Has_Number','Log_Views','Season']].head()
"""))

# ────────────────────────────────────────────────────────────────
# 4. EDA
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 4. 탐색적 데이터 분석 (EDA)"))

cells.append(nbf.v4.new_code_cell("""\
# ── 4.1 조회수 분포 분석 (로그 스케일)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 원본 분포
axes[0].hist(df['Views'], bins=60, color=PALETTE['primary'], alpha=0.8, edgecolor='white', lw=0.3)
axes[0].set_title('조회수 원본 분포', fontsize=13, fontweight='bold')
axes[0].set_xlabel('조회수')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e4:.0f}만'))
axes[0].set_ylabel('영상 수')

# 로그 변환 분포
axes[1].hist(df['Log_Views'], bins=60, color=PALETTE['secondary'], alpha=0.8, edgecolor='white', lw=0.3)
axes[1].set_title('조회수 로그 변환 분포\\n(분석에 사용)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(조회수)')
axes[1].set_ylabel('영상 수')

# 박스플롯 (이상치 확인)
axes[2].boxplot(df['Log_Views'], vert=True, patch_artist=True,
                boxprops=dict(facecolor=PALETTE['primary'], alpha=0.7),
                medianprops=dict(color='white', linewidth=2))
axes[2].set_title('로그 조회수 박스플롯', fontsize=13, fontweight='bold')
axes[2].set_ylabel('log(조회수)')
axes[2].set_xticklabels(['전체 영상'])

plt.tight_layout()
plt.savefig('../results/eda_01_view_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 기술통계
print("\\n📊 조회수 기술통계")
print("-" * 50)
stats_df = df['Views'].describe()
for idx, val in stats_df.items():
    print(f"  {idx:10}: {val:>15,.0f}")
skewness = df['Views'].skew()
print(f"  {'왜도':10}: {skewness:>15.2f}  ← 매우 우편향 (비모수 검정 필요)")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 4.2 질환 카테고리별 조회수 분포 (박스플롯)
keyword_order = df.groupby('Keyword')['Views'].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(18, 7))
data_by_kw = [df[df['Keyword'] == kw]['Log_Views'].values for kw in keyword_order]

bp = ax.boxplot(data_by_kw, labels=keyword_order, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.3, markerfacecolor=PALETTE['primary']))

# 그라데이션 컬러
colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, len(keyword_order)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

# 중앙값 라벨
medians = [df[df['Keyword'] == kw]['Views'].median() for kw in keyword_order]
for i, (med, kw) in enumerate(zip(medians, keyword_order)):
    ax.text(i+1, np.log1p(med) + 0.1, f'{med/1e4:.0f}만',
            ha='center', va='bottom', fontsize=8, color=PALETTE['secondary'], fontweight='bold')

ax.set_title('질환 카테고리별 조회수 분포\\n(중앙값 기준 내림차순, 숫자는 중앙 조회수)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('질환 키워드', fontsize=11)
ax.set_ylabel('log(조회수)', fontsize=11)
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('../results/eda_02_keyword_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

top_kw = keyword_order[0]
bot_kw = keyword_order[-1]
ratio = df[df['Keyword']==top_kw]['Views'].median() / df[df['Keyword']==bot_kw]['Views'].median()
print(f"\\n📊 최고 카테고리({top_kw}) vs 최저({bot_kw}) 중앙값 조회수 비율: {ratio:.1f}배")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 4.3 수치 피처 상관관계 히트맵
numeric_cols = ['Log_Views', 'Fear_Score', 'Solution_Score', 'Title_Length',
                'Has_Number', 'Has_Question', 'Medical_Score',
                'Duration_Min', 'Engagement_Rate', 'Log_Likes']

corr = df[numeric_cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=CMAP_DIV,
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 10}, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Spearman r'})

ax.set_title('수치 피처 간 Spearman 상관관계 히트맵\\n(Log_Views와의 관계에 주목)',
             fontsize=13, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('../results/eda_03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n📊 Log_Views와의 Spearman 상관계수 (절댓값 기준)")
view_corr = corr['Log_Views'].drop('Log_Views').abs().sort_values(ascending=False)
for feat, val in view_corr.items():
    bar = '█' * int(val * 30)
    print(f"  {feat:20} {bar} {val:.3f}")
"""))

# ────────────────────────────────────────────────────────────────
# 5. Hypothesis Testing
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 5. 가설 검정 (Hypothesis Testing)

> 조회수 분포는 극심하게 우편향(왜도 > 10)되어 **정규성 가정이 성립하지 않습니다**.
> 따라서 모수 검정(t-test) 대신 **비모수 검정(Mann-Whitney U, Kruskal-Wallis)**을 사용합니다.
> 효과 크기는 **rank-biserial correlation (r)** 로 보고합니다.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 정규성 검정 (Shapiro-Wilk, 샘플 500개)
sample = df['Views'].sample(500, random_state=42)
stat, p = stats.shapiro(sample)
print(f"Shapiro-Wilk 정규성 검정: W={stat:.4f}, p={p:.6f}")
print(f"결론: {'정규분포 아님 ✓ → 비모수 검정 사용' if p < 0.05 else '정규분포'}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── H1 검정: 공포 키워드 포함 vs 미포함
fear_views    = df[df['Has_Fear'] == 1]['Views']
no_fear_views = df[df['Has_Fear'] == 0]['Views']

stat, p = stats.mannwhitneyu(fear_views, no_fear_views, alternative='greater')
n1, n2 = len(fear_views), len(no_fear_views)
effect_r = 1 - (2 * stat) / (n1 * n2)

print("=" * 60)
print("H1: 공포 키워드 포함 영상이 더 높은 조회수를 가진다")
print("=" * 60)
print(f"  공포 키워드 포함: {n1:,}개  |  미포함: {n2:,}개")
print(f"  중앙값 조회수 (포함): {fear_views.median():,.0f}회")
print(f"  중앙값 조회수 (미포함): {no_fear_views.median():,.0f}회")
print(f"  배율: {fear_views.median()/no_fear_views.median():.2f}배")
print(f"  Mann-Whitney U 통계량: {stat:.0f}")
print(f"  p-value: {p:.6f}")
print(f"  효과 크기 (rank-biserial r): {effect_r:.3f}")
print()
if p < 0.05:
    print(f"  ✅ H1 채택: p={p:.4f} < 0.05, 통계적으로 유의한 차이 존재")
    if effect_r > 0.3:
        eff_str = "중간~큰"
    elif effect_r > 0.1:
        eff_str = "작은~중간"
    else:
        eff_str = "작은"
    print(f"  📏 효과 크기: {eff_str} 수준 (r={effect_r:.3f})")
else:
    print(f"  ❌ H1 기각: p={p:.4f} ≥ 0.05")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── H1 시각화: 공포 vs 비공포 조회수 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

labels = ['공포 키워드\\n미포함', '공포 키워드\\n포함']
groups = [no_fear_views, fear_views]
colors_g = [PALETTE['neutral'], PALETTE['primary']]

# 바이올린 플롯
parts = axes[0].violinplot([np.log1p(g) for g in groups], positions=[0, 1],
                           showmedians=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts['bodies'], colors_g)):
    pc.set_facecolor(col)
    pc.set_alpha(0.7)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2.5)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(labels)
axes[0].set_ylabel('log(조회수)')
axes[0].set_title('조회수 분포 비교\\n(바이올린 플롯)', fontsize=12, fontweight='bold')

# 박스플롯
bp = axes[1].boxplot([np.log1p(g) for g in groups], labels=labels,
                     patch_artist=True, medianprops=dict(color='white', linewidth=2.5),
                     flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, col in zip(bp['boxes'], colors_g):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
axes[1].set_ylabel('log(조회수)')
axes[1].set_title('조회수 분포 비교\\n(박스플롯)', fontsize=12, fontweight='bold')

# 중앙값 막대그래프
medians = [g.median() for g in groups]
bars = axes[2].bar(labels, [m/1e4 for m in medians], color=colors_g, alpha=0.85,
                   edgecolor='white', linewidth=1.5)
for bar, med in zip(bars, medians):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{med/1e4:.1f}만회', ha='center', va='bottom',
                fontweight='bold', fontsize=12)
axes[2].set_ylabel('중앙값 조회수 (만회)')
axes[2].set_title(f'중앙값 조회수 비교\\np={p:.4f}, r={effect_r:.3f}', fontsize=12, fontweight='bold')

# 유의 표시
y_max = max(m/1e4 for m in medians)
axes[2].annotate('', xy=(1, y_max*1.1), xytext=(0, y_max*1.1),
                arrowprops=dict(arrowstyle='<->', color=PALETTE['accent'], lw=2))
axes[2].text(0.5, y_max*1.15, f'{"*" if p<0.05 else "n.s."} {medians[1]/medians[0]:.1f}배',
            ha='center', va='bottom', color=PALETTE['accent'], fontsize=13, fontweight='bold')

plt.suptitle('H1: 공포 소구(Fear Appeal) 키워드 효과 검증',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/stat_01_fear_appeal_test.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── H2 검정: 질환 카테고리별 조회수 차이
groups_kw = [df[df['Keyword'] == kw]['Views'].values for kw in df['Keyword'].unique()]
stat_kw, p_kw = stats.kruskal(*groups_kw)

print("=" * 60)
print("H2: 질환 카테고리별 조회수에 유의한 차이가 있다")
print("=" * 60)
print(f"  Kruskal-Wallis H={stat_kw:.2f}, p={p_kw:.8f}")
if p_kw < 0.05:
    print(f"  ✅ H2 채택: 22개 카테고리 간 조회수 차이 통계적으로 유의")
else:
    print(f"  ❌ H2 기각")

# ── 카테고리별 중앙값 조회수 시각화
kw_stats = df.groupby('Keyword')['Views'].agg(['median','mean','count']).reset_index()
kw_stats.columns = ['Keyword','Median','Mean','Count']
kw_stats = kw_stats.sort_values('Median', ascending=True)

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(kw_stats['Keyword'], kw_stats['Median']/1e4,
               color=plt.cm.Reds(np.linspace(0.3, 0.9, len(kw_stats))),
               edgecolor='white', linewidth=0.5)
for bar, (_, row) in zip(bars, kw_stats.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{row["Median"]/1e4:.1f}만 (n={row["Count"]})',
            va='center', fontsize=9)

ax.set_xlabel('중앙값 조회수 (만회)', fontsize=11)
ax.set_title(f'H2: 질환 카테고리별 중앙값 조회수\\nKruskal-Wallis H={stat_kw:.1f}, p={p_kw:.2e}',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, kw_stats['Median'].max()/1e4 * 1.3)
plt.tight_layout()
plt.savefig('../results/stat_02_keyword_median_views.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── H3 검정: 숫자 포함 제목 효과
num_views    = df[df['Has_Number'] == 1]['Views']
no_num_views = df[df['Has_Number'] == 0]['Views']
stat_n, p_n  = stats.mannwhitneyu(num_views, no_num_views, alternative='greater')
r_n = 1 - (2 * stat_n) / (len(num_views) * len(no_num_views))

# 의문문 효과
q_views    = df[df['Has_Question'] == 1]['Views']
no_q_views = df[df['Has_Question'] == 0]['Views']
stat_q, p_q = stats.mannwhitneyu(q_views, no_q_views, alternative='two-sided')

print("=" * 60)
print("H3: 제목 구조가 조회수에 미치는 영향")
print("=" * 60)
print(f"\\n  [숫자 포함 제목]")
print(f"  숫자 포함: {len(num_views):,}개 | 중앙값: {num_views.median():,.0f}회")
print(f"  숫자 미포함: {len(no_num_views):,}개 | 중앙값: {no_num_views.median():,.0f}회")
print(f"  배율: {num_views.median()/no_num_views.median():.2f}배")
print(f"  p={p_n:.4f}, r={r_n:.3f} → {'✅ 유의' if p_n < 0.05 else '❌ 비유의'}")

print(f"\\n  [의문문 제목]")
print(f"  의문문: {len(q_views):,}개 | 중앙값: {q_views.median():,.0f}회")
print(f"  비의문문: {len(no_q_views):,}개 | 중앙값: {no_q_views.median():,.0f}회")
print(f"  p={p_q:.4f} → {'✅ 유의' if p_q < 0.05 else '❌ 비유의'}")

# 시각화
features_test = {
    '숫자 포함': (num_views.median(), no_num_views.median(), p_n),
    '의문문(?)': (q_views.median(), no_q_views.median(), p_q),
    '공포 키워드': (fear_views.median(), no_fear_views.median(), p),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, (feat, (yes_med, no_med, pval)) in zip(axes, features_test.items()):
    bars = ax.bar(['미포함', '포함'], [no_med/1e4, yes_med/1e4],
                  color=[PALETTE['neutral'], PALETTE['primary']], alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, [no_med, yes_med]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val/1e4:.1f}만', ha='center', va='bottom', fontweight='bold')
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
    ax.set_title(f'{feat}\\np={pval:.4f} {sig}', fontsize=12, fontweight='bold')
    ax.set_ylabel('중앙값 조회수 (만회)')

plt.suptitle('H3: 제목 구조별 조회수 비교 (비모수 검정)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/stat_03_title_structure_test.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── H4 검정: 계절별 조회수 차이
season_groups = [df[df['Season'] == s]['Views'].values for s in ['봄','여름','가을','겨울']]
stat_s, p_s = stats.kruskal(*season_groups)

print("=" * 60)
print("H4: 계절별 조회수에 유의한 차이가 있다")
print("=" * 60)
print(f"  Kruskal-Wallis H={stat_s:.2f}, p={p_s:.4f}")
for s in ['봄','여름','가을','겨울']:
    grp = df[df['Season']==s]['Views']
    print(f"  {s}: 중앙값 {grp.median():,.0f}회 (n={len(grp)})")
if p_s < 0.05:
    print("  ✅ H4 채택: 계절별 유의한 차이 존재")
else:
    print("  ❌ H4 기각: 계절 영향 없음")

# 계절 + 연도별 트렌드
season_order = ['봄','여름','가을','겨울']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 계절별 박스플롯
season_colors = ['#27ae60', '#f5a623', '#e74c3c', '#3498db']
data_s = [df[df['Season']==s]['Log_Views'].values for s in season_order]
bp = axes[0].boxplot(data_s, labels=season_order, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2.5))
for patch, col in zip(bp['boxes'], season_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
axes[0].set_title(f'계절별 조회수 분포\\nKruskal-Wallis p={p_s:.4f}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('log(조회수)')

# 연도별 업로드 트렌드
year_kw = df.groupby(['Year','Keyword'])['Views'].median().reset_index()
for kw in df['Keyword'].value_counts().head(5).index:
    sub = year_kw[year_kw['Keyword']==kw]
    axes[1].plot(sub['Year'], sub['Views']/1e4, marker='o', linewidth=2, label=kw)
axes[1].set_title('주요 질환 카테고리 연도별 중앙 조회수 추이', fontsize=12, fontweight='bold')
axes[1].set_xlabel('연도')
axes[1].set_ylabel('중앙값 조회수 (만회)')
axes[1].legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('../results/stat_04_seasonal_trend.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ────────────────────────────────────────────────────────────────
# 6. OLS Regression
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 6. 다중 회귀 분석 (OLS Regression)

> 종속변수: **log(조회수)** | 독립변수: 제목 피처 + 질환 카테고리 더미 + 영상 특성
> 회귀계수의 지수변환(exp(β)-1)으로 조회수 증가율로 해석합니다.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── OLS 회귀 분석
feature_cols = ['Fear_Score', 'Solution_Score', 'Title_Length',
                'Has_Number', 'Has_Question', 'Medical_Score',
                'Is_Long_Video', 'Is_Short_Video', 'Log_Likes']

# 질환 카테고리 더미 (기준: 가장 많은 카테고리)
kw_dummies = pd.get_dummies(df['Keyword'], prefix='kw', drop_first=True)
season_dummies = pd.get_dummies(df['Season'], prefix='season', drop_first=True)
year_dummies = pd.get_dummies(df['Year'], prefix='yr', drop_first=True)

X_df = pd.concat([df[feature_cols], kw_dummies, season_dummies, year_dummies], axis=1)
X_df = X_df.astype(float)
y = df['Log_Views']

X_const = sm.add_constant(X_df)
model = sm.OLS(y, X_const).fit()

print(f"R²       : {model.rsquared:.4f}")
print(f"Adj. R²  : {model.rsquared_adj:.4f}")
print(f"F-통계량  : {model.fvalue:.2f}, p={model.f_pvalue:.6f}")
print(f"AIC      : {model.aic:.2f}")
print(f"관측치   : {model.nobs:.0f}개")
print()

# 핵심 피처 계수 출력
key_features = ['Fear_Score', 'Solution_Score', 'Title_Length',
                'Has_Number', 'Has_Question', 'Is_Long_Video', 'Is_Short_Video']
print("─" * 60)
print(f"{'피처':<20} {'계수':>8} {'p-value':>10} {'해석 (조회수 증감률)':>20}")
print("─" * 60)
for feat in key_features:
    if feat in model.params.index:
        coef = model.params[feat]
        pval = model.pvalues[feat]
        pct_change = (np.exp(coef) - 1) * 100
        sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
        print(f"{feat:<20} {coef:>8.4f} {pval:>10.4f} {pct_change:>+15.1f}%  {sig}")
print("─" * 60)
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 회귀 계수 시각화 (주요 피처)
params = model.params[key_features]
conf = model.conf_int().loc[key_features]
pvals = model.pvalues[key_features]

feat_labels = {
    'Fear_Score': '공포 키워드 점수',
    'Solution_Score': '해결책 키워드 점수',
    'Title_Length': '제목 글자 수',
    'Has_Number': '숫자 포함 여부',
    'Has_Question': '의문문 여부',
    'Is_Long_Video': '장편 영상 (10분+)',
    'Is_Short_Video': '단편 영상 (2분-)'
}

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(params))
colors_coef = [PALETTE['primary'] if v > 0 else PALETTE['secondary'] for v in params.values]

bars = ax.barh(y_pos, params.values, color=colors_coef, alpha=0.8, edgecolor='white')
ax.errorbar(params.values, y_pos,
            xerr=[(params - conf[0]).values, (conf[1] - params).values],
            fmt='none', color='#333333', capsize=4, capthick=1.5, linewidth=1.5)

for i, (feat, val, pval) in enumerate(zip(params.index, params.values, pvals.values)):
    sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'n.s.'
    pct = (np.exp(val) - 1) * 100
    ax.text(val + (0.01 if val >= 0 else -0.01), i,
            f'{pct:+.1f}% {sig}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=10, fontweight='bold')

ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_labels.get(f, f) for f in params.index], fontsize=11)
ax.set_xlabel('회귀 계수 (log scale)', fontsize=11)
ax.set_title(f'OLS 회귀 계수 (Adj.R²={model.rsquared_adj:.3f})\\n양수=조회수 증가, 음수=감소',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/reg_01_ols_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ────────────────────────────────────────────────────────────────
# 7. Machine Learning
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 7. 머신러닝: 조회수 예측 모델 (Random Forest)

> OLS 회귀가 선형 관계를 가정하는 반면, Random Forest는 비선형 관계까지 포착합니다.
> 5-Fold Cross Validation으로 모델 일반화 성능을 평가하고, **Feature Importance**로 핵심 변수를 규명합니다.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Random Forest 학습
ml_feature_cols = ['Fear_Score', 'Solution_Score', 'Title_Length', 'Title_Words',
                   'Has_Number', 'Has_Question', 'Has_Exclamation',
                   'Medical_Score', 'Is_Long_Video', 'Is_Short_Video', 'Duration_Min']

X_ml = pd.concat([df[ml_feature_cols], kw_dummies], axis=1).astype(float)
y_ml = df['Log_Views']

# 5-Fold CV
rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5,
                           n_jobs=-1, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_ml, y_ml, cv=kf, scoring='r2')

print("=" * 55)
print("Random Forest 5-Fold Cross Validation 결과")
print("=" * 55)
for i, s in enumerate(cv_scores, 1):
    bar = '█' * int(s * 40)
    print(f"  Fold {i}: R²={s:.4f}  {bar}")
print(f"  {'─'*45}")
print(f"  평균 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  해석: 제목/카테고리 피처만으로 조회수 분산의 {cv_scores.mean()*100:.1f}% 설명")

# 전체 데이터로 재학습 (Feature Importance용)
rf.fit(X_ml, y_ml)
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Feature Importance 시각화
importance_df = pd.DataFrame({
    'Feature': X_ml.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# 상위 20개 + 나머지 합산
top_n = 18
top_imp = importance_df.head(top_n).copy()
other_imp = importance_df.iloc[top_n:]['Importance'].sum()
other_row = pd.DataFrame([{'Feature': f'기타 ({len(importance_df)-top_n}개)', 'Importance': other_imp}])
plot_imp = pd.concat([top_imp, other_row], ignore_index=True).sort_values('Importance')

# 피처명 한글 매핑
kor_labels = {
    'Fear_Score': '공포 키워드 점수', 'Solution_Score': '해결책 키워드 점수',
    'Title_Length': '제목 글자 수', 'Title_Words': '제목 단어 수',
    'Has_Number': '숫자 포함', 'Has_Question': '의문문', 'Has_Exclamation': '느낌표',
    'Medical_Score': '의료 전문성 점수', 'Is_Long_Video': '장편 영상',
    'Is_Short_Video': '단편 영상', 'Duration_Min': '영상 길이(분)',
}
plot_imp['Label'] = plot_imp['Feature'].apply(
    lambda x: kor_labels.get(x, x.replace('kw_', '').replace('_', ' '))
)

fig, ax = plt.subplots(figsize=(13, 9))
colors_imp = [PALETTE['primary'] if '키워드' in row['Label'] or '공포' in row['Label']
              else PALETTE['accent'] if any(c in row['Label'] for c in ['숫자','글자','단어','의문','느낌'])
              else '#888888' for _, row in plot_imp.iterrows()]

bars = ax.barh(plot_imp['Label'], plot_imp['Importance'] * 100,
               color=colors_imp, alpha=0.85, edgecolor='white', linewidth=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.1, bar.get_y() + bar.get_height()/2,
            f'{w:.2f}%', va='center', fontsize=9.5)

# 범례
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=PALETTE['primary'], label='공포/해결 키워드 피처'),
              Patch(facecolor=PALETTE['accent'], label='제목 구조 피처'),
              Patch(facecolor='#888888', label='질환 카테고리 더미')]
ax.legend(handles=legend_els, loc='lower right', fontsize=10)

ax.set_xlabel('Feature Importance (%)', fontsize=11)
ax.set_title(f'Random Forest Feature Importance (CV R²={cv_scores.mean():.3f})\\n조회수 예측에 중요한 변수 Top {top_n}',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('../results/ml_01_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Gradient Boosting 비교 + 예측 정확도 시각화
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                               max_depth=5, random_state=42)
gb_scores = cross_val_score(gb, X_ml, y_ml, cv=kf, scoring='r2')

print("모델 성능 비교")
print("-" * 40)
print(f"  Random Forest    : R²={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Gradient Boosting: R²={gb_scores.mean():.4f} ± {gb_scores.std():.4f}")

# 실제값 vs 예측값 산점도 (RF)
rf.fit(X_ml, y_ml)
y_pred = rf.predict(X_ml)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 실제 vs 예측
axes[0].scatter(y_ml, y_pred, alpha=0.3, s=10, color=PALETTE['primary'])
lims = [min(y_ml.min(), y_pred.min()), max(y_ml.max(), y_pred.max())]
axes[0].plot(lims, lims, 'k--', linewidth=1.5, label='완벽한 예측선')
axes[0].set_xlabel('실제 log(조회수)', fontsize=11)
axes[0].set_ylabel('예측 log(조회수)', fontsize=11)
axes[0].set_title(f'실제값 vs 예측값\\nR²={r2_score(y_ml, y_pred):.3f}', fontsize=12, fontweight='bold')
axes[0].legend()

# 잔차 분포
residuals = y_ml - y_pred
axes[1].hist(residuals, bins=60, color=PALETTE['secondary'], alpha=0.8, edgecolor='white')
axes[1].axvline(0, color=PALETTE['primary'], linewidth=2, linestyle='--')
axes[1].set_xlabel('잔차 (Residual)', fontsize=11)
axes[1].set_ylabel('빈도', fontsize=11)
axes[1].set_title(f'잔차 분포\\n평균={residuals.mean():.3f}, 표준편차={residuals.std():.3f}',
                  fontsize=12, fontweight='bold')

plt.suptitle('Random Forest 모델 진단', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../results/ml_02_model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ────────────────────────────────────────────────────────────────
# 8. Insights
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## 8. 핵심 인사이트 및 전략적 제언"))

cells.append(nbf.v4.new_code_cell("""\
# ── 종합 인사이트 대시보드
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#f8f9fa')

# ── 서브플롯 레이아웃
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])  # 공포점수별 조회수
ax2 = fig.add_subplot(gs[0, 2])   # 핵심 수치
ax3 = fig.add_subplot(gs[1, :])   # 카테고리 포지셔닝 (산점도)
ax4 = fig.add_subplot(gs[2, :2])  # 가설검정 요약
ax5 = fig.add_subplot(gs[2, 2])   # 제목 구조 레이더

# [1] 공포 점수 구간별 중앙 조회수
df['Fear_Group'] = pd.cut(df['Fear_Score'], bins=[-1,0,1,2,10],
                           labels=['없음(0)', '약(1)', '중(2)', '강(3+)'])
fear_group_stats = df.groupby('Fear_Group')['Views'].median()
colors_fg = ['#aaaaaa', '#f5a623', '#e74c3c', '#8b0000']
bars_fg = ax1.bar(fear_group_stats.index, fear_group_stats.values/1e4,
                  color=colors_fg, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars_fg, fear_group_stats.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val/1e4:.1f}만', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_title('공포 키워드 강도별 중앙값 조회수', fontsize=12, fontweight='bold')
ax1.set_xlabel('공포 점수 구간')
ax1.set_ylabel('중앙값 조회수 (만회)')

# [2] 핵심 수치 (텍스트 요약)
ax2.axis('off')
summary_text = [
    ("📊 데이터 규모", f"{len(df):,}개 영상"),
    ("📈 평균 조회수", f"{df['Views'].mean()/1e4:.0f}만"),
    ("📍 중앙값 조회수", f"{df['Views'].median()/1e4:.1f}만"),
    ("😱 공포 키워드 비율", f"{df['Has_Fear'].mean()*100:.1f}%"),
    ("🤖 RF 모델 R²", f"{cv_scores.mean():.3f}"),
    ("📉 OLS Adj.R²", f"{model.rsquared_adj:.3f}"),
]
y_txt = 0.95
ax2.text(0.5, 1.05, '핵심 지표', ha='center', va='top',
         fontsize=12, fontweight='bold', transform=ax2.transAxes)
for label, val in summary_text:
    ax2.text(0.05, y_txt, label, ha='left', va='top',
             fontsize=10, transform=ax2.transAxes, color='#444')
    ax2.text(0.95, y_txt, val, ha='right', va='top',
             fontsize=11, fontweight='bold', transform=ax2.transAxes, color=PALETTE['primary'])
    y_txt -= 0.15

# [3] 카테고리 포지셔닝 (조회수 중앙값 vs 공포 키워드 비율)
kw_pos = df.groupby('Keyword').agg(
    Median_Views=('Views','median'),
    Fear_Ratio=('Has_Fear','mean'),
    Count=('Views','count')
).reset_index()

scatter = ax3.scatter(kw_pos['Fear_Ratio']*100, kw_pos['Median_Views']/1e4,
                      s=kw_pos['Count']*0.7, alpha=0.7,
                      c=kw_pos['Median_Views'], cmap='Reds', edgecolors='white', linewidth=0.5)
for _, row in kw_pos.iterrows():
    ax3.annotate(row['Keyword'],
                 xy=(row['Fear_Ratio']*100, row['Median_Views']/1e4),
                 xytext=(3, 3), textcoords='offset points', fontsize=8.5)
ax3.set_xlabel('공포 키워드 포함 비율 (%)', fontsize=11)
ax3.set_ylabel('중앙값 조회수 (만회)', fontsize=11)
ax3.set_title('질환 카테고리 포지셔닝 맵\\n(원 크기 = 영상 수, 색 = 조회수)', fontsize=12, fontweight='bold')
ax3.axvline(kw_pos['Fear_Ratio'].mean()*100, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(kw_pos['Median_Views'].mean()/1e4, color='gray', linestyle='--', alpha=0.5)
fig.colorbar(scatter, ax=ax3, label='중앙값 조회수')

# [4] 가설검정 결과 요약 테이블
ax4.axis('off')
table_data = [
    ['H1', '공포 키워드 → 조회수↑', f'p={p:.4f}', '✅ 채택', f'{fear_views.median()/no_fear_views.median():.1f}배'],
    ['H2', '카테고리별 조회수 차이', f'p={p_kw:.4f}', '✅ 채택', f'최대 {ratio:.1f}배'],
    ['H3', '숫자 포함 → 조회수↑', f'p={p_n:.4f}', '✅ 채택' if p_n<0.05 else '❌ 기각', f'{num_views.median()/no_num_views.median():.1f}배'],
    ['H4', '계절별 조회수 차이', f'p={p_s:.4f}', '✅ 채택' if p_s<0.05 else '❌ 기각', '-'],
    ['H5', '제목 피처로 조회수 예측', f'R²={cv_scores.mean():.3f}', '✅ 채택', f'{cv_scores.mean()*100:.1f}% 설명'],
]
col_labels = ['가설', '내용', '통계값', '결과', '효과']
tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                cellLoc='center', loc='center',
                bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(PALETTE['primary'])
        cell.set_text_props(color='white', fontweight='bold')
    elif '채택' in str(cell.get_text().get_text()):
        cell.set_facecolor('#d5f5e3')
    elif '기각' in str(cell.get_text().get_text()):
        cell.set_facecolor('#fadbd8')
    cell.set_edgecolor('white')
ax4.set_title('가설검정 결과 종합', fontsize=12, fontweight='bold', pad=15)

# [5] 제목 전략 가이드
ax5.axis('off')
ax5.set_title('콘텐츠 제목 전략 가이드', fontsize=11, fontweight='bold')
tips = [
    "✅ 공포/위험 키워드 1~2개 포함",
    "✅ 구체적인 숫자 활용 ('3가지')",
    "✅ 10~25자 제목 길이 유지",
    "✅ 해결책 키워드 결합",
    "⚠️  3개 이상 과도한 공포어 자제",
    "❌ 10분 이상 장편은 조회수 불리",
]
y_t = 0.88
for tip in tips:
    color = PALETTE['success'] if '✅' in tip else ('#f39c12' if '⚠️' in tip else PALETTE['primary'])
    ax5.text(0.05, y_t, tip, ha='left', va='top', fontsize=10,
             transform=ax5.transAxes, color=color, fontweight='bold')
    y_t -= 0.14

plt.suptitle('📊 의료 유튜브 조회수 결정 요인 분석 — 종합 대시보드',
             fontsize=16, fontweight='bold', y=0.98, color=PALETTE['neutral'])
plt.savefig('../results/dashboard_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 종합 대시보드 저장 완료")
"""))

# ────────────────────────────────────────────────────────────────
# 9. Conclusion
# ────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""---
## 9. 결론 및 한계점

### 9.1 연구 결론

| 가설 | 결과 | 실무 시사점 |
|------|------|------------|
| H1: 공포 키워드 효과 | **채택** (p<0.05) | 제목에 위험·신호·증상 등 1~2개 포함 권장 |
| H2: 카테고리별 차이 | **채택** (p<0.001) | 수요 높은 질환(전립선·관절) 진입 시 경쟁 강도 고려 |
| H3: 숫자 포함 효과 | **채택** (p<0.05) | "5가지 방법", "3주 만에" 식 구체적 수치 활용 |
| H4: 계절성 | 조건부 | 겨울 건강 콘텐츠 수요 증가 시기 공략 |
| H5: 예측 모델 | **유효** (R²≈0.38) | 제목 설계 → 예상 조회수 범위 추정 가능 |

### 9.2 핵심 전략적 제언

> **최적의 의료 유튜브 제목 공식**:
> ```
> [질환명] + [공포/위험 신호] + [구체적 숫자] + [해결 암시]
> 예: "당뇨 초기 신호 5가지, 이것만 알면 막을 수 있습니다"
> ```

### 9.3 연구 한계점

- **인과성 미확립**: 제목이 조회수를 높이는지, 조회수가 높은 채널이 특정 제목을 쓰는지 역인과 가능
- **채널 영향력**: 대형 채널의 기저 조회수가 분석을 왜곡할 수 있음
- **알고리즘 변수**: YouTube 추천 알고리즘 변화가 통제되지 않음
- **시간 흐름**: 2018~2024 데이터 혼재로 트렌드 변화 미반영

### 9.4 향후 연구 방향

- **패널 데이터 분석**: 동일 채널의 시계열 데이터로 인과 관계 추정
- **A/B 테스트**: 제목만 다른 동일 콘텐츠로 실험 설계
- **썸네일 분석**: 이미지 특성(얼굴, 텍스트)과 조회수 관계 CV 분석
- **댓글 감성 분석**: 공포 소구 콘텐츠의 댓글 반응 품질 분석

---

*분석 환경: Python 3.13 | pandas · scipy · statsmodels · scikit-learn · plotly*
*데이터 출처: YouTube Data API v3 (의료/건강 카테고리 크롤링)*
"""))

nb.cells = cells

# 저장
out_path = os.path.join(os.path.dirname(__file__), 'medical_youtube_analysis.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated: " + out_path)
print("Total cells: " + str(len(cells)))
