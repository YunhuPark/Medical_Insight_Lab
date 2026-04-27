"""
Phase 2-G: 이상치 탐지 — 조회수 조작(어뷰징) 의심 영상 식별
방법: Isolation Forest + 다차원 이상 탐지
의심 패턴: 조회수 극히 높은데 좋아요/댓글 비율 비정상적으로 낮음
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime, timezone

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')

def run_anomaly_detection():
    df = pd.read_csv(DATA_PATH)
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df['Published_At'] = pd.to_datetime(df['Published_At'], utc=True)
    df['Age_Days'] = (NOW - df['Published_At']).dt.days.clip(lower=1)
    df['VPD']      = df['Views'] / df['Age_Days']
    df['Like_Rate'] = df['Likes'] / (df['Views'] + 1)
    df['Log_Views'] = np.log1p(df['Views'])
    df['Log_VPD']   = np.log1p(df['VPD'])
    df['Log_Likes'] = np.log1p(df['Likes'])
    df['Log_LR']    = np.log1p(df['Like_Rate'])

    # Isolation Forest 피처
    features = ['Log_VPD', 'Log_LR', 'Log_Views', 'Log_Likes']
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    df['Anomaly']    = iso.fit_predict(X_scaled)   # -1 = 이상치
    df['Anomaly_Score'] = iso.score_samples(X_scaled)  # 낮을수록 이상

    n_anomaly = (df['Anomaly'] == -1).sum()
    print(f"탐지된 이상 영상: {n_anomaly}개 ({n_anomaly/len(df)*100:.1f}%)")

    # Z-score 기반 추가 분류
    df['LR_zscore'] = stats.zscore(df['Like_Rate'].fillna(0))
    df['VPD_zscore'] = stats.zscore(df['VPD'].fillna(0))

    # 의심 유형 분류
    df['Suspect_Type'] = 'normal'
    # Type 1: 조회수는 극히 높은데 좋아요 비율 비정상적으로 낮음 → 조회수 구매 의심
    mask1 = (df['VPD_zscore'] > 3) & (df['LR_zscore'] < -1)
    df.loc[mask1, 'Suspect_Type'] = '조회수 조작 의심'
    # Type 2: 좋아요 비율 극히 높음 → 좋아요 구매 의심
    mask2 = (df['LR_zscore'] > 4) & (df['VPD_zscore'] < 1)
    df.loc[mask2, 'Suspect_Type'] = '좋아요 조작 의심'
    # Type 3: 전반적 이상
    mask3 = (df['Anomaly'] == -1) & (df['Suspect_Type'] == 'normal')
    df.loc[mask3, 'Suspect_Type'] = '복합 이상'

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('이상치 탐지: 조회수/좋아요 조작 의심 영상 식별', fontsize=15, fontweight='bold', y=1.01)

    # [1] 이상치 산점도 (조회수 vs 좋아요 비율)
    colors_s = {'normal': '#aaaaaa55', '조회수 조작 의심': '#e9456088',
                '좋아요 조작 의심': '#f5a62388', '복합 이상': '#8b000088'}
    for stype, col in colors_s.items():
        sub = df[df['Suspect_Type'] == stype]
        axes[0,0].scatter(sub['Log_VPD'], sub['Log_LR'], c=col, s=15,
                          label=f'{stype} ({len(sub)}개)', zorder=2 if stype!='normal' else 1)
    axes[0,0].set_xlabel('log(Views/Day)')
    axes[0,0].set_ylabel('log(좋아요 비율)')
    axes[0,0].set_title('이상치 탐지 결과 (Isolation Forest)')
    axes[0,0].legend(fontsize=9, markerscale=2)

    # [2] Anomaly Score 분포
    axes[0,1].hist(df[df['Anomaly']==1]['Anomaly_Score'], bins=40,
                   alpha=0.7, color='steelblue', label='정상', density=True)
    axes[0,1].hist(df[df['Anomaly']==-1]['Anomaly_Score'], bins=20,
                   alpha=0.7, color='#e94560', label='이상', density=True)
    axes[0,1].set_xlabel('Isolation Forest Score')
    axes[0,1].set_title('Anomaly Score 분포\n(낮을수록 이상 가능성↑)')
    axes[0,1].legend()

    # [3] 이상치 카테고리 분포
    anom_by_kw = df[df['Anomaly']==-1]['Keyword'].value_counts()
    axes[1,0].bar(anom_by_kw.index, anom_by_kw.values,
                  color=plt.cm.Reds(np.linspace(0.4,0.9,len(anom_by_kw))),
                  edgecolor='white')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylabel('이상치 영상 수')
    axes[1,0].set_title('카테고리별 이상치 분포')

    # [4] 조회수 조작 의심 TOP 10
    suspects = df[df['Suspect_Type'] != 'normal'].nlargest(10, 'VPD')
    axes[1,1].barh(range(len(suspects)), suspects['VPD'],
                   color=['#e94560' if t=='조회수 조작 의심' else '#f5a623' if t=='좋아요 조작 의심' else '#8b0000'
                          for t in suspects['Suspect_Type']],
                   alpha=0.85, edgecolor='white')
    axes[1,1].set_yticks(range(len(suspects)))
    axes[1,1].set_yticklabels([f"{row['Title'][:18]}..." for _, row in suspects.iterrows()], fontsize=8)
    axes[1,1].set_xlabel('Views/Day')
    axes[1,1].set_title('의심 영상 TOP 10 (Views/Day 기준)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_06_anomaly_detection.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 이상치 제거 후 분석이 달라지는지 확인
    df_clean = df[df['Anomaly'] != -1]
    from scipy import stats as sp
    feat_views = df_clean[df_clean['Suspect_Type']=='normal']
    print(f"\n이상치 제거 후: {len(df_clean)}개 영상")

    # 저장
    df[['Video_ID','Title','Channel','Views','VPD','Like_Rate','Anomaly','Anomaly_Score','Suspect_Type']]\
        .to_csv(os.path.join(BASE_DIR,'../anomaly_report.csv'), index=False, encoding='utf-8-sig')
    print("anomaly_report.csv 저장 완료")

if __name__ == '__main__':
    run_anomaly_detection()
