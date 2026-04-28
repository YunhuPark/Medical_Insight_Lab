"""
⑤ TF-IDF 데이터 기반 공포 키워드 추출 및 검증
- 선험적 키워드 목록의 한계를 보완
- TF-IDF로 공포/비공포 영상을 구분하는 핵심 단어 추출
- 기존 키워드 목록과 비교 검증
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import re

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')

PRIOR_FEAR_WORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

def tokenize_korean(text):
    """간단한 한국어 형태소 분리 (음절 n-gram 기반)"""
    text = re.sub(r'[^\w가-힣]', ' ', str(text))
    tokens = text.split()
    # 2~4글자 단어만 사용
    return [t for t in tokens if 2 <= len(t) <= 6]

def run_tfidf_analysis():
    df = pd.read_csv(DATA_PATH)
    df['Has_Fear_Prior'] = df['Title'].astype(str).apply(
        lambda x: int(any(k in x for k in PRIOR_FEAR_WORDS))
    )

    titles = df['Title'].astype(str).tolist()
    labels = df['Has_Fear_Prior'].values

    print(f"전체 영상: {len(df)}개")
    print(f"기존 키워드 기준 공포 영상: {labels.sum()}개 ({labels.mean()*100:.1f}%)")

    # TF-IDF 벡터화 (문자 n-gram 포함)
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=tokenize_korean,
        token_pattern=None,
        max_features=500,
        min_df=5,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(titles)
    feature_names = vectorizer.get_feature_names_out()

    # 로지스틱 회귀로 구분력 높은 단어 추출
    lr = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    lr.fit(X, labels)

    coef = lr.coef_[0]
    top_fear_idx    = np.argsort(coef)[::-1][:30]
    top_nonfear_idx = np.argsort(coef)[:30]

    top_fear_words    = [(feature_names[i], coef[i]) for i in top_fear_idx]
    top_nonfear_words = [(feature_names[i], coef[i]) for i in top_nonfear_idx]

    # 교차검증 정확도
    cv_scores = cross_val_score(lr, X, labels, cv=5, scoring='roc_auc')
    print(f"\nTF-IDF 분류 AUC (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 신규 발견 키워드 (기존 목록에 없는 것)
    new_fear_words = [w for w, _ in top_fear_words if w not in PRIOR_FEAR_WORDS]
    print(f"\n[TF-IDF 공포 예측 상위 30개]")
    for w, c in top_fear_words:
        tag = "" if w in PRIOR_FEAR_WORDS else " ★신규"
        print(f"  {w}: {c:+.3f}{tag}")

    print(f"\n[신규 발견 공포 키워드 (기존 목록 미포함)]: {new_fear_words[:15]}")

    # 데이터 기반 Has_Fear 재정의
    extended_fear = PRIOR_FEAR_WORDS + [w for w in new_fear_words[:10] if len(w) >= 2]
    df['Has_Fear_Extended'] = df['Title'].astype(str).apply(
        lambda x: int(any(k in x for k in extended_fear))
    )
    added = df['Has_Fear_Extended'].sum() - df['Has_Fear_Prior'].sum()
    print(f"\n확장 키워드 적용 시 공포 영상: {df['Has_Fear_Extended'].sum()}개 (+{added}개)")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # [1] 공포 예측 상위 20개 단어
    words_f = [w for w, _ in top_fear_words[:20]]
    coefs_f = [c for _, c in top_fear_words[:20]]
    colors_f = ['#e94560' if w in PRIOR_FEAR_WORDS else '#ff8c42' for w in words_f]
    axes[0].barh(words_f[::-1], coefs_f[::-1], color=colors_f[::-1])
    axes[0].axvline(0, color='white', lw=0.8, linestyle='--')
    axes[0].set_title('TF-IDF 공포 예측 상위 단어\n(주황=신규 발견)', color='white')
    axes[0].set_xlabel('로지스틱 회귀 계수', color='white')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#0f3460')
    axes[0].spines[:].set_color('#333')

    # [2] 기존 vs 확장 키워드 적용 영상 수 비교
    categories = ['기존 키워드\n(선험적)', '확장 키워드\n(TF-IDF 보완)']
    counts = [df['Has_Fear_Prior'].sum(), df['Has_Fear_Extended'].sum()]
    bars = axes[1].bar(categories, counts, color=['#e94560', '#ff8c42'], width=0.5)
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     str(cnt), ha='center', color='white', fontsize=12, fontweight='bold')
    axes[1].set_title('키워드 방식별 공포 영상 분류 수', color='white')
    axes[1].set_ylabel('영상 수', color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#0f3460')
    axes[1].spines[:].set_color('#333')

    fig.patch.set_facecolor('#16213e')
    plt.suptitle('TF-IDF 데이터 기반 공포 키워드 검증\n(선험적 목록의 한계 보완)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'v2_14_tfidf_keyword_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#16213e')
    print(f"\n저장: {out_path}")

    # 확장 키워드 목록 저장
    kw_path = os.path.join(BASE_DIR, '../fear_keywords_extended.txt')
    with open(kw_path, 'w', encoding='utf-8') as f:
        f.write("# 확장 공포 키워드 (기존 + TF-IDF 신규)\n")
        for w in extended_fear:
            f.write(w + '\n')
    print(f"확장 키워드 저장: {kw_path}")

    return extended_fear, new_fear_words

if __name__ == '__main__':
    run_tfidf_analysis()
