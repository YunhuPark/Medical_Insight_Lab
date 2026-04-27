"""
Phase 3-C: KoBERT 제목 임베딩 분석
- 사전 매칭 없이 제목의 의미 자체를 벡터화
- 유사한 제목 클러스터링 (UMAP + K-Means)
- 조회수 높은 영역 vs 낮은 영역 의미 패턴 분석
- 공포/비공포 제목 임베딩 공간 분리도 측정
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../3_medical_platinum_final.csv')
SAVE_DIR  = os.path.join(BASE_DIR, '../../results')

FEAR_KEYWORDS = [
    '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
    '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
    '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
]

def get_kobert_embeddings(titles):
    """KoBERT 또는 multilingual-e5로 제목 임베딩"""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        print("KoBERT 모델 로드 중...")
        # multilingual-e5-small (경량, 한국어 지원)
        model_name = "intfloat/multilingual-e5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings = []
        batch_size = 32
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size].tolist()
            batch = ["query: " + t for t in batch]
            encoded = tokenizer(batch, max_length=64, padding=True,
                                truncation=True, return_tensors='pt')
            with torch.no_grad():
                output = model(**encoded)
                # Mean pooling
                attn = encoded['attention_mask'].unsqueeze(-1).float()
                emb = (output.last_hidden_state * attn).sum(1) / attn.sum(1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embeddings.append(emb.numpy())
            if (i//batch_size+1) % 5 == 0:
                print(f"  임베딩 {min(i+batch_size, len(titles))}/{len(titles)}")

        return np.vstack(embeddings)

    except Exception as e:
        print(f"BERT 모델 실패: {e}")
        print("TF-IDF 폴백 사용")
        return _tfidf_fallback(titles)

def _tfidf_fallback(titles):
    """BERT 불가 시 TF-IDF 대체"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    def extract_nouns(text):
        tokens = kiwi.tokenize(str(text))
        return ' '.join([t.form for t in tokens if t.tag in ['NNG','NNP'] and len(t.form)>1])
    processed = titles.apply(extract_nouns)
    vectorizer = TfidfVectorizer(max_features=500)
    return vectorizer.fit_transform(processed).toarray()

def run_kobert_analysis():
    df = pd.read_csv(DATA_PATH)
    df['Has_Fear']  = df['Title'].apply(lambda x: int(any(kw in str(x) for kw in FEAR_KEYWORDS)))
    df['Log_Views'] = np.log1p(df['Views'])
    df['View_Tier'] = pd.qcut(df['Views'], q=3, labels=['하위', '중위', '상위'])

    titles = df['Title']
    print(f"임베딩 대상: {len(titles)}개 제목")

    # 임베딩 생성
    emb_path = os.path.join(BASE_DIR, '../title_embeddings.npy')
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        print(f"캐시된 임베딩 로드: {embeddings.shape}")
    else:
        embeddings = get_kobert_embeddings(titles)
        np.save(emb_path, embeddings)
        print(f"임베딩 저장: {emb_path}")

    # PCA 2D 축소
    pca = PCA(n_components=50, random_state=42)
    emb_50d = pca.fit_transform(embeddings)
    print(f"PCA 50d 설명 분산: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    pca2 = PCA(n_components=2, random_state=42)
    emb_2d = pca2.fit_transform(embeddings)

    # 최적 K 탐색 (Elbow + Silhouette)
    sil_scores = []
    k_range = range(3, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb_50d)
        sil_scores.append(silhouette_score(emb_50d, labels, sample_size=500))

    best_k = k_range[np.argmax(sil_scores)]
    print(f"최적 클러스터 수: {best_k} (Silhouette={max(sil_scores):.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = km_final.fit_predict(emb_50d)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('KoBERT 제목 임베딩 분석: 의미 공간에서의 조회수 패턴',
                 fontsize=15, fontweight='bold', y=1.01)

    cmap_view = plt.cm.RdYlGn
    colors_cl  = plt.cm.tab10(np.linspace(0, 1, best_k))

    # [1] 조회수 기준 색상 임베딩 지도
    sc = axes[0,0].scatter(emb_2d[:,0], emb_2d[:,1],
                           c=df['Log_Views'], cmap=cmap_view, s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0,0], label='log(조회수)')
    axes[0,0].set_title('PCA 2D 임베딩 공간\n(색=조회수)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('PC1'); axes[0,0].set_ylabel('PC2')

    # [2] 클러스터 색상 임베딩 지도
    for cl in range(best_k):
        mask = df['Cluster'] == cl
        axes[0,1].scatter(emb_2d[mask,0], emb_2d[mask,1],
                          c=[colors_cl[cl]], s=8, alpha=0.6, label=f'Cluster {cl}')
    axes[0,1].set_title(f'K-Means 클러스터 (k={best_k})\nSilhouette={max(sil_scores):.3f}',
                         fontsize=12, fontweight='bold')
    axes[0,1].legend(fontsize=8, markerscale=2)

    # [3] 공포 키워드 분포
    has_fear = df['Has_Fear'].values
    axes[0,2].scatter(emb_2d[has_fear==0,0], emb_2d[has_fear==0,1],
                      c='steelblue', s=6, alpha=0.4, label='공포 미포함')
    axes[0,2].scatter(emb_2d[has_fear==1,0], emb_2d[has_fear==1,1],
                      c='#e94560', s=6, alpha=0.5, label='공포 포함')
    axes[0,2].set_title('공포 키워드 포함 여부\n임베딩 공간 분리도', fontsize=12, fontweight='bold')
    axes[0,2].legend(fontsize=9, markerscale=2)

    # [4] Silhouette 점수 (k 선택 근거)
    axes[1,0].plot(list(k_range), sil_scores, 'o-', color='#e94560', lw=2.5, markersize=8)
    axes[1,0].axvline(best_k, color='orange', lw=2, linestyle='--', label=f'최적 k={best_k}')
    axes[1,0].set_xlabel('클러스터 수 (k)')
    axes[1,0].set_ylabel('Silhouette Score')
    axes[1,0].set_title('최적 클러스터 수 선택', fontsize=12, fontweight='bold')
    axes[1,0].legend()

    # [5] 클러스터별 중앙값 조회수
    cl_views = df.groupby('Cluster')['Views'].median().sort_values(ascending=True)
    axes[1,1].barh(range(len(cl_views)),
                   cl_views.values/1e4,
                   color=[colors_cl[i] for i in cl_views.index],
                   edgecolor='white', alpha=0.85)
    for i, (cl, v) in enumerate(cl_views.items()):
        n = (df['Cluster']==cl).sum()
        axes[1,1].text(v/1e4+0.3, i, f'{v/1e4:.1f}만 (n={n})',
                       va='center', fontsize=9)
    axes[1,1].set_yticks(range(len(cl_views)))
    axes[1,1].set_yticklabels([f'Cluster {c}' for c in cl_views.index])
    axes[1,1].set_xlabel('중앙값 조회수 (만회)')
    axes[1,1].set_title('의미 클러스터별 중앙값 조회수', fontsize=12, fontweight='bold')

    # [6] 클러스터별 공포 키워드 비율
    cl_fear = df.groupby('Cluster')['Has_Fear'].mean() * 100
    axes[1,2].bar(range(len(cl_fear)), cl_fear.values,
                  color=[colors_cl[i] for i in cl_fear.index], edgecolor='white', alpha=0.85)
    axes[1,2].set_xticks(range(len(cl_fear)))
    axes[1,2].set_xticklabels([f'Cluster {c}' for c in cl_fear.index])
    axes[1,2].set_ylabel('공포 키워드 포함 비율 (%)')
    axes[1,2].set_title('클러스터별 공포 키워드 비율\n높을수록 공포 소구 클러스터', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_09_kobert_embedding.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 클러스터별 대표 제목
    print("\n[클러스터별 대표 제목]")
    for cl in range(best_k):
        cl_df = df[df['Cluster']==cl]
        top3  = cl_df.nlargest(3, 'Views')['Title'].values
        fear_pct = cl_df['Has_Fear'].mean()*100
        med_views = cl_df['Views'].median()/1e4
        print(f"\nCluster {cl} | 중앙조회수: {med_views:.1f}만 | 공포비율: {fear_pct:.1f}%")
        for t in top3:
            print(f"  - {t[:45]}")

if __name__ == '__main__':
    run_kobert_analysis()
