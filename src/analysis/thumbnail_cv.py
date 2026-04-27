"""
Phase 3-A: 썸네일 컴퓨터 비전 분석
- 빨간색 비율 (경고/공포 색상)
- 얼굴 유무 (친밀감)
- 밝기/대비 (시각적 임팩트)
- 텍스트 면적 추정 (고채도 흰/노란 영역)
→ 각 지표와 조회수 상관관계 분석
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

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV 미설치 — pip install opencv-python")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '../3_medical_platinum_final.csv')
THUMB_DIR  = os.path.join(BASE_DIR, '../../thumbnails')
SAVE_DIR   = os.path.join(BASE_DIR, '../../results')
# 한글 경로 문제로 ASCII 경로 사용
FACE_XML = 'C:/Temp/haarcascade_frontalface_default.xml'

def analyze_thumbnail(img_path):
    if not CV2_AVAILABLE:
        return {}
    # 한글 경로 지원: imdecode 사용
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return {}
    if img is None:
        return {}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img.shape[:2]
    total   = h * w

    # 1. 빨간색 비율 (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0,80,80]),   np.array([10,255,255]))
    red2 = cv2.inRange(hsv, np.array([160,80,80]), np.array([180,255,255]))
    red_ratio = (cv2.countNonZero(red1) + cv2.countNonZero(red2)) / total

    # 2. 얼굴 탐지 (Haar Cascade)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_XML)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    has_face = int(len(faces) > 0)
    n_faces  = len(faces)

    # 3. 밝기 (V 채널 평균)
    brightness = hsv[:,:,2].mean() / 255

    # 4. 대비 (표준편차)
    contrast = gray.std() / 255

    # 5. 채도 (S 채널 평균)
    saturation = hsv[:,:,1].mean() / 255

    # 6. 텍스트 추정 (고채도+고밝기 흰/노랑 영역)
    text_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,50,255]))   # 흰색
    text_mask2= cv2.inRange(hsv, np.array([20,150,200]), np.array([35,255,255])) # 노란색
    text_ratio = (cv2.countNonZero(text_mask) + cv2.countNonZero(text_mask2)) / total

    # 7. 엣지 밀도 (정보량)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.countNonZero(edges) / total

    return {
        'red_ratio':   red_ratio,
        'has_face':    has_face,
        'n_faces':     n_faces,
        'brightness':  brightness,
        'contrast':    contrast,
        'saturation':  saturation,
        'text_ratio':  text_ratio,
        'edge_density': edge_density,
    }

def run_thumbnail_analysis():
    df = pd.read_csv(DATA_PATH)

    # 썸네일 존재 확인
    thumb_files = set(os.listdir(THUMB_DIR)) if os.path.exists(THUMB_DIR) else set()
    df['Thumb_Path'] = df['Video_ID'].apply(
        lambda vid: os.path.join(THUMB_DIR, f"{vid}.jpg")
        if f"{vid}.jpg" in thumb_files else None
    )
    df_thumb = df[df['Thumb_Path'].notna()].copy()
    print(f"분석 가능 썸네일: {len(df_thumb)}개 / {len(df)}개")

    if len(df_thumb) == 0:
        print("썸네일을 먼저 다운로드하세요: python src/enrichment/thumbnail_downloader.py")
        return

    # 분석 실행
    results = []
    for i, (_, row) in enumerate(df_thumb.iterrows()):
        metrics = analyze_thumbnail(row['Thumb_Path'])
        if metrics:
            metrics['Video_ID'] = row['Video_ID']
            metrics['Views']    = row['Views']
            metrics['Keyword']  = row['Keyword']
            results.append(metrics)
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(df_thumb)} 완료")

    result_df = pd.DataFrame(results)
    if result_df.empty or 'Views' not in result_df.columns:
        print(f"분석된 썸네일 없음 (결과 0개). OpenCV 이미지 읽기 실패 가능성.")
        # 더미 분석: 색상 통계만이라도 저장
        return
    result_df['Log_Views'] = np.log1p(result_df['Views'])

    # 저장
    result_df.to_csv(os.path.join(BASE_DIR, '../thumbnail_features.csv'),
                     index=False, encoding='utf-8-sig')

    # 시각화
    metrics_list = ['red_ratio','has_face','brightness','contrast','saturation','text_ratio','edge_density']
    metrics_kor  = {'red_ratio':'빨간색 비율','has_face':'얼굴 유무','brightness':'밝기',
                    'contrast':'대비','saturation':'채도','text_ratio':'텍스트 비율','edge_density':'엣지 밀도'}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes_flat = axes.flatten()

    corr_results = []
    for i, m in enumerate(metrics_list):
        ax = axes_flat[i]
        r, p = stats.spearmanr(result_df[m], result_df['Log_Views'])
        corr_results.append((m, r, p))
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        ax.scatter(result_df[m], result_df['Log_Views'], alpha=0.3, s=10,
                   color='#e94560' if (p<0.05 and r>0) else '#0f3460' if (p<0.05 and r<0) else '#aaaaaa')
        z = np.polyfit(result_df[m], result_df['Log_Views'], 1)
        xp = np.linspace(result_df[m].min(), result_df[m].max(), 100)
        ax.plot(xp, np.polyval(z, xp), 'r-', lw=2, alpha=0.8)
        ax.set_xlabel(metrics_kor.get(m, m))
        ax.set_ylabel('log(조회수)')
        ax.set_title(f'r={r:.3f} {sig}', fontsize=11, fontweight='bold',
                     color='#e94560' if p<0.05 else '#888')

    # [8] 요약 막대
    ax = axes_flat[7]
    corr_df = pd.DataFrame(corr_results, columns=['metric','r','p'])
    corr_df['sig'] = corr_df['p'] < 0.05
    corr_df = corr_df.sort_values('r', ascending=True)
    colors_bar = ['#e94560' if v>0 else '#0f3460' for v in corr_df['r']]
    bars = ax.barh([metrics_kor.get(m,m) for m in corr_df['metric']],
                   corr_df['r'], color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axvline(0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Spearman r')
    ax.set_title('썸네일 피처 vs 조회수\n상관관계 요약', fontweight='bold')

    plt.suptitle('썸네일 컴퓨터 비전 분석: 시각적 요소와 조회수 관계',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_07_thumbnail_cv.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n[썸네일 피처 상관관계 요약]")
    for m, r, p in sorted(corr_results, key=lambda x: abs(x[1]), reverse=True):
        sig = '★ 유의' if p<0.05 else '비유의'
        print(f"  {metrics_kor.get(m,m):12}: r={r:+.3f}, p={p:.4f} {sig}")

if __name__ == '__main__':
    run_thumbnail_analysis()
