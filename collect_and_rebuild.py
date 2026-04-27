"""
데이터 완전 수집 → 전체 분석 재실행 마스터 스크립트
실행: venv/Scripts/python collect_and_rebuild.py

API 할당량 리셋 후 실행하세요 (매일 자정 태평양 시간 = 한국 오후 4~5시)
예상 API 사용량: ~2,600유닛 (일 할당량 10,000의 26%)
"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE, 'venv', 'Scripts', 'python')

def run(script, label):
    print(f"\n{'='*60}")
    print(f"[{label}] 실행 중...")
    print('='*60)
    result = subprocess.run([PYTHON, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[경고] {label} 실패 또는 할당량 초과 — 계속 진행")
    return result.returncode == 0

def check_progress():
    import pandas as pd
    ch = pd.read_csv(os.path.join(BASE, 'src', 'channel_stats.csv'))
    subs_ok = ch['Subscribers'].notna().sum()
    print(f"\n현재 상태:")
    print(f"  구독자 데이터: {subs_ok}/{len(ch)} ({subs_ok/len(ch)*100:.1f}%)")
    comments_path = os.path.join(BASE, 'src', 'comments.csv')
    if os.path.exists(comments_path):
        comments = pd.read_csv(comments_path)
        print(f"  댓글 수:       {len(comments)}개 ({comments['Video_ID'].nunique()}개 영상)")
        return subs_ok, len(comments)
    print(f"  댓글 수:       0개")
    return subs_ok, 0

def rebuild_dataset():
    """precovid_videos.csv의 미병합 데이터를 4_medical_full_dataset.csv에 추가"""
    import pandas as pd
    dataset_path  = os.path.join(BASE, 'src', '4_medical_full_dataset.csv')
    precovid_path = os.path.join(BASE, 'src', 'precovid_videos.csv')

    if not os.path.exists(precovid_path):
        print("  precovid_videos.csv 없음 — 건너뜀")
        return

    df = pd.read_csv(dataset_path)
    pc = pd.read_csv(precovid_path)

    existing_ids = set(df['Video_ID'])
    new_rows = pc[~pc['Video_ID'].isin(existing_ids)].copy()

    if len(new_rows) == 0:
        print("  새로운 precovid 데이터 없음 — 건너뜀")
        return

    new_rows['Medical_Score'] = 0.5
    new_rows['Type'] = 'General'
    # 컬럼을 메인 데이터셋 순서에 맞춤 (없는 컬럼은 제거)
    new_rows = new_rows[[c for c in df.columns if c in new_rows.columns]]
    for c in df.columns:
        if c not in new_rows.columns:
            new_rows[c] = None

    merged = pd.concat([df, new_rows[df.columns]], ignore_index=True)
    merged.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    print(f"  {len(new_rows)}개 추가 → 총 {len(merged)}개 영상")

if __name__ == '__main__':
    print("=" * 60)
    print("데이터 완전 수집 + 분석 재실행 마스터 스크립트")
    print("=" * 60)

    subs_before, comments_before = check_progress()

    # 1단계: 구독자 수 완전 수집
    run(os.path.join(BASE, 'src', 'enrichment', 'collect_all_subscribers.py'),
        "구독자 수 전체 수집 (921개 채널)")

    # 2단계: COVID 이전 영상 보완 수집 (DiD 불균형 해소)
    run(os.path.join(BASE, 'src', 'enrichment', 'collect_precovid_videos.py'),
        "COVID 이전(2016-2019) 영상 보완 수집")

    # 2-B단계: precovid 데이터를 메인 데이터셋에 병합
    print(f"\n{'='*60}")
    print("[데이터셋 재병합] precovid_videos → 4_medical_full_dataset")
    print('='*60)
    rebuild_dataset()

    # 3단계: 댓글 확대 수집
    run(os.path.join(BASE, 'src', 'enrichment', 'collect_all_comments.py'),
        "댓글 확대 수집 (2,512개 영상)")

    subs_after, comments_after = check_progress()
    print(f"\n  구독자: {subs_before} → {subs_after}개")
    print(f"  댓글:   {comments_before} → {comments_after}개")

    # 3단계: 전체 분석 재실행
    analyses = [
        ('src/analysis/velocity_analysis.py',    '바이럴 속도 분석'),
        ('src/analysis/posthoc_tests.py',         'Dunn Post-hoc 검정'),
        ('src/analysis/fixed_effects.py',         '채널 고정효과 회귀'),
        ('src/analysis/psm_analysis.py',          'PSM 인과 추정'),
        ('src/analysis/covid_experiment.py',      'COVID DiD'),
        ('src/analysis/anomaly_detection.py',     '이상치 탐지'),
        ('src/analysis/subscriber_analysis.py',   '구독자 규모 분석'),
        ('src/analysis/thumbnail_cv.py',          '썸네일 CV'),
        ('src/analysis/comment_sentiment.py',     '댓글 감성 분석'),
        ('src/analysis/kobert_analysis.py',       'KoBERT 임베딩'),
        ('src/analysis/type_fear_interaction.py', 'Type × Fear 상호작용'),
        ('src/analysis/shorts_analysis.py',       'Shorts 분리 분석'),
    ]
    for script, label in analyses:
        run(os.path.join(BASE, script), label)

    # 4단계: 노트북 재생성
    print("\n[노트북 재생성]")
    run(os.path.join(BASE, 'notebooks', 'generate_notebook_v2.py'), '노트북 생성')

    print("\n[노트북 실행]")
    result = subprocess.run([
        PYTHON, '-m', 'nbconvert', '--to', 'notebook', '--execute',
        os.path.join(BASE, 'notebooks', 'medical_youtube_final.ipynb'),
        '--output', 'medical_youtube_final_executed.ipynb',
        '--ExecutePreprocessor.timeout=600',
        '--ExecutePreprocessor.kernel_name=python3'
    ], capture_output=False)

    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    check_progress()
    print("=" * 60)
