"""
전체 파이프라인 실행 마스터 스크립트
실행: venv/Scripts/python run_all.py
"""
import subprocess, sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = os.path.join(BASE, 'venv/Scripts/python')

PIPELINE = [
    # ── Phase 1: 데이터 수집 (API + 다운로드)
    ("Phase 1-A: 채널 구독자 수 수집",   "src/enrichment/channel_stats.py"),
    ("Phase 1-B: 댓글 수집",             "src/enrichment/comment_collector.py"),
    ("Phase 1-C: 썸네일 다운로드",        "src/enrichment/thumbnail_downloader.py"),
    # ── Phase 2: 통계 분석
    ("Phase 2-A: 바이럴 속도 분석",       "src/analysis/velocity_analysis.py"),
    ("Phase 2-C: 채널 고정효과 회귀",     "src/analysis/fixed_effects.py"),
    ("Phase 2-D: PSM 인과 추정",          "src/analysis/psm_analysis.py"),
    ("Phase 2-E: Dunn's Post-hoc Test",   "src/analysis/posthoc_tests.py"),
    ("Phase 2-F: COVID-19 자연실험",      "src/analysis/covid_experiment.py"),
    ("Phase 2-G: 이상치 탐지",            "src/analysis/anomaly_detection.py"),
    # ── Phase 3: 고급 분석
    ("Phase 3-A: 썸네일 CV 분석",         "src/analysis/thumbnail_cv.py"),
    ("Phase 3-B: 댓글 감성 분석",         "src/analysis/comment_sentiment.py"),
    ("Phase 3-C: KoBERT 임베딩 분석",     "src/analysis/kobert_analysis.py"),
]

def run_step(name, script_path):
    full_path = os.path.join(BASE, script_path)
    print(f"\n{'='*60}")
    print(f"▶  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PY, full_path], cwd=BASE, capture_output=False,
                            encoding='utf-8', errors='replace')
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"{'─'*40}")
    print(f"  {status} | {elapsed:.1f}초 소요")
    return result.returncode == 0

if __name__ == '__main__':
    print("Medical Insight Lab — 전체 파이프라인 실행")
    print(f"총 {len(PIPELINE)}개 단계\n")

    results = []
    for name, script in PIPELINE:
        ok = run_step(name, script)
        results.append((name, ok))

    print(f"\n{'='*60}")
    print("실행 결과 요약")
    print(f"{'='*60}")
    for name, ok in results:
        icon = "OK  " if ok else "FAIL"
        print(f"  [{icon}] {name}")

    ok_count = sum(1 for _, ok in results if ok)
    print(f"\n완료: {ok_count}/{len(results)}")
    print("\nStreamlit 대시보드 실행:")
    print(f"  {PY} -m streamlit run dashboard/app.py")
