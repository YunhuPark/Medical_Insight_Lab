"""
④ 조회수 패널 데이터 수집 스크립트
- 실행할 때마다 현재 조회수를 날짜별로 스냅샷 저장
- 2회 이상 실행하면 시간별 조회수 변화율 분석 가능
- 권장: 지금 1회, 한 달 뒤 1회 실행

사용법:
    python src/enrichment/collect_panel_snapshot.py
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
API_KEY = os.getenv('YOUTUBE_API_KEY')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '../4_medical_full_dataset.csv')
PANEL_DIR  = os.path.join(BASE_DIR, '../panel_snapshots')
os.makedirs(PANEL_DIR, exist_ok=True)

BATCH_SIZE = 50  # YouTube API는 한 번에 최대 50개 video ID 조회 가능

def collect_snapshot():
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    out_path = os.path.join(PANEL_DIR, f'views_{today}.csv')

    if os.path.exists(out_path):
        print(f"오늘({today}) 스냅샷이 이미 존재합니다: {out_path}")
        return

    df = pd.read_csv(DATA_PATH)
    video_ids = df['Video_ID'].dropna().unique().tolist()
    print(f"수집 대상: {len(video_ids)}개 영상")
    print(f"저장 경로: {out_path}")

    youtube = build('youtube', 'v3', developerKey=API_KEY)
    results = []

    for i in range(0, len(video_ids), BATCH_SIZE):
        batch = video_ids[i:i + BATCH_SIZE]
        try:
            resp = youtube.videos().list(
                part='statistics',
                id=','.join(batch),
                maxResults=BATCH_SIZE
            ).execute()

            for item in resp.get('items', []):
                stats = item.get('statistics', {})
                results.append({
                    'Video_ID':      item['id'],
                    'Snapshot_Date': today,
                    'Views':         int(stats.get('viewCount', 0)),
                    'Likes':         int(stats.get('likeCount', 0)),
                    'Comments':      int(stats.get('commentCount', 0)),
                })

            if (i // BATCH_SIZE + 1) % 10 == 0:
                print(f"  {i + BATCH_SIZE}/{len(video_ids)} 완료")

            time.sleep(0.05)

        except HttpError as e:
            if 'quotaExceeded' in str(e):
                print(f"\n할당량 초과 — {i}개까지 수집 후 중단")
                break
            print(f"오류: {e}")
            time.sleep(0.2)

    snap_df = pd.DataFrame(results)
    snap_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n=== 스냅샷 저장 완료 ===")
    print(f"수집된 영상: {len(snap_df)}개")
    print(f"파일: {out_path}")

    # 기존 스냅샷과 비교 가능 여부 확인
    existing = [f for f in os.listdir(PANEL_DIR) if f.startswith('views_') and f != f'views_{today}.csv']
    if existing:
        print(f"\n기존 스냅샷 {len(existing)}개 발견: {existing}")
        print("→ panel_analysis.py 실행하면 조회수 변화율 분석 가능")
    else:
        print("\n첫 번째 스냅샷입니다. 한 달 뒤 재실행하면 패널 데이터 완성.")

def analyze_panel():
    """2개 이상의 스냅샷이 있을 때 변화율 분석"""
    snapshots = sorted([f for f in os.listdir(PANEL_DIR) if f.startswith('views_')])
    if len(snapshots) < 2:
        print("스냅샷이 2개 이상 필요합니다.")
        return

    df_list = []
    for snap in snapshots:
        df = pd.read_csv(os.path.join(PANEL_DIR, snap))
        df_list.append(df)

    # 첫 번째 vs 마지막 스냅샷 비교
    first = df_list[0].set_index('Video_ID')[['Views']].rename(columns={'Views': 'Views_T1'})
    last  = df_list[-1].set_index('Video_ID')[['Views', 'Snapshot_Date']].rename(
        columns={'Views': 'Views_T2', 'Snapshot_Date': 'Date_T2'})

    panel = first.join(last, how='inner')
    panel['Views_Growth'] = panel['Views_T2'] - panel['Views_T1']
    panel['Days_Between'] = (
        pd.to_datetime(panel['Date_T2']) -
        pd.to_datetime(df_list[0]['Snapshot_Date'].iloc[0])
    ).dt.days
    panel['VPD_Growth'] = panel['Views_Growth'] / panel['Days_Between'].clip(lower=1)

    # 원본 데이터와 병합 (Has_Fear 등)
    base = pd.read_csv(DATA_PATH)[['Video_ID','Has_Fear','Type','Keyword']] if os.path.exists(DATA_PATH) else None
    if base is not None:
        panel = panel.reset_index().merge(base, on='Video_ID', how='left')

    out_path = os.path.join(PANEL_DIR, 'panel_analysis.csv')
    panel.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"패널 분석 저장: {out_path}")
    print(f"기간: {snapshots[0]} ~ {snapshots[-1]}")
    print(f"영상 수: {len(panel)}개")
    if 'Has_Fear' in panel.columns:
        g = panel.groupby('Has_Fear')['VPD_Growth'].median()
        print(f"\n공포 키워드별 일평균 조회수 증가:")
        print(f"  미포함: {g.get(0, 0):,.1f} VPD/day")
        print(f"  포함:   {g.get(1, 0):,.1f} VPD/day")
    return panel

if __name__ == '__main__':
    collect_snapshot()
