"""
COVID 이전(2016-2019) 의료 유튜브 영상 추가 수집
DiD 분석의 pre-COVID 샘플 부족 문제 해결
현재: pre-COVID 공포 영상 14개 → 목표: 100개 이상
API 비용: 키워드 22개 × 4년 × 2페이지 = ~176유닛
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
API_KEY = os.getenv('YOUTUBE_API_KEY')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../3_medical_platinum_final.csv')
OUT_PATH  = os.path.join(BASE_DIR, '../precovid_videos.csv')

KEYWORDS = [
    '당뇨', '고혈압', '암', '갑상선', '관절염', '허리디스크', '전립선',
    '치매', '뇌졸중', '심장', '폐렴', '당뇨병', '비만', '탈모', '위암',
    '대장암', '유방암', '간염', '신장', '골다공증', '요실금', '아토피'
]

# COVID 이전 연도별 수집 구간
PERIODS = [
    ('2016-01-01T00:00:00Z', '2016-12-31T23:59:59Z'),
    ('2017-01-01T00:00:00Z', '2017-12-31T23:59:59Z'),
    ('2018-01-01T00:00:00Z', '2018-12-31T23:59:59Z'),
    ('2019-01-01T00:00:00Z', '2019-12-31T23:59:59Z'),
]

def parse_duration(duration_str):
    import re
    h = int(re.search(r'(\d+)H', duration_str).group(1)) if 'H' in duration_str else 0
    m = int(re.search(r'(\d+)M', duration_str).group(1)) if 'M' in duration_str else 0
    s = int(re.search(r'(\d+)S', duration_str).group(1)) if 'S' in duration_str else 0
    return h*3600 + m*60 + s

def collect():
    existing_df = pd.read_csv(DATA_PATH)
    existing_ids = set(existing_df['Video_ID'].unique())

    # 이미 수집된 precovid 데이터
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        existing_ids |= set(prev['Video_ID'].unique())
        results = prev.to_dict('records')
        print(f"기존 수집: {len(prev)}개")
    else:
        results = []

    youtube = build('youtube', 'v3', developerKey=API_KEY)
    total_new = 0

    for start, end in PERIODS:
        year = start[:4]
        print(f"\n=== {year}년 수집 ===")
        yr_count = 0

        for kw in KEYWORDS:
            try:
                resp = youtube.search().list(
                    q=f'{kw} 의료 건강',
                    type='video',
                    part='id',
                    maxResults=20,
                    publishedAfter=start,
                    publishedBefore=end,
                    relevanceLanguage='ko',
                    regionCode='KR',
                    order='viewCount'
                ).execute()

                vid_ids = [item['id']['videoId'] for item in resp.get('items', [])
                           if item['id']['videoId'] not in existing_ids]

                if not vid_ids:
                    continue

                # 상세 정보 수집
                detail = youtube.videos().list(
                    id=','.join(vid_ids),
                    part='snippet,statistics,contentDetails'
                ).execute()

                for item in detail.get('items', []):
                    vid_id = item['id']
                    snippet = item['snippet']
                    stats   = item.get('statistics', {})
                    content = item.get('contentDetails', {})

                    views = int(stats.get('viewCount', 0))
                    if views < 100:
                        continue

                    duration_str = content.get('duration', 'PT0S')
                    try:
                        duration_sec = parse_duration(duration_str)
                    except:
                        duration_sec = 0

                    results.append({
                        'Video_ID':     vid_id,
                        'Keyword':      kw,
                        'Title':        snippet.get('title', ''),
                        'Channel':      snippet.get('channelTitle', ''),
                        'Views':        views,
                        'Likes':        int(stats.get('likeCount', 0)),
                        'Published_At': snippet.get('publishedAt', ''),
                        'Duration_Sec': duration_sec,
                        'Description':  snippet.get('description', '')[:500],
                        'Source_Type':  'PreCovid_Supplement',
                    })
                    existing_ids.add(vid_id)
                    yr_count += 1
                    total_new += 1

                time.sleep(0.15)

            except HttpError as e:
                if 'quotaExceeded' in str(e):
                    print(f"  할당량 초과 — {total_new}개 수집 후 중단")
                    pd.DataFrame(results).to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
                    return
                time.sleep(1)

        print(f"  {year}년 수집: {yr_count}개")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n=== 완료 ===")
    print(f"총 수집: {len(out_df)}개 (신규 {total_new}개)")
    print(f"저장: {OUT_PATH}")

if __name__ == '__main__':
    collect()
