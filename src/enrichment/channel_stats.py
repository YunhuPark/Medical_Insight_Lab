"""
Phase 1-A: YouTube API로 채널별 구독자 수 수집
50개씩 배치 요청 → 약 19회 API 호출
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
API_KEY = os.getenv('YOUTUBE_API_KEY')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../src/3_medical_platinum_final.csv')
OUT_PATH  = os.path.join(BASE_DIR, '../../src/channel_stats.csv')

def collect_channel_stats():
    df = pd.read_csv(DATA_PATH)
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # 채널명 → 검색으로 채널ID 확보
    channels = df['Channel'].unique()
    print(f"수집 대상 채널: {len(channels)}개")

    results = []
    for i in range(0, len(channels), 50):
        batch = channels[i:i+50]
        try:
            # 채널 검색
            for ch_name in batch:
                try:
                    search_res = youtube.search().list(
                        q=ch_name, type='channel', part='id', maxResults=1
                    ).execute()
                    if not search_res.get('items'):
                        results.append({'Channel': ch_name, 'Subscribers': None, 'Total_Videos': None})
                        continue
                    ch_id = search_res['items'][0]['id']['channelId']

                    # 채널 통계
                    ch_res = youtube.channels().list(
                        id=ch_id, part='statistics,snippet'
                    ).execute()
                    if ch_res.get('items'):
                        stats = ch_res['items'][0]['statistics']
                        results.append({
                            'Channel': ch_name,
                            'Channel_ID': ch_id,
                            'Subscribers': int(stats.get('subscriberCount', 0)),
                            'Total_Views': int(stats.get('viewCount', 0)),
                            'Total_Videos': int(stats.get('videoCount', 0)),
                        })
                    else:
                        results.append({'Channel': ch_name, 'Subscribers': None})
                    time.sleep(0.1)
                except Exception as e:
                    results.append({'Channel': ch_name, 'Subscribers': None})

        except Exception as e:
            print(f"배치 오류: {e}")

        print(f"  진행: {min(i+50, len(channels))}/{len(channels)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {OUT_PATH} ({len(out_df)}개 채널)")
    return out_df

if __name__ == '__main__':
    collect_channel_stats()
