"""
전체 채널 구독자 수 완전 수집 (효율적 방법)
1단계: Video_ID → videos.list → channelId  (1유닛/50개)
2단계: channelId → channels.list → 구독자  (1유닛/50개)
총 API 비용: ~70유닛 (기존 방식 92,100유닛 대비 1,300배 절약)
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
API_KEY = os.getenv('YOUTUBE_API_KEY')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../3_medical_platinum_final.csv')
OUT_PATH  = os.path.join(BASE_DIR, '../channel_stats.csv')

def collect():
    df = pd.read_csv(DATA_PATH)
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # ── 기존 데이터 로드
    existing = {}
    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        for _, row in old.iterrows():
            if pd.notna(row.get('Channel_ID')) and pd.notna(row.get('Subscribers')):
                existing[row['Channel']] = row.to_dict()
    print(f"기존 수집된 채널: {len(existing)}개")

    # ── 1단계: Video_ID → Channel_ID 매핑
    vid_to_ch = {}
    video_ids = df['Video_ID'].unique().tolist()
    print(f"\n1단계: {len(video_ids)}개 영상에서 채널 ID 수집...")
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                id=','.join(batch),
                part='snippet'
            ).execute()
            for item in resp.get('items', []):
                vid_id = item['id']
                ch_id  = item['snippet']['channelId']
                ch_name = item['snippet']['channelTitle']
                vid_to_ch[ch_name] = ch_id
            if (i // 50 + 1) % 10 == 0:
                print(f"  {i+50}/{len(video_ids)} 완료")
            time.sleep(0.05)
        except HttpError as e:
            print(f"  오류 (batch {i//50}): {e}")
            time.sleep(2)

    print(f"  → 채널 ID 확보: {len(vid_to_ch)}개")

    # 모든 채널 목록
    all_channels = df['Channel'].unique().tolist()
    channels_needing_collection = [
        ch for ch in all_channels
        if ch not in existing or pd.isna(existing[ch].get('Subscribers'))
    ]
    print(f"\n구독자 수 수집 필요: {len(channels_needing_collection)}개")

    # ── 2단계: Channel_ID → 구독자 수
    ch_id_map = vid_to_ch  # channel_name -> channel_id
    ch_id_list = [ch_id_map[ch] for ch in channels_needing_collection if ch in ch_id_map]
    ch_names_with_id = [ch for ch in channels_needing_collection if ch in ch_id_map]
    ch_names_without_id = [ch for ch in channels_needing_collection if ch not in ch_id_map]

    print(f"  채널 ID 있음: {len(ch_names_with_id)}개")
    print(f"  채널 ID 없음 (이름으로 대체): {len(ch_names_without_id)}개")

    new_results = {}
    print(f"\n2단계: {len(ch_id_list)}개 채널 구독자 수 수집...")
    for i in range(0, len(ch_id_list), 50):
        batch_ids   = ch_id_list[i:i+50]
        batch_names = ch_names_with_id[i:i+50]
        try:
            resp = youtube.channels().list(
                id=','.join(batch_ids),
                part='statistics,snippet'
            ).execute()
            id_to_stats = {item['id']: item for item in resp.get('items', [])}
            for ch_name, ch_id in zip(batch_names, batch_ids):
                item = id_to_stats.get(ch_id)
                if item:
                    stats = item['statistics']
                    new_results[ch_name] = {
                        'Channel':       ch_name,
                        'Channel_ID':    ch_id,
                        'Subscribers':   int(stats.get('subscriberCount', 0)) if stats.get('subscriberCount') else None,
                        'Total_Views':   int(stats.get('viewCount', 0)),
                        'Total_Videos':  int(stats.get('videoCount', 0)),
                    }
                else:
                    new_results[ch_name] = {
                        'Channel': ch_name, 'Channel_ID': ch_id,
                        'Subscribers': None, 'Total_Views': None, 'Total_Videos': None
                    }
            if (i // 50 + 1) % 5 == 0:
                print(f"  {i+50}/{len(ch_id_list)} 완료")
            time.sleep(0.05)
        except HttpError as e:
            print(f"  오류 (batch {i//50}): {e}")
            time.sleep(2)

    # ── 채널 ID 없는 채널: 이름으로 검색 (최대 50개만)
    if ch_names_without_id:
        print(f"\n이름 검색으로 {min(50, len(ch_names_without_id))}개 추가 수집...")
        for ch_name in ch_names_without_id[:50]:
            try:
                search = youtube.search().list(
                    q=ch_name, type='channel', part='id', maxResults=1
                ).execute()
                if search.get('items'):
                    ch_id = search['items'][0]['id']['channelId']
                    resp2 = youtube.channels().list(
                        id=ch_id, part='statistics'
                    ).execute()
                    if resp2.get('items'):
                        stats = resp2['items'][0]['statistics']
                        new_results[ch_name] = {
                            'Channel': ch_name, 'Channel_ID': ch_id,
                            'Subscribers': int(stats.get('subscriberCount', 0)) if stats.get('subscriberCount') else None,
                            'Total_Views': int(stats.get('viewCount', 0)),
                            'Total_Videos': int(stats.get('videoCount', 0)),
                        }
                time.sleep(0.2)
            except HttpError as e:
                print(f"  검색 실패: {ch_name} - {e}")
                time.sleep(1)

    # ── 병합 및 저장
    all_results = {**existing, **new_results}
    result_df = pd.DataFrame(list(all_results.values()))
    result_df = result_df.merge(
        pd.DataFrame({'Channel': all_channels}), on='Channel', how='right'
    )
    result_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    has_subs = result_df['Subscribers'].notna().sum()
    print(f"\n=== 수집 완료 ===")
    print(f"전체 채널: {len(result_df)}개")
    print(f"구독자 수 확보: {has_subs}개 ({has_subs/len(result_df)*100:.1f}%)")
    print(f"저장: {OUT_PATH}")
    return result_df

if __name__ == '__main__':
    collect()
