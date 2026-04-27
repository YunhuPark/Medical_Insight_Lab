"""
Phase 1-B: 상위 300개 영상의 댓글 수집 (영상당 최대 20개)
감성 분석용 데이터 수집
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
OUT_PATH  = os.path.join(BASE_DIR, '../../src/comments.csv')

def collect_comments(n_videos=300, comments_per_video=20):
    df = pd.read_csv(DATA_PATH)
    top_videos = df.nlargest(n_videos, 'Views')[['Video_ID', 'Title', 'Keyword', 'Views', 'Has_Fear' if 'Has_Fear' in df.columns else 'Views']]
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    all_comments = []
    failed = 0

    for idx, (_, row) in enumerate(top_videos.iterrows()):
        vid_id = row['Video_ID']
        try:
            resp = youtube.commentThreads().list(
                videoId=vid_id,
                part='snippet',
                maxResults=comments_per_video,
                order='relevance',
                textFormat='plainText'
            ).execute()

            for item in resp.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                all_comments.append({
                    'Video_ID': vid_id,
                    'Title': row['Title'],
                    'Keyword': row['Keyword'],
                    'Views': row['Views'],
                    'Comment': snippet['textDisplay'],
                    'Comment_Likes': snippet['likeCount'],
                    'Published': snippet['publishedAt'],
                })
            time.sleep(0.15)

        except Exception as e:
            failed += 1

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{n_videos} 완료 (실패: {failed}개)")

    out_df = pd.DataFrame(all_comments)
    out_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"수집 완료: {len(out_df)}개 댓글 → {OUT_PATH}")
    return out_df

if __name__ == '__main__':
    collect_comments()
