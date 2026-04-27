"""
전체 영상 댓글 완전 수집 (재시도 안전 버전)
- 이미 수집된 Video_ID는 건너뜀
- HTTP 오류별 처리 (403 댓글 비활성, 404 없음, 429 할당량)
- 영상당 최대 30개, 전체 2,512개 영상 대상
- API 비용: 영상당 1유닛 = 최대 2,512유닛 (일 할당량 10,000의 25%)
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
OUT_PATH  = os.path.join(BASE_DIR, '../comments.csv')

COMMENTS_PER_VIDEO = 30

def collect(max_videos=2512):
    df = pd.read_csv(DATA_PATH)
    FEAR_WORDS = ['충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
                  '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
                  '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격']
    df['Has_Fear'] = df['Title'].astype(str).apply(lambda x: int(any(k in x for k in FEAR_WORDS)))

    # 이미 수집된 영상 확인
    existing_ids = set()
    existing_rows = []
    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        existing_ids = set(old['Video_ID'].unique())
        existing_rows = old.to_dict('records')
        print(f"기존 댓글: {len(old)}개 ({len(existing_ids)}개 영상)")

    # 수집 대상: 공포/비공포 균형 맞추기 (각 절반씩)
    fear_videos   = df[df['Has_Fear']==1].nlargest(max_videos//2, 'Views')
    nofear_videos = df[df['Has_Fear']==0].nlargest(max_videos//2, 'Views')
    target = pd.concat([fear_videos, nofear_videos])
    to_collect = target[~target['Video_ID'].isin(existing_ids)]
    print(f"수집 대상: {len(to_collect)}개 영상 (기존 제외)")

    youtube = build('youtube', 'v3', developerKey=API_KEY)
    new_rows = []
    disabled = 0

    for i, (_, row) in enumerate(to_collect.iterrows()):
        vid_id = row['Video_ID']
        try:
            resp = youtube.commentThreads().list(
                videoId=vid_id,
                part='snippet',
                maxResults=COMMENTS_PER_VIDEO,
                order='relevance',
                textFormat='plainText'
            ).execute()

            for item in resp.get('items', []):
                s = item['snippet']['topLevelComment']['snippet']
                new_rows.append({
                    'Video_ID':    vid_id,
                    'Title':       row['Title'],
                    'Keyword':     row['Keyword'],
                    'Has_Fear':    row['Has_Fear'],
                    'Views':       row['Views'],
                    'Comment':     s['textDisplay'],
                    'Like_Count':  s.get('likeCount', 0),
                    'Published_At': s.get('publishedAt', ''),
                })
            time.sleep(0.1)

        except HttpError as e:
            code = e.resp.status
            if code == 403:
                disabled += 1  # 댓글 비활성화
            elif code == 404:
                pass  # 영상 삭제됨
            elif code == 429 or 'quotaExceeded' in str(e):
                print(f"\n할당량 초과 — {i}개 영상까지 수집 후 중단")
                break
            else:
                print(f"오류 {code}: {vid_id}")
            time.sleep(0.1)

        if (i + 1) % 100 == 0:
            collected = len(existing_rows) + len(new_rows)
            print(f"  {i+1}/{len(to_collect)} 완료 | 댓글 누계: {collected}개 | 비활성: {disabled}개")
            # 중간 저장
            save_df = pd.DataFrame(existing_rows + new_rows)
            save_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    # 최종 저장
    final_df = pd.DataFrame(existing_rows + new_rows)
    final_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\n=== 수집 완료 ===")
    print(f"새로 수집된 댓글: {len(new_rows)}개")
    print(f"전체 댓글: {len(final_df)}개 ({final_df['Video_ID'].nunique()}개 영상)")
    print(f"댓글 비활성 영상: {disabled}개")
    print(f"저장: {OUT_PATH}")
    return final_df

if __name__ == '__main__':
    collect()
