"""
Phase 1-C: 썸네일 이미지 다운로드 (2,512개)
이미 Thumbnail_URL 컬럼 존재 → HTTP 요청만으로 수집 가능
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '../../src/3_medical_platinum_final.csv')
THUMB_DIR  = os.path.join(BASE_DIR, '../../thumbnails')

def download_one(args):
    vid_id, url = args
    out_path = os.path.join(THUMB_DIR, f"{vid_id}.jpg")
    if os.path.exists(out_path):
        return vid_id, True, 'cached'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return vid_id, True, 'downloaded'
        return vid_id, False, f'http_{r.status_code}'
    except Exception as e:
        return vid_id, False, str(e)[:30]

def download_all_thumbnails(workers=20):
    os.makedirs(THUMB_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    tasks = list(zip(df['Video_ID'], df['Thumbnail_URL']))

    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            vid_id, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)} | 성공: {success} | 실패: {fail}")

    print(f"다운로드 완료: 성공 {success}개 / 실패 {fail}개")
    return success, fail

if __name__ == '__main__':
    download_all_thumbnails()
