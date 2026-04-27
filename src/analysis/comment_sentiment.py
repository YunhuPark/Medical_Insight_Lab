"""
Phase 3-B: 댓글 감성 분석
공포 키워드 포함 영상 vs 일반 영상의 댓글 반응 품질 차이
- 긍정/부정/중립 감성 비율
- 공포 반응 댓글 비율 ("무서워", "충격", "걱정" 등)
- 참여 품질 (좋아요 많은 댓글 vs 일반 댓글)
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
COMMENT_PATH = os.path.join(BASE_DIR, '../comments.csv')
DATA_PATH    = os.path.join(BASE_DIR, '../3_medical_platinum_final.csv')
SAVE_DIR     = os.path.join(BASE_DIR, '../../results')

# 감성 사전 (의료 도메인 특화)
POSITIVE_WORDS = [
    '감사','도움','좋아','최고','훌륭','명쾌','완치','건강','희망','나았',
    '좋아졌','회복','완벽','유익','정보','알았','실천','해봐야','감동','효과'
]
NEGATIVE_WORDS = [
    '무서워','걱정','불안','두렵','공포','심각','위험','최악','슬프','절망',
    '힘들','고통','괴롭','겁','무섭','끔찍','억울','화가','실망','불만'
]
FEAR_REACTION = [
    '무서워','충격','깜짝','놀랐','헉','대박','소름','경악','세상에','말도'
]
ACTION_WORDS = [
    '해봐야','실천','바꿔야','조심','주의','먹어봐','가봐야','상담','병원','검사'
]

def analyze_sentiment_dict(text):
    text = str(text).lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    fear = sum(1 for w in FEAR_REACTION if w in text)
    action = sum(1 for w in ACTION_WORDS if w in text)
    if pos > neg:
        label = 'positive'
    elif neg > pos:
        label = 'negative'
    else:
        label = 'neutral'
    return {'sentiment': label, 'pos_score': pos, 'neg_score': neg,
            'fear_reaction': fear, 'action_intent': action}

def run_comment_sentiment():
    if not os.path.exists(COMMENT_PATH):
        print(f"댓글 파일 없음: {COMMENT_PATH}")
        print("먼저 실행: python src/enrichment/comment_collector.py")
        _run_without_comments()
        return

    df_video = pd.read_csv(DATA_PATH)
    df_comm  = pd.read_csv(COMMENT_PATH)

    FEAR_KEYWORDS = [
        '충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
        '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
        '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격'
    ]
    df_video['Has_Fear'] = df_video['Title'].apply(
        lambda x: int(any(kw in str(x) for kw in FEAR_KEYWORDS))
    )
    vid_fear = df_video.set_index('Video_ID')['Has_Fear']
    df_comm['Has_Fear'] = df_comm['Video_ID'].map(vid_fear).fillna(0).astype(int)

    # 감성 분석
    sentiment_results = df_comm['Comment'].apply(analyze_sentiment_dict)
    df_comm = pd.concat([df_comm, pd.DataFrame(list(sentiment_results))], axis=1)

    # 그룹별 비교
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('댓글 감성 분석: 공포 소구 영상 vs 일반 영상 반응 품질',
                 fontsize=15, fontweight='bold', y=1.01)

    labels = ['일반 영상', '공포 키워드 영상']
    colors = ['#0f3460', '#e94560']

    # [1] 감성 분포
    for i, (fear_val, label) in enumerate([(0, '일반'), (1, '공포')]):
        sub = df_comm[df_comm['Has_Fear']==fear_val]['sentiment'].value_counts(normalize=True)
        ax = axes[0,0]
        if i == 0:
            x = np.arange(3)
        for j, s in enumerate(['positive','neutral','negative']):
            val = sub.get(s, 0)
            axes[0,0].bar(j + i*0.35, val*100, 0.35,
                          color=colors[i] if j==0 else (colors[i]+'88'), alpha=0.85,
                          label=label if j==0 else '_nolegend_', edgecolor='white')
    axes[0,0].set_xticks([0.18, 1.18, 2.18])
    axes[0,0].set_xticklabels(['긍정', '중립', '부정'])
    axes[0,0].set_ylabel('비율 (%)')
    axes[0,0].set_title('댓글 감성 비율 비교')
    axes[0,0].legend()

    # [2] 공포 반응 댓글 비율
    fear_r = [df_comm[df_comm['Has_Fear']==v]['fear_reaction'].mean() for v in [0,1]]
    axes[0,1].bar(labels, fear_r, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(axes[0,1].patches, fear_r):
        axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                       f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0,1].set_ylabel('댓글당 공포반응 단어 평균')
    axes[0,1].set_title('댓글 내 공포 반응 강도')

    # [3] 행동 의도 댓글 비율
    action_r = [df_comm[df_comm['Has_Fear']==v]['action_intent'].mean() for v in [0,1]]
    axes[0,2].bar(labels, action_r, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(axes[0,2].patches, action_r):
        axes[0,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                       f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0,2].set_ylabel('댓글당 행동 의도 단어 평균')
    axes[0,2].set_title('공포 영상의 행동 의도 유발 효과\n("병원 가야겠다", "조심해야겠다")')

    # [4] 영상별 댓글 감성 분포 (상자그림)
    groups_sent = [df_comm[df_comm['Has_Fear']==v]['pos_score'] - df_comm[df_comm['Has_Fear']==v]['neg_score']
                   for v in [0,1]]
    bp = axes[1,0].boxplot(groups_sent, labels=labels, patch_artist=True,
                           medianprops=dict(color='white', lw=2.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    stat_s, p_s = stats.mannwhitneyu(groups_sent[1], groups_sent[0], alternative='two-sided')
    axes[1,0].set_ylabel('감성 점수 (긍정-부정)')
    axes[1,0].set_title(f'댓글 감성 점수 분포\nMann-Whitney p={p_s:.4f}')

    # [5] 좋아요 많이 받은 댓글 패턴
    top_liked = df_comm.nlargest(20, 'Comment_Likes')
    top_liked_fear = top_liked['Has_Fear'].value_counts(normalize=True)
    axes[1,1].pie([top_liked_fear.get(0,0), top_liked_fear.get(1,0)],
                  labels=['일반 영상', '공포 영상'],
                  colors=colors, autopct='%1.1f%%', startangle=90,
                  wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[1,1].set_title('좋아요 TOP 20 댓글\n공포 vs 일반 영상 비율')

    # [6] 워드빈도 (공포 영상 댓글)
    fear_comments = ' '.join(df_comm[df_comm['Has_Fear']==1]['Comment'].astype(str))
    word_counts = Counter(fear_comments.split())
    top_words = [(w, c) for w, c in word_counts.most_common(50)
                 if len(w) > 1 and w not in ['이', '가', '을', '를', '은', '는', '의', '에', '도', '로', '으로', '와', '과']][:15]
    if top_words:
        words, counts = zip(*top_words)
        axes[1,2].barh(list(words)[::-1], list(counts)[::-1],
                       color=plt.cm.Reds(np.linspace(0.4,0.9,len(words))), edgecolor='white')
        axes[1,2].set_xlabel('빈도')
        axes[1,2].set_title('공포 키워드 영상 댓글\n자주 등장 단어 TOP 15')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'v2_08_comment_sentiment.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n총 분석 댓글: {len(df_comm)}개")
    print(f"공포 영상 댓글: {(df_comm['Has_Fear']==1).sum()}개")
    print(f"일반 영상 댓글: {(df_comm['Has_Fear']==0).sum()}개")

def _run_without_comments():
    """댓글 없이 텍스트 기반 시뮬레이션"""
    print("댓글 데이터 없음 — 분석 대기 중")

if __name__ == '__main__':
    run_comment_sentiment()
