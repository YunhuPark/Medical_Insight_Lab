# 한국 의료 유튜브에서 공포 소구가 조회수에 미치는 인과적 영향:
# 성향점수매칭과 이중차분법을 활용한 분석

**Causal Effects of Fear Appeals on View Counts in Korean Medical YouTube:
A Propensity Score Matching and Difference-in-Differences Approach**

---

**저자**: 박윤후  
**소속**: 인제대학교 의료IT학과  
**이메일**: byunhu35@gmail.com  
**투고일**: 2026년 4월

---

## 국문 초록

**목적**: 본 연구는 한국 의료 유튜브 콘텐츠에서 공포·위험 키워드가 포함된 제목이 조회수에 미치는 인과적 영향을 규명하고자 하였다. 기존 연구들은 단순 상관관계 분석에 그쳐 채널 규모 등 주요 혼란변수를 통제하지 못했다는 한계가 있다.

**방법**: YouTube Data API v3를 통해 37개 질환 카테고리의 한국 의료 유튜브 영상 3,713개(921개 채널, 2013–2026년)를 수집하였다. 분석은 단순 비교(Mann-Whitney U), 영상 나이 보정(Views Per Day), 성향점수매칭(PSM), 채널 고정효과 회귀, COVID-19 이중차분법(DiD)의 5단계로 진행하였다. 채널 유형(의료전문 vs. 일반)별 조절 효과를 검토하고, OpenCV 기반 썸네일 분석, KoBERT 제목 임베딩, 댓글 감성 분석을 포함하는 멀티모달 파이프라인을 구축하였다.

**결과**: 단순 비교에서 공포 키워드 포함 영상이 2.1배 높은 조회수를 보였으나(p<0.05), 영상 나이 보정 후 효과는 소멸하였다(p=0.9999). PSM으로 채널 규모를 통제한 결과 공포 키워드가 오히려 조회수를 0.63배 감소시키는 역효과가 확인되었다(p=0.019, Bootstrap 95% CI: [0.44, 0.85]). 채널 고정효과 분석에서는 +7%(p=0.306), COVID-19 DiD 분석에서는 +3.4%(95% CI: [-20.8%, +34.7%])로 모두 비유의하였다. 채널 유형별 조절 효과 분석에서 의료전문 채널의 경우 공포 키워드가 조회수를 0.83배 유의하게 감소시킨 반면(p=0.017, Hedge's g=-0.269), 일반 채널에서는 유의한 효과가 없었다(1.14배, p=0.558).

**결론**: 공포 소구의 조회수 증대 효과는 채널 규모에 의한 착시였으며, 인과 추론 방법 적용 시 효과가 소멸하거나 역전되었다. 특히 의료전문 채널에서 공포 키워드는 조회수를 유의하게 감소시켜, 전문성 기반 신뢰를 기대하는 시청자의 반응과 일치하였다. 본 결과는 의료 정보 채널의 제목 전략 수립과 공중보건 커뮤니케이션 정책에 실질적 시사점을 제공한다.

**주요어**: 공포 소구, 의료 유튜브, 인과 추론, 성향점수매칭, 이중차분법, 공중보건 커뮤니케이션

---

## Abstract

**Purpose**: This study aimed to identify the causal effect of fear- and risk-laden keywords in video titles on view counts in Korean medical YouTube content. Previous studies have been limited to simple correlation analyses and failed to control for major confounders such as channel size.

**Methods**: A total of 3,713 Korean medical YouTube videos (921 channels, 2013–2026) across 37 disease categories were collected via the YouTube Data API v3. Analysis proceeded in five stages: naive comparison (Mann-Whitney U), video age correction (Views Per Day), propensity score matching (PSM), channel fixed-effects regression, and a COVID-19 difference-in-differences (DiD) natural experiment. Moderating effects by channel type (Medical Professional vs. General) were examined. A multimodal pipeline was constructed incorporating OpenCV-based thumbnail analysis, KoBERT title embeddings, and comment sentiment analysis.

**Results**: In naive comparison, videos containing fear keywords showed 2.1 times higher view counts (p<0.05); however, this effect disappeared after age correction (p=0.9999). PSM controlling for channel size revealed a reverse effect, with fear keywords associated with a 0.63-fold decrease in views (p=0.019, Bootstrap 95% CI: [0.44, 0.85]). Channel fixed-effects yielded a non-significant +7% (p=0.306), and COVID-19 DiD analysis yielded a non-significant +3.4% (95% CI: [-20.8%, +34.7%]). Moderation analysis showed that fear keywords significantly decreased views in Medical Professional channels (0.83x, p=0.017, Hedge's g=-0.269), while no significant effect was found in General channels (1.14x, p=0.558).

**Conclusion**: The apparent view-count advantage of fear appeals was an artifact of channel size confounding; causal methods revealed effect attenuation or reversal. In Medical Professional channels, fear keywords significantly reduced view counts, consistent with audience expectations of expert, trustworthy communication. These findings provide practical implications for title strategy in medical information channels and public health communication policy.

**Keywords**: fear appeal, medical YouTube, causal inference, propensity score matching, difference-in-differences, public health communication

---

## 1. 서론

### 1.1 연구 배경

디지털 미디어의 발전과 함께 유튜브는 의료 정보 탐색의 주요 경로로 부상하였다. 국내 인터넷 이용자의 60% 이상이 건강 관련 정보를 검색할 때 동영상 플랫폼을 활용하며[1], 유튜브의 의료 관련 콘텐츠는 전문 의료진이 제공하는 채널부터 비전문 건강 정보 채널까지 매우 다양한 형태로 존재한다[2]. 이러한 환경에서 콘텐츠 제작자들은 더 많은 시청자를 유인하기 위한 다양한 제목 전략을 구사하며, 그 중 공포 소구(fear appeal)가 빈번하게 활용된다.

공포 소구란 메시지 수신자에게 잠재적 위협이나 위험을 강조하여 태도 변화 또는 행동 유발을 이끌어 내는 설득 전략이다[3]. 보호동기이론(Protection Motivation Theory; PMT)[4]과 확장평행과정모델(Extended Parallel Process Model; EPPM)[5]은 공포 소구가 인지된 위협의 심각성과 취약성을 높임으로써 보호 동기를 촉진한다고 설명한다. 그러나 공포 수준이 지나치게 높을 경우 오히려 방어적 회피 반응을 유발할 수 있음도 이론적으로 제시되어 있다[5].

의료 유튜브 맥락에서 공포 소구 전략의 효과는 단순히 설득적 효과를 넘어, 플랫폼의 알고리즘과 결합된 조회수 증대 효과로 연결될 수 있다. 높은 조회수는 알고리즘에 의한 추천 가능성을 높이고, 이는 더 많은 시청자에게 해당 정보가 노출됨을 의미한다. 따라서 의료 콘텐츠에서 공포 소구의 조회수 효과를 정확히 규명하는 것은 공중보건 정보 전달의 관점에서 중요한 의미를 갖는다.

### 1.2 선행 연구의 한계

선행 연구들은 의료 유튜브 콘텐츠의 품질[6], 정보 정확성[7], 시청자 참여도[8] 등을 분석해 왔으나, 제목 전략과 조회수의 인과관계를 엄밀하게 추정한 연구는 거의 없다. 대부분의 연구가 단순 상관분석 또는 기술통계에 머물러 있으며, 이 접근법은 심각한 방법론적 한계를 내포한다.

가장 핵심적인 한계는 **채널 규모 혼란변수**의 문제이다. 대형 채널일수록 공포 키워드를 포함한 자극적인 제목을 사용하는 경향이 있으며, 동시에 구독자 기반으로 인해 조회수 자체가 높다. 이 경우 단순 비교는 공포 소구의 효과와 채널 규모의 효과를 분리하지 못하고, 채널 규모에 의한 착시 효과(confounding bias)를 공포 소구의 효과로 잘못 귀속시키게 된다.

또한 **영상 나이 편향** 역시 중요한 혼란 요인이다. 오래된 영상일수록 더 많은 누적 조회수를 가질 가능성이 높으며, 특정 시기에 공포 키워드의 사용 빈도가 달랐다면 단순 조회수 비교는 시간적 편향을 포함하게 된다.

### 1.3 연구 목적 및 연구 가설

본 연구는 이러한 한계를 극복하기 위해 인과 추론(causal inference) 방법론을 단계적으로 적용하여 공포 소구가 의료 유튜브 조회수에 미치는 순수한 인과적 영향을 추정하고자 한다. 구체적인 연구 가설은 다음과 같다.

- **H1**: 공포 키워드를 포함한 의료 유튜브 영상은 그렇지 않은 영상보다 높은 조회수를 보일 것이다 (단순 비교).
- **H2**: 영상 나이를 보정하면 공포 키워드의 조회수 효과는 감소하거나 소멸할 것이다.
- **H3**: 채널 규모를 통제(PSM)하면 공포 키워드의 순수 인과 효과가 드러날 것이다.
- **H4**: 채널 고정효과를 적용하면 채널 내 공포 키워드 효과가 비유의할 것이다.
- **H5**: COVID-19라는 외생적 충격을 이용한 DiD 분석에서 공포 키워드의 조건부 효과는 불확실할 것이다.
- **H6**: 채널 유형(의료전문 vs. 일반)이 공포 소구의 효과 방향을 조절할 것이다.

---

## 2. 연구방법

### 2.1 데이터 수집

YouTube Data API v3를 활용하여 2013년부터 2026년 4월까지 업로드된 한국 의료 유튜브 영상을 수집하였다. 검색 키워드는 '당뇨', '고혈압', '암', '심장', '뇌졸중' 등 37개 질환 카테고리로 구성하였으며, 수집 기준은 (1) 한국어 제목 및 설명, (2) 조회수 100회 이상, (3) 업로드 후 30일 이상 경과로 설정하였다.

최종 분석 데이터셋은 **3,713개 영상, 921개 채널**로 구성되었다. pre-COVID 기간(2018–2019년)의 데이터 부족 문제를 보완하기 위해 targeted collection을 통해 606개 영상을 추가 수집하여 병합하였다. 데이터 수집 기준일은 2026년 4월 19일이며, 모든 조회수 및 좋아요 수는 해당 시점의 스냅샷이다.

### 2.2 변수 정의

#### 2.2.1 처치변수: 공포 키워드 포함 여부 (Has_Fear)

선행 연구[3,9]의 공포 소구 조작화 방식을 참고하여, 의료 유튜브 맥락에서 공포·위험·긴급성을 나타내는 30개의 키워드 목록을 구성하였다: '충격', '경악', '사망', '암', '위험', '절대', '금지', '무시', '신호', '전조', '증상', '말기', '시한부', '응급', '마비', '실명', '절단', '투석', '쇼크', '발작', '최악', '경고', '주의', '폭발', '급증', '심각', '치명', '돌연', '급격'. 영상 제목에 이 중 하나 이상이 포함된 경우 Has_Fear=1, 그렇지 않은 경우 Has_Fear=0으로 이진 부호화하였다.

#### 2.2.2 결과변수: 조회수 및 Views Per Day (VPD)

원 조회수(Views)는 수집 시점까지의 누적 조회수이다. 영상 나이 편향을 보정하기 위해 조회수를 영상 경과 일수(Age_Days)로 나눈 일평균 조회수(Views Per Day; VPD)를 산출하였다. 통계 모형에서는 극심한 우편향을 보정하기 위해 log1p 변환값을 사용하였다.

#### 2.2.3 채널 유형 (Type)

의료전문 채널(Medical Pro)은 의사, 간호사, 병원 공식 채널 등 면허를 가진 의료 전문가가 운영하거나 의료기관이 운영하는 채널로, 채널명, 소개, 콘텐츠 특성을 기반으로 분류하였다. 나머지는 일반 채널(General)로 분류하였다.

#### 2.2.4 공변량

PSM 분석을 위한 공변량으로 (1) 채널 중앙 조회수의 로그값(Log_Ch_Med: 채널 규모 대리변수), (2) 영상 경과 일수의 로그값(Log_Age), (3) 제목 길이(Title_Length), (4) 제목 내 숫자 포함 여부(Has_Number), (5) 의료 점수(Medical_Score: 의료 전문 어휘 빈도 기반 신뢰도 지수)를 사용하였다.

### 2.3 분석 방법

#### 2.3.1 단순 비교 (H1)

공포 키워드 포함 여부에 따른 조회수 차이를 비모수 검정인 Mann-Whitney U 검정으로 비교하였다. 조회수의 극심한 우편향(왜도 ≈ 7.2)으로 인해 모수적 검정을 적용하지 않았다.

#### 2.3.2 나이 보정 비교 (H2)

영상 나이 편향을 보정하기 위해 VPD를 결과변수로 사용하여 동일한 Mann-Whitney U 검정을 수행하였다.

#### 2.3.3 성향점수매칭 (H3)

채널 규모 등 관찰 가능한 공변량을 통제하여 인과 효과를 추정하기 위해 성향점수매칭(Propensity Score Matching; PSM)을 적용하였다[10]. 성향 점수는 앞서 기술한 5개 공변량과 37개 질환 카테고리 더미변수를 포함하는 로지스틱 회귀로 추정하였으며, 1:1 최근접이웃 매칭(caliper=0.2 SD)을 적용하였다. 매칭 후 표준화 평균 차이(Standardized Mean Difference; SMD)를 통해 공변량 균형을 진단하였으며[11], 처치군 평균 효과(ATT)는 Wilcoxon 부호순위 검정으로, 95% 신뢰구간은 Bootstrap(2,000회 반복)으로 추정하였다. 인과 주장의 강건성은 Rosenbaum 민감도 분석[12]으로 검토하였다.

#### 2.3.4 채널 고정효과 회귀 (H4)

채널 수준의 시불변 특성(관찰 불가능한 혼란변수 포함)을 통제하기 위해 채널 고정효과 모형(Within estimator)을 적용하였다[13]. 동일 채널 내 공포 키워드 포함 영상과 미포함 영상의 조회수 차이를 추정함으로써 채널 간 이질성을 제거하였다.

#### 2.3.5 이중차분법 — COVID-19 자연실험 (H5)

COVID-19 팬데믹(2020년 이후)을 외생적 충격으로 활용하여 이중차분법(Difference-in-Differences; DiD)을 적용하였다[14]. DiD 추정량 = (Post × Fear 집단의 VPD 변화) − (Post × NoFear 집단의 VPD 변화)로 산출하였으며, DiD 핵심 가정인 평행추세 가정은 pre-COVID 기간(2015–2019)의 상호작용 항 회귀분석으로 검정하였다. Bootstrap 95% CI를 산출하였으며, 초기 +42.5% 추정치는 pre-COVID 샘플 부족(n=73)에 의한 추정 불안정으로 판명되어 추가 수집(n=606) 후 재분석하였다.

#### 2.3.6 채널 유형별 조절 효과 (H6)

채널 유형(Medical Pro / General)을 조절변수로 하여 각 그룹 내 공포 키워드 효과를 독립적으로 추정하고, 효과 크기는 Hedge's g와 rank-biserial 상관계수로 보고하였다.

### 2.4 멀티모달 분석

#### 2.4.1 썸네일 컴퓨터 비전

OpenCV를 활용하여 썸네일 이미지(n=2,372)에서 빨간색 비율(red_ratio), 밝기(brightness), 대비(contrast), 채도(saturation), 엣지 밀도(edge_density), 텍스트 비율(text_ratio)을 추출하였다. 각 피처와 조회수의 연관성은 Spearman 상관계수로 분석하였다.

#### 2.4.2 댓글 감성 분석

YouTube Data API v3로 수집한 19,232개 댓글(840개 영상)에 대해 사전 구축된 감성 사전을 활용하여 긍정, 부정, 공포 반응 댓글 비율을 산출하였다. 공포 키워드 포함 영상과 미포함 영상 간 댓글 감성 차이를 채널 유형별로 분석하였다.

#### 2.4.3 KoBERT 제목 임베딩

multilingual-e5-small 모델[15]을 활용하여 2,512개 영상 제목을 벡터화하였다. PCA로 2차원 축소 후 K-Means(k=9) 클러스터링을 적용하여 의미적 제목 유형을 도출하였다.

---

## 3. 연구결과

### 3.1 기술 통계

수집된 3,713개 영상의 조회수 중앙값은 89,421회(IQR: 24,830–387,620)였으며, 분포는 극심한 우편향(왜도=7.2)을 나타내었다. 공포 키워드 포함 영상은 전체의 30.8%(n=1,144)였다. 채널 유형별로는 의료전문 채널 42.3%, 일반 채널 57.7%였다. Shorts 포맷(60초 이하)은 전체의 27.1%를 차지하였다.

분석 기간인 2013년부터 2026년까지 공포 키워드 포함 영상의 비율은 꾸준히 증가하는 추세였으며, COVID-19 팬데믹 이후인 2020년부터 일시적으로 증가하였다가 이후 감소하는 패턴을 보였다.

### 3.2 단순 비교 (H1)

공포 키워드 포함 영상의 조회수 중앙값(126,847회)은 미포함 영상(60,348회)보다 2.1배 높았으며, Mann-Whitney U 검정 결과 통계적으로 유의하였다(U=1,847,293, p<0.001). 이는 H1을 지지하는 것으로 보이나, 이후 분석에서 혼란변수에 의한 착시임이 밝혀졌다.

### 3.3 영상 나이 보정 (H2)

VPD를 결과변수로 적용하였을 때, 공포 키워드 포함 영상(중앙 VPD=184.2)과 미포함 영상(중앙 VPD=161.8) 간의 차이는 통계적으로 유의하지 않았다(p=0.9999). 영상 나이 보정 전 2.1배였던 효과가 VPD 기준에서는 1.14배로 수렴하였으며, 이는 오래된 영상의 편향이 단순 조회수 비교를 교란하였음을 의미한다. H2는 지지되었다.

### 3.4 성향점수매칭 (H3)

로지스틱 회귀로 추정된 성향 점수(AUC=0.74)를 이용하여 caliper 매칭 후 582쌍의 매칭 쌍을 확보하였다. 매칭 전 채널 중앙 조회수(Log_Ch_Med)의 SMD는 -0.079였으며, 매칭 후에도 -0.159로 잔존 불균형이 관찰되었다. 영상 나이(Log_Age, SMD: 0.249→0.053), 제목 길이(Title_Length, SMD: 0.079→-0.010), 의료 점수(Medical_Score, SMD: -0.242→-0.015)는 매칭 후 균형을 달성하였다(|SMD|<0.1).

ATT 추정 결과 공포 키워드가 조회수를 **0.63배 감소**시켰으며(Wilcoxon p=0.019, Bootstrap 95% CI: [0.44, 0.85]), 이는 채널 규모를 통제하면 공포 소구가 오히려 역효과를 나타냄을 의미한다. H3는 부분 지지되었다(역방향 효과 확인).

Rosenbaum 민감도 분석 결과 Γ≈1.15 수준의 숨겨진 혼란변수가 존재할 경우 결과가 번복될 수 있어(p_upper=0.240), 인과 해석에는 보수적 접근이 요구된다.

### 3.5 채널 고정효과 (H4)

채널 고정효과 회귀 분석에서 공포 키워드의 추정 효과는 +7.2%(SE=6.9%, p=0.306)로 통계적으로 유의하지 않았다. 채널 내 비교에서는 공포 키워드가 조회수를 증가시키지도, 감소시키지도 않았으며, 이는 채널 간 이질성이 제거되면 공포 소구의 독립적 효과가 존재하지 않음을 시사한다. H4는 지지되었다.

### 3.6 COVID-19 이중차분법 (H5)

DiD 평행추세 가정을 pre-COVID 기간(2015–2019) 상호작용 항 회귀로 검정한 결과, 상호작용 계수는 +0.006(t=0.074, p=0.941)으로 평행추세 가정이 충족되었음을 확인하였다.

DiD 추정량은 +3.4%(Bootstrap 95% CI: [-20.8%, +34.7%])로, 신뢰구간이 0을 포함하여 통계적으로 유의하지 않았다. 초기 분석에서 +42.5%(p=0.015)로 나타난 추정치는 pre-COVID 샘플 부족(n=73)에 의한 추정 불안정으로, 보완 수집(n=606) 후 +3.4%로 수렴하였다. H5는 지지되었다(팬데믹 조건부 효과 불확실).

### 3.7 채널 유형별 조절 효과 (H6)

채널 유형별 분석 결과, **의료전문 채널**에서 공포 키워드는 조회수를 0.83배 유의하게 감소시켰다(p=0.017, Hedge's g=-0.269). g=-0.269는 Cohen[16]의 기준에 따라 'small' 수준의 효과크기이며, 사후 검정력은 0.618로 산출되었다. **일반 채널**에서는 1.14배로 비유의하였다(p=0.558, g=+0.071, 효과크기 'negligible'). H6는 지지되었다.

표 1. 분석 방법별 공포 키워드 효과 추정치 요약

| 분석 방법 | 효과 추정치 | p값 | 95% CI | 유의 여부 |
|-----------|------------|-----|--------|----------|
| 단순 비교 (Mann-Whitney) | +2.1배 | <0.001 | — | ✅ 유의 |
| 나이 보정 (VPD) | +1.14배 | 0.9999 | — | ❌ 비유의 |
| PSM (Bootstrap) | 0.63배 역효과 | 0.019 | [0.44, 0.85] | ✅ 유의 (역방향) |
| 채널 고정효과 | +7.2% | 0.306 | — | ❌ 비유의 |
| DiD (COVID-19) | +3.4% | — | [-20.8%, +34.7%] | ❌ 비유의 |
| Type×Fear: Medical Pro | 0.83배 | 0.017 | — | ✅ 유의 (역방향) |
| Type×Fear: General | 1.14배 | 0.558 | — | ❌ 비유의 |

### 3.8 멀티모달 분석 결과

썸네일 분석에서 빨간색 비율(r=0.18, p<0.001), 대비(r=0.14, p<0.001), 엣지 밀도(r=0.12, p<0.001), 텍스트 비율(r=0.11, p=0.003)이 조회수와 유의한 양의 상관을 보였다. 즉 시각적으로 자극적인 썸네일일수록 조회수가 높았으며, 이는 제목의 공포 소구와 독립적인 시각적 자극 경로가 존재함을 시사한다.

KoBERT 클러스터링 결과 9개의 의미적 군집이 도출되었다(Silhouette=0.116). 클러스터 간 중앙 조회수 차이가 유의하여, 제목의 의미적 유형이 주제 선택과 연관된 조회수 차이를 부분적으로 설명함을 확인하였다.

---

## 4. 고찰

### 4.1 주요 발견의 해석

본 연구의 핵심 발견은 단순 비교에서 유의하게 나타난 공포 소구의 +2.1배 효과가 채널 규모에 의한 **혼란 편향(confounding bias)**임을 실증적으로 규명한 것이다. 구체적으로, 대형 채널일수록 공포 키워드를 빈번하게 사용하며, 동시에 구독자 기반으로 인해 조회수가 구조적으로 높다는 점이 착시를 만들어 낸다. PSM으로 채널 규모를 통제하자 효과가 역전되었고, 채널 고정효과와 DiD 분석에서도 유의한 효과가 발견되지 않았다.

이러한 결과는 온라인 건강 정보에서 자극적 제목 전략의 효과에 대한 기존 가정에 의문을 제기한다. 다수의 선행 연구[2,6]가 클릭베이트적 요소와 참여도의 양의 관계를 보고하였으나, 이는 채널 규모라는 교란변수를 통제하지 않은 결과일 가능성이 높다.

### 4.2 채널 유형의 조절 역할

의료전문 채널에서 공포 키워드가 조회수를 유의하게 감소시킨 결과는 **신뢰 기반 커뮤니케이션 이론**으로 해석할 수 있다. 의료전문 채널의 시청자는 전문성과 권위에 근거한 정보를 기대하며, 선정적이거나 과장된 제목은 채널의 신뢰도 인식을 저해할 수 있다[17]. 이는 의료 분야에서 공포 소구가 역효과를 낼 수 있다는 이론적 논의[5]를 실증적으로 뒷받침한다.

반면 일반 채널에서는 유의한 효과가 없었는데, 이는 일반 건강 정보 콘텐츠 시청자가 전문성보다는 오락성 또는 접근성을 더 중시할 수 있음을 시사한다. 채널 유형에 따른 이 같은 차별적 반응은 향후 의료 커뮤니케이션 전략 수립 시 채널 정체성과 타겟 시청자를 명확히 구분할 필요성을 강조한다.

### 4.3 방법론적 기여

본 연구는 의료 유튜브 분석에 인과 추론 방법론을 체계적으로 적용한 최초의 연구 중 하나로, 다음과 같은 방법론적 기여를 한다.

첫째, 단일 방법이 아닌 PSM, 고정효과, DiD를 순차적으로 적용하여 각 방법의 강점과 한계를 상호 보완적으로 활용하였다. 둘째, 효과크기(Hedge's g), 검정력, Rosenbaum 민감도 분석을 통해 결과의 통계적 견고성과 실용적 중요성을 함께 보고하였다. 셋째, 인과 추론을 넘어 썸네일 컴퓨터 비전, 댓글 감성 분석, KoBERT 임베딩을 통합한 멀티모달 분석 파이프라인을 구축하였다.

### 4.4 연구 한계

본 연구는 다음과 같은 한계를 갖는다. 첫째, 댓글 데이터는 전체 영상의 33%(840개)에만 수집되어 댓글 분석 결과의 일반화에 제한이 있다. 둘째, PSM 분석에서 Log_Ch_Med와 Has_Number 변수의 매칭 후 SMD가 0.1을 초과하여 완전한 균형 달성에 실패하였으며, 이는 잔존 혼란 편향의 가능성을 시사한다. 셋째, Rosenbaum 민감도 분석 결과 Γ≈1.15 수준의 숨겨진 혼란변수로 인과 결과가 번복될 수 있어, 관찰 연구의 본질적 한계를 완전히 극복하지 못하였다. 넷째, 영상 조회수는 수집 시점의 스냅샷으로, 시간에 따른 조회수 변화 패턴을 추적하지 못하였다. 다섯째, 공포 키워드 목록이 선험적으로 구성되었으며, 맥락에 따라 동일 키워드의 감성 강도가 다를 수 있다.

---

## 5. 결론

본 연구는 한국 의료 유튜브에서 공포 소구의 조회수 효과를 다단계 인과 추론으로 분석하여, 단순 비교에서 나타난 +2.1배 효과가 채널 규모에 의한 혼란 편향임을 실증하였다. PSM으로 채널 규모를 통제한 결과 공포 소구는 오히려 조회수를 0.63배 감소시켰으며, 채널 고정효과와 DiD 분석에서도 유의한 효과가 발견되지 않았다. 특히 의료전문 채널에서는 공포 키워드가 조회수를 유의하게 감소시켜(g=-0.269), 전문성 기반 신뢰를 기대하는 시청자의 반응을 확인하였다.

이러한 결과는 실무적으로 다음과 같은 시사점을 제공한다. 의료전문 채널은 공포 키워드를 지양하고 전문성·신뢰성을 전달하는 정보 제공형 제목 전략이 유리하다. 일반 채널 역시 공포 키워드가 통계적으로 유의한 이득을 주지 않으므로, 콘텐츠 품질과 채널 성장에 집중하는 것이 합리적이다. 공중보건 관점에서 볼 때, 의료 정보의 클릭베이트화는 단기적으로도 효과적이지 않을 뿐만 아니라 전문 채널의 신뢰도를 저하시켜 양질의 의료 정보 전달을 방해할 수 있다.

향후 연구에서는 종단적 패널 데이터를 구축하여 조회수의 시간 경로를 추적하고, 공포 키워드의 강도를 연속 변수로 측정하는 정교화된 조작화, 그리고 영상 시청 완료율 등 심층 참여 지표를 포함한 확장 분석이 필요하다.

---

## 참고문헌

[1] 한국인터넷진흥원. 2024 인터넷이용실태조사. 서울: 한국인터넷진흥원; 2024.

[2] Madathil KC, Rivera-Rodriguez AJ, Greenstein JS, Gramopadhye AK. Healthcare information on YouTube: A systematic review. Health Informatics J. 2015;21(3):173-194.

[3] Witte K. Putting the fear back into fear appeals: The extended parallel process model. Commun Monogr. 1992;59(4):329-349.

[4] Rogers RW. A protection motivation theory of fear appeals and attitude change. J Psychol. 1975;91(1):93-114.

[5] Witte K, Allen M. A meta-analysis of fear appeals: Implications for effective public health campaigns. Health Educ Behav. 2000;27(5):591-615.

[6] Singh AG, Singh S, Singh PP. YouTube for information on rheumatoid arthritis—patient or physician? J Rheumatol. 2012;39(5):899-903.

[7] Kunze KN, Cohn MR, Levine WN, Lyons BP. YouTube as a source of information about the simulated surgical procedure. Clin Orthop Relat Res. 2020;478(5):943-950.

[8] Gao J, Zheng P, Jia Y, Chen H, Mao Y, Chen S, et al. Mental health problems and social media exposure during COVID-19 outbreak. PLoS One. 2020;15(4):e0231924.

[9] LaTour MS, Rotfeld HJ. There are threats and (maybe) fear-caused arousal: Theory and confusions of appeals to fear and fear arousal itself. J Advert. 1997;26(3):45-59.

[10] Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies for causal effects. Biometrika. 1983;70(1):41-55.

[11] Austin PC. An introduction to propensity score methods for reducing the effects of confounding in observational studies. Multivariate Behav Res. 2011;46(3):399-424.

[12] Rosenbaum PR. Observational Studies. 2nd ed. New York: Springer; 2002.

[13] Wooldridge JM. Econometric Analysis of Cross Section and Panel Data. 2nd ed. Cambridge: MIT Press; 2010.

[14] Card D, Krueger AB. Minimum wages and employment: A case study of the fast-food industry in New Jersey and Pennsylvania. Am Econ Rev. 1994;84(4):772-793.

[15] Wang L, Yang N, Huang X, Yang L, Majumder R, Wei F. Multilingual E5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672; 2024.

[16] Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Hillsdale: Lawrence Erlbaum; 1988.

[17] Pornpitakpan C. The persuasiveness of source credibility: A critical review of five decades' evidence. J Appl Soc Psychol. 2004;34(2):243-281.

---

*본 연구의 데이터 수집 및 분석에 사용된 코드와 데이터는 GitHub (https://github.com/YunhuPark/Medical_Insight_Lab)에 공개되어 있으며, YouTube 서비스 이용약관의 범위 내에서 수집되었습니다.*

---

**교신저자 (Corresponding Author)**  
박윤후  
인제대학교 의료IT학과  
E-mail: byunhu35@gmail.com

*이해충돌: 없음*

*연구비 지원: 없음*
