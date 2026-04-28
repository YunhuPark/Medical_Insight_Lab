# Data Dictionary — Medical Insight Lab

## 4_medical_full_dataset.csv (Main Dataset)

3,713 rows × 11 columns. Unit of observation: one YouTube video.

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `Video_ID` | string | YouTube 11-char ID | Unique identifier for each video. Primary key. |
| `Keyword` | string | 37 disease categories | Search keyword used to discover the video (e.g., "당뇨", "고혈압", "암"). |
| `Title` | string | — | Video title as returned by YouTube Data API v3. |
| `Channel` | string | 921 unique channels | YouTube channel name (display name, not channel ID). |
| `Views` | int | 101 – 9,985,009 | Total cumulative view count at time of collection (2026-04-19). |
| `Likes` | int | 0 – 107,767 | Total like count at collection time. |
| `Published_At` | datetime (UTC) | 2013 – 2026 | ISO 8601 timestamp of video publication. |
| `Duration_Sec` | int | 5 – 42,899 | Video duration in seconds. Videos ≤ 60 sec are classified as Shorts. |
| `Medical_Score` | float | 0.5 – 115.4 | Heuristic credibility score based on title term frequency analysis. Higher = more medically specific vocabulary. Used as a proxy for channel expertise level alongside `Type`. |
| `Type` | string | `Medical Pro`, `General` | Channel type label. `Medical Pro`: channel operated by licensed medical professionals or hospitals. `General`: health/wellness content creators without stated clinical credentials. |
| `Source_Type` | string | `Description`, `PreCovid_Supplement` | Data collection method. `Description`: collected via YouTube search API (main dataset). `PreCovid_Supplement`: targeted collection of 2018–2019 videos to address pre-COVID data imbalance (606 rows added). |

---

## enriched_base.csv (Feature-Engineered Dataset)

Same 3,713 rows with derived features added during analysis. Superset of `4_medical_full_dataset.csv`.

| Column | Description |
|--------|-------------|
| `Age_Days` | Days since publication as of 2026-04-19. Minimum 1. |
| `VPD` | Views Per Day = `Views / Age_Days`. Normalized engagement rate, controls for video age bias. |
| `Log_Views` | `log1p(Views)`. Used in regression models to reduce skew (raw Views skewness ≈ 7.2). |
| `Fear_Score` | Count of fear/risk keywords found in `Title`. Keywords: 충격, 경악, 사망, 암, 위험, 절대, 금지, 무시, 신호, 전조, 증상, 말기, 시한부, 응급, 마비, 실명, 절단, 투석, 쇼크, 발작, 최악, 경고, 주의, 폭발, 급증, 심각, 치명, 돌연, 급격 (30 keywords). |
| `Has_Fear` | Binary treatment indicator. 1 if `Fear_Score > 0`, else 0. Used as treatment in PSM and DiD. |
| `Is_Shorts` | 1 if `Duration_Sec ≤ 60`, else 0. Shorts use a separate recommendation algorithm. |
| `Like_Rate` | `Likes / (Views + 1)`. Engagement quality proxy. |
| `Ch_Median` | Median views of all videos in the same channel. Used as channel-size covariate in PSM. |
| `Log_Ch_Med` | `log1p(Ch_Median)`. Used in logistic regression for propensity score estimation. |
| `Year` | Publication year extracted from `Published_At`. |

---

## comments.csv (Comment Dataset)

19,232 rows. One row per comment. 840 unique videos covered (33% of main dataset).

| Column | Type | Description |
|--------|------|-------------|
| `Video_ID` | string | Foreign key → `4_medical_full_dataset.csv.Video_ID`. |
| `Comment` | string | Raw comment text (Korean). Not pre-processed. |
| `Keyword` | string | Disease category of the parent video. |
| `Has_Fear` | int | Fear keyword indicator of the parent video's title (0/1). |

---

## channel_stats.csv (Channel Statistics)

920 rows. One row per channel (920/921 channels; 1 channel had no accessible data).

| Column | Type | Description |
|--------|------|-------------|
| `Channel` | string | Channel name. Foreign key → main dataset. |
| `Subscribers` | int | Subscriber count at collection time (2026-04-19). |
| `Total_Views` | int | Cumulative channel-level view count. |

---

## thumbnail_features.csv (Computer Vision Features)

2,372 rows (thumbnails successfully downloaded and processed by OpenCV).

| Column | Type | Description |
|--------|------|-------------|
| `Video_ID` | string | Foreign key. |
| `Views` | int | View count (joined from main dataset). |
| `red_ratio` | float | Proportion of pixels in HSV red range. Correlated with clickbait style. |
| `brightness` | float | Mean pixel brightness (0–255 scale). |
| `contrast` | float | Standard deviation of grayscale pixel values. |
| `saturation` | float | Mean HSV saturation. |
| `edge_density` | float | Proportion of edge pixels detected via Canny algorithm. Proxy for visual complexity. |
| `text_ratio` | float | Estimated proportion of thumbnail area covered by text (via contour analysis). |

---

## Notes

- **Collection date**: All view/like counts are snapshots from 2026-04-19. This is a cross-sectional study — longitudinal trends are approximated via `Age_Days` / `VPD`.
- **Coverage limitation**: Comments cover 33% of videos (840/2,512 non-Shorts videos). Comment analyses should be interpreted with this caveat.
- **Shorts exclusion**: For causal analyses (PSM, Fixed Effects, DiD), Shorts (`Is_Shorts = 1`) are excluded by default. They use a separate recommendation algorithm, violating comparability assumptions.
- **Medical_Score**: Not used as a primary variable in hypothesis testing. Used only as a covariate in PSM to control for content type differences.
