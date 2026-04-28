"""
의료 유튜브 분석 대시보드
실행: venv/Scripts/streamlit run dashboard.py
"""
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ── 페이지 설정
st.set_page_config(
    page_title="의료 유튜브 분석",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 색상
C = {"red": "#e94560", "blue": "#0f3460", "green": "#27ae60",
     "gray": "#888888", "gold": "#f5a623", "bg": "#0a0a1a"}

# ── 전역 CSS
st.markdown("""
<style>
  .metric-card{background:#1a1a3e;border-radius:12px;padding:20px;text-align:center;border:1px solid #e94560;}
  .metric-val{font-size:2em;font-weight:700;color:#e94560;}
  .metric-lbl{color:#aaa;font-size:.85em;margin-top:4px;}
  .insight-box{background:#1a2a1a;border-left:4px solid #27ae60;padding:14px 18px;border-radius:8px;margin:8px 0;color:#e0e0e0;}
  .warn-box{background:#2a1a1a;border-left:4px solid #e94560;padding:14px 18px;border-radius:8px;margin:8px 0;color:#e0e0e0;}
  .info-box{background:#1a1a2a;border-left:4px solid #0f3460;padding:14px 18px;border-radius:8px;margin:8px 0;color:#e0e0e0;}
  div[data-testid="stMetricValue"]{color:#e94560;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
FEAR_WORDS = ['충격','경악','사망','암','위험','절대','금지','무시','신호','전조',
              '증상','말기','시한부','응급','마비','실명','절단','투석','쇼크',
              '발작','최악','경고','주의','폭발','급증','심각','치명','돌연','급격']

# ── 데이터 로드 (캐시)
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "src/4_medical_full_dataset.csv"))
    NOW = datetime(2026, 4, 19, tzinfo=timezone.utc)
    df["Published_At"] = pd.to_datetime(df["Published_At"], utc=True)
    df["Year"]       = df["Published_At"].dt.year
    df["Age_Days"]   = (NOW - df["Published_At"]).dt.days.clip(lower=1)
    df["VPD"]        = df["Views"] / df["Age_Days"]
    df["Log_Views"]  = np.log1p(df["Views"])
    t = df["Title"].astype(str)
    df["Fear_Score"] = t.apply(lambda x: sum(1 for w in FEAR_WORDS if w in x))
    df["Has_Fear"]   = (df["Fear_Score"] > 0).astype(int)
    df["Is_Shorts"]  = (df["Duration_Sec"] <= 60).astype(int)
    df["Like_Rate"]  = df["Likes"] / (df["Views"] + 1)
    ch_med = df.groupby("Channel")["Views"].median()
    df["Ch_Median"]  = df["Channel"].map(ch_med)
    return df

@st.cache_data
def load_comments():
    p = os.path.join(BASE, "src/comments.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_channel_stats():
    p = os.path.join(BASE, "src/channel_stats.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_thumbnail_features():
    p = os.path.join(BASE, "src/thumbnail_features.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

df  = load_data()
com = load_comments()
ch  = load_channel_stats()
tf  = load_thumbnail_features()

# ══════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 의료 유튜브 분석")
    st.markdown("**공포 소구의 진실**: 단순 상관에서 인과 추정까지")
    st.markdown("---")

    st.markdown("### 📊 데이터 필터")
    sel_type  = st.multiselect("채널 유형", ["General", "Medical Pro"],
                                default=["General", "Medical Pro"])
    sel_years = st.slider("업로드 연도", 2016, 2025, (2016, 2025))
    sel_shorts = st.radio("영상 유형", ["전체", "일반 영상만", "Shorts만"], index=1)

    df_f = df[df["Type"].isin(sel_type) &
              df["Year"].between(*sel_years)].copy()
    if sel_shorts == "일반 영상만":
        df_f = df_f[df_f["Is_Shorts"] == 0]
    elif sel_shorts == "Shorts만":
        df_f = df_f[df_f["Is_Shorts"] == 1]

    st.markdown("---")
    st.markdown(f"**필터 결과**: {len(df_f):,}개 영상")
    st.markdown(f"채널 {df_f['Channel'].nunique()}개 · 카테고리 {df_f['Keyword'].nunique()}개")

    st.markdown("---")
    st.markdown("### 📁 프로젝트 정보")
    st.markdown("- **분석 기준일**: 2026-04-19")
    st.markdown("- **수집 기간**: 2013 – 2026")
    st.markdown("- **분석 방법**: 18가지")

# ══════════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔥 Fear Analysis", "🔬 Causal Inference",
    "📡 Channel Explorer", "🎨 Multimodal", "💡 전략 가이드"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("## 📊 데이터셋 개요")

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (f"{len(df):,}", "전체 영상"),
        (f"{df['Channel'].nunique():,}", "채널 수"),
        (f"{df['Keyword'].nunique()}", "질환 카테고리"),
        (f"{len(com):,}" if not com.empty else "–", "수집 댓글"),
        (f"{df['Has_Fear'].mean()*100:.1f}%", "공포 키워드 비율"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 연도별 영상 수")
        yr = df.groupby(["Year", "Source_Type"]).size().reset_index(name="count")
        yr = yr[yr["Year"].between(2016, 2025)]
        fig = px.bar(yr, x="Year", y="count", color="Source_Type",
                     color_discrete_map={"Description": C["blue"], "PreCovid_Supplement": C["gold"]},
                     labels={"count": "영상 수", "Source_Type": "출처"},
                     barmode="stack")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", legend_title_text="")
        fig.add_vline(x=2019.5, line_dash="dash", line_color=C["red"],
                      annotation_text="COVID-19", annotation_font_color=C["red"])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 카테고리별 중앙 조회수")
        kw_med = df_f.groupby("Keyword")["Views"].median().sort_values(ascending=True)
        fig2 = px.bar(x=kw_med.values / 1e4, y=kw_med.index, orientation="h",
                      labels={"x": "중앙 조회수 (만회)", "y": ""},
                      color=kw_med.values, color_continuous_scale="Reds")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", showlegend=False, coloraxis_showscale=False,
                           height=450)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 조회수 분포")
        fig3 = px.histogram(df_f, x="Log_Views", nbins=50, color_discrete_sequence=[C["red"]])
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", xaxis_title="log(조회수)", yaxis_title="빈도")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"""<div class="info-box">
        조회수 왜도: <b>{df_f['Views'].skew():.1f}</b><br>
        → 극심한 우편향 → 비모수 검정 필요
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("### 채널 유형 분포")
        type_cnt = df_f["Type"].value_counts()
        fig4 = px.pie(values=type_cnt.values, names=type_cnt.index,
                      color_discrete_sequence=[C["blue"], C["red"]])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig4, use_container_width=True)

    with col_c:
        st.markdown("### Shorts 비율 추이")
        yr_s = df[df["Year"].between(2019, 2025)].groupby("Year")["Is_Shorts"].mean() * 100
        fig5 = px.line(x=yr_s.index, y=yr_s.values, markers=True,
                       labels={"x": "연도", "y": "Shorts 비율 (%)"},
                       color_discrete_sequence=[C["red"]])
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white")
        st.plotly_chart(fig5, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: FEAR ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("## 🔥 공포 키워드 분석")

    st.markdown("""<div class="warn-box">
    ⚠️ <b>결론 미리보기</b>: 단순 비교에서 2.1배로 보이던 공포 키워드 효과는,
    영상 나이 보정 후 <b>p=0.9999로 소멸</b>합니다. 채널 규모를 통제(PSM)하면 오히려 <b>0.63배 역효과</b>.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 공포 키워드 포함 비율 (카테고리별)")
        kw_fear = df_f.groupby("Keyword")["Has_Fear"].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=kw_fear.values, y=kw_fear.index, orientation="h",
                     color=kw_fear.values, color_continuous_scale="Reds",
                     labels={"x": "공포 키워드 비율 (%)", "y": ""})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 공포 키워드 유무별 조회수 분포")
        fear_v   = df_f[df_f["Has_Fear"] == 1]["Log_Views"]
        nofear_v = df_f[df_f["Has_Fear"] == 0]["Log_Views"]
        stat, p  = stats.mannwhitneyu(fear_v, nofear_v, alternative="greater")
        ratio    = np.exp(fear_v.median() - nofear_v.median())

        fig2 = go.Figure()
        fig2.add_trace(go.Violin(y=fear_v, name=f"공포 포함 (n={len(fear_v)})",
                                  fillcolor=C["red"], line_color=C["red"], opacity=0.7,
                                  box_visible=True, meanline_visible=True))
        fig2.add_trace(go.Violin(y=nofear_v, name=f"공포 미포함 (n={len(nofear_v)})",
                                  fillcolor=C["blue"], line_color=C["blue"], opacity=0.7,
                                  box_visible=True, meanline_visible=True))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", yaxis_title="log(조회수)", violingap=0.3)
        st.plotly_chart(fig2, use_container_width=True)

        sig = "✅ 유의 (p<0.05)" if p < 0.05 else "❌ 비유의 (p≥0.05)"
        st.markdown(f"""<div class="info-box">
        단순 비교 (Mann-Whitney U)<br>
        공포 포함 중앙: <b>{fear_v.median():.2f}</b> | 미포함: <b>{nofear_v.median():.2f}</b><br>
        배율: <b>{ratio:.2f}배</b> | p = {p:.4f} → {sig}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 나이 보정 비교 (Views/Day)")

    col3, col4 = st.columns(2)
    with col3:
        fear_vpd   = df_f[df_f["Has_Fear"] == 1]["VPD"]
        nofear_vpd = df_f[df_f["Has_Fear"] == 0]["VPD"]
        _, p_vpd   = stats.mannwhitneyu(fear_vpd, nofear_vpd, alternative="greater")

        fig3 = go.Figure()
        fig3.add_trace(go.Box(y=np.log1p(fear_vpd), name="공포 포함",
                              fillcolor=C["red"], line_color=C["red"], opacity=0.8))
        fig3.add_trace(go.Box(y=np.log1p(nofear_vpd), name="공포 미포함",
                              fillcolor=C["blue"], line_color=C["blue"], opacity=0.8))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", yaxis_title="log(Views/Day)",
                           title=f"Views/Day 기준 — p={p_vpd:.4f}")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 나이 보정 전후 비교")
        raw_ratio = df_f[df_f["Has_Fear"]==1]["Views"].median() / df_f[df_f["Has_Fear"]==0]["Views"].median()
        vpd_ratio = fear_vpd.median() / nofear_vpd.median() if nofear_vpd.median() > 0 else 1

        fig4 = go.Figure(go.Bar(
            x=["원본 조회수\n(나이 미보정)", "Views/Day\n(나이 보정)"],
            y=[raw_ratio, vpd_ratio],
            marker_color=[C["blue"], C["red"]],
            text=[f"{raw_ratio:.2f}배", f"{vpd_ratio:.2f}배"],
            textposition="outside",
        ))
        fig4.add_hline(y=1, line_dash="dash", line_color="white")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", yaxis_title="공포 포함/미포함 배율",
                           yaxis_range=[0, max(raw_ratio, vpd_ratio) * 1.3])
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(f"""<div class="warn-box">
        나이 보정 후 효과 <b>{'감소' if vpd_ratio < raw_ratio else '유지'}</b><br>
        {raw_ratio:.2f}배 → {vpd_ratio:.2f}배 | p={p_vpd:.4f}<br>
        → {'효과 소멸: 영상 나이 편향이 원인이었음' if p_vpd >= 0.05 else '효과 유지됨'}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔑 공포 키워드 출현 빈도 TOP 20")
    kw_counts = {}
    for w in FEAR_WORDS:
        cnt = df_f["Title"].astype(str).str.contains(w).sum()
        if cnt > 0:
            kw_counts[w] = cnt
    kw_df = pd.DataFrame(list(kw_counts.items()), columns=["키워드", "출현수"]).sort_values("출현수", ascending=False).head(20)
    fig5 = px.bar(kw_df, x="키워드", y="출현수", color="출현수",
                  color_continuous_scale="Reds")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="white", coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: CAUSAL INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("## 🔬 인과 추정: 채널 규모를 통제하면?")

    st.markdown("""<div class="info-box">
    <b>핵심 질문</b>: 공포 키워드 효과가 진짜인가, 아니면 "대형 채널이 공포 제목을 많이 쓰는" 패턴의 착시인가?<br>
    PSM과 채널 고정효과라는 두 가지 독립적 방법으로 검증합니다.
    </div>""", unsafe_allow_html=True)

    # ── 방법론 비교 요약 차트
    st.markdown("### 방법론별 공포 키워드 효과 추정치")
    methods = ["단순 비교", "나이 보정\n(VPD)", "PSM\n매칭", "채널\n고정효과"]
    effects = [2.1, 0.53, 0.63, 1.07]
    pvals   = [0.001, 0.9999, 0.019, 0.306]
    colors  = [C["blue"] if p < 0.05 else C["gray"] for p in pvals]
    colors[2] = C["red"]  # PSM 역효과 강조

    fig = go.Figure()
    fig.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=1.5,
                  annotation_text="효과 없음 (1.0)", annotation_font_color="white")
    fig.add_trace(go.Bar(x=methods, y=effects, marker_color=colors,
                         text=[f"{e:.2f}배<br>p={p:.3f}" for e, p in zip(effects, pvals)],
                         textposition="outside"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", yaxis_title="공포 포함/미포함 배율",
                      yaxis_range=[0, 2.8],
                      title="빨강=유의 역효과 | 회색=비유의 | 파랑=유의 양효과")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PSM — 채널 규모 균형 확인")
        df_psm = df_f[["Channel", "Has_Fear", "Log_Views", "Age_Days", "Title"]].copy()
        df_psm["Title_Length"] = df_f["Title"].astype(str).str.len()
        df_psm["Log_Ch"] = np.log1p(df_f["Ch_Median"])
        df_psm["Log_Age"] = np.log1p(df_f["Age_Days"])
        df_psm = df_psm.dropna()

        X_cov = df_psm[["Log_Ch", "Log_Age", "Title_Length"]].astype(float)
        T = df_psm["Has_Fear"]
        if len(T.unique()) == 2 and T.sum() > 5:
            sc = StandardScaler()
            X_sc = sc.fit_transform(X_cov)
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_sc, T)
            ps = lr.predict_proba(X_sc)[:, 1]
            df_psm["PS"] = ps

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=ps[T==0], name="공포 미포함", opacity=0.6,
                                         marker_color=C["blue"], nbinsx=30, histnorm="probability"))
            fig2.add_trace(go.Histogram(x=ps[T==1], name="공포 포함", opacity=0.6,
                                         marker_color=C["red"], nbinsx=30, histnorm="probability"))
            fig2.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                               xaxis_title="성향 점수", yaxis_title="비율",
                               title="성향 점수 분포 (공통 지지 구간)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("PSM 계산을 위한 데이터가 부족합니다.")

    with col2:
        st.markdown("### DiD — COVID-19 자연실험")
        df_did = df[df["Year"].between(2015, 2025)].copy()
        df_did["Post"] = (df_did["Year"] >= 2020).astype(int)
        yr_trend = df_did.groupby(["Year", "Has_Fear"])["VPD"].median().reset_index()
        yr_trend.columns = ["Year", "Has_Fear", "VPD"]

        fig3 = go.Figure()
        for hf, name, col in [(0, "공포 미포함", C["blue"]), (1, "공포 포함", C["red"])]:
            sub = yr_trend[yr_trend["Has_Fear"] == hf]
            fig3.add_trace(go.Scatter(x=sub["Year"], y=sub["VPD"], mode="lines+markers",
                                      name=name, line=dict(color=col, width=2),
                                      marker=dict(size=7)))
        fig3.add_vline(x=2019.5, line_dash="dash", line_color="orange",
                       annotation_text="COVID-19", annotation_font_color="orange")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", xaxis_title="연도",
                           yaxis_title="중앙값 Views/Day",
                           title="DiD: 연도별 Views/Day 추이")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""<div class="info-box">
        <b>DiD 결과 (pre-COVID 데이터 보완 후)</b><br>
        추정치: +3.4% | Bootstrap 95% CI: [-20.8%, +34.7%]<br>
        → 불확실성 구간이 0 포함 → <b>비유의</b><br>
        초기 +42.5%는 pre-COVID 샘플 부족(73개)의 추정 불안정이었음
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 채널 유형별 공포 효과 (Type × Fear)")
    col3, col4 = st.columns(2)

    with col3:
        type_results = []
        for t in ["General", "Medical Pro"]:
            sub = df_f[df_f["Type"] == t]
            f  = sub[sub["Has_Fear"]==1]["Views"]
            nf = sub[sub["Has_Fear"]==0]["Views"]
            if len(f) >= 5 and len(nf) >= 5:
                _, pv = stats.mannwhitneyu(f, nf, alternative="two-sided")
                ratio = f.median() / nf.median() if nf.median() > 0 else 1
                type_results.append({"유형": t, "배율": ratio, "p값": pv,
                                      "n(공포)": len(f), "n(미포함)": len(nf)})
        if type_results:
            tr_df = pd.DataFrame(type_results)
            fig4 = go.Figure(go.Bar(
                x=tr_df["유형"], y=tr_df["배율"],
                marker_color=[C["blue"] if r > 1 else C["red"] for r in tr_df["배율"]],
                text=[f"{r:.2f}배<br>p={p:.3f}" for r, p in zip(tr_df["배율"], tr_df["p값"])],
                textposition="outside"
            ))
            fig4.add_hline(y=1.0, line_dash="dash", line_color="white")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", yaxis_title="공포 포함/미포함 배율",
                               yaxis_range=[0, max(tr_df["배율"])*1.4])
            st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown("#### 해석")
        st.markdown("""<div class="warn-box">
        <b>Medical Pro 채널</b>: 공포 키워드 → 조회수 <b>감소</b> (0.83배, p=0.017 유의)<br>
        전문 의료 채널 시청자는 권위 있는 어조를 기대 — 선정적 제목은 신뢰도 저하
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        <b>General 채널</b>: 1.14배, p=0.558 비유의<br>
        효과 없음 — 긍정도 부정도 아님
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box">
        ✅ <b>전략적 시사점</b><br>
        의료 전문가 채널일수록 공포 키워드 <b>사용 자제</b>가 유리<br>
        일반 채널도 공포 키워드로 특별한 이득 없음
        </div>""", unsafe_allow_html=True)

    # ── SCI급 통계 섹션
    st.markdown("---")
    st.markdown("### 🔬 SCI-Level Statistical Reporting")
    st.markdown("*효과크기 · PSM 균형 진단 · 평행추세 검정 · 로젠바움 민감도*")

    sci_col1, sci_col2 = st.columns(2)

    with sci_col1:
        # ── 효과크기 (Hedge's g + rank-r)
        st.markdown("#### 효과크기 요약 (Hedge's g)")
        eff_data = {
            "비교": ["단순 비교\n(Log_Views)", "나이 보정\n(Log_VPD)", "General\n(Type×Fear)", "Medical Pro\n(Type×Fear)"],
            "g":    [-0.053, -0.164, 0.071, -0.269],
            "r":    [0.036, 0.093, -0.041, 0.133],
            "power":[0.306, 0.963, 0.318, 0.618],
            "p":    [0.147, 0.000, 0.137, 0.024],
        }
        eff_df = pd.DataFrame(eff_data)
        fig_eff = go.Figure()
        colors_eff = [C["red"] if abs(g) >= 0.2 else C["gray"] for g in eff_df["g"]]
        fig_eff.add_trace(go.Bar(
            y=eff_df["비교"], x=eff_df["g"],
            orientation="h",
            marker_color=colors_eff,
            text=[f"g={g:+.3f} | power={pw:.2f} | p={p:.3f}"
                  for g, pw, p in zip(eff_df["g"], eff_df["power"], eff_df["p"])],
            textposition="outside",
        ))
        fig_eff.add_vline(x=0,   line_dash="dash",  line_color="white", line_width=1)
        fig_eff.add_vline(x=0.2, line_dash="dot",   line_color="#27ae60", line_width=1)
        fig_eff.add_vline(x=-0.2,line_dash="dot",   line_color="#27ae60", line_width=1)
        fig_eff.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="Hedge's g",
            title="빨강=|g|≥0.2 (small+) | 녹색점선=±0.2 임계값",
            height=280, margin=dict(l=130, r=10)
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with sci_col2:
        # ── PSM Love Plot (SMD)
        st.markdown("#### PSM Love Plot — 공변량 균형 (SMD)")
        smd_data = {
            "공변량":    ["Log_Ch_Med", "Log_Age", "Title_Length", "Has_Number", "Medical_Score"],
            "SMD_전":   [-0.079,  0.249, 0.079,  0.002, -0.242],
            "SMD_후":   [-0.159,  0.053, -0.010, 0.217, -0.015],
        }
        smd_d = pd.DataFrame(smd_data)
        fig_smd = go.Figure()
        fig_smd.add_trace(go.Scatter(
            x=smd_d["SMD_전"], y=smd_d["공변량"],
            mode="markers", name="매칭 전",
            marker=dict(color=C["red"], size=12, symbol="diamond")
        ))
        fig_smd.add_trace(go.Scatter(
            x=smd_d["SMD_후"], y=smd_d["공변량"],
            mode="markers", name="매칭 후",
            marker=dict(color="#27ae60", size=12, symbol="circle")
        ))
        for i, row in smd_d.iterrows():
            fig_smd.add_trace(go.Scatter(
                x=[row["SMD_전"], row["SMD_후"]], y=[row["공변량"], row["공변량"]],
                mode="lines", line=dict(color="gray", width=1), showlegend=False
            ))
        fig_smd.add_vline(x=0,    line_dash="dash", line_color="white",   line_width=1.5)
        fig_smd.add_vline(x=0.1,  line_dash="dot",  line_color="#f5a623", line_width=1)
        fig_smd.add_vline(x=-0.1, line_dash="dot",  line_color="#f5a623", line_width=1)
        fig_smd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="SMD (|<0.1| = 균형)",
            title="황색점선=±0.1 임계값 / Log_Ch_Med·Has_Number 불균형 주의",
            height=280, margin=dict(l=130, r=10)
        )
        st.plotly_chart(fig_smd, use_container_width=True)

    sci_col3, sci_col4 = st.columns(2)

    with sci_col3:
        # ── 평행추세 검정
        st.markdown("#### DiD 평행추세 가정 검정 (2015–2019)")
        pre_trends = df[(df["Year"].between(2015, 2019)) & (df["Is_Shorts"] == 0)].copy()
        pt_grp = pre_trends.groupby(["Year", "Has_Fear"])["VPD"].mean().reset_index()
        fig_pt = go.Figure()
        for hf, label, col_ in [(0, "공포 미포함", C["blue"]), (1, "공포 포함", C["red"])]:
            sub = pt_grp[pt_grp["Has_Fear"] == hf]
            fig_pt.add_trace(go.Scatter(
                x=sub["Year"], y=sub["VPD"], mode="lines+markers",
                name=label, line=dict(color=col_, width=2.5), marker=dict(size=8)
            ))
            if len(sub) >= 2:
                z = np.polyfit(sub["Year"], sub["VPD"], 1)
                fig_pt.add_trace(go.Scatter(
                    x=sub["Year"], y=np.poly1d(z)(sub["Year"]),
                    mode="lines", line=dict(color=col_, dash="dash", width=1),
                    showlegend=False
                ))
        fig_pt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="연도 (Pre-COVID)",
            yaxis_title="평균 Views/Day",
            title="상호작용항 p=0.941 → ✓ 평행추세 충족",
            height=280
        )
        st.plotly_chart(fig_pt, use_container_width=True)

    with sci_col4:
        # ── 로젠바움 민감도
        st.markdown("#### 로젠바움 민감도 분석 (PSM 강건성)")
        rb_data = {
            "Gamma": [1.00, 1.05, 1.10, 1.25, 1.50, 2.00, 3.00],
            "p_upper": [0.0001, 0.0009, 0.0061, 0.2404, 1.0, 1.0, 1.0],
        }
        rb_df = pd.DataFrame(rb_data)
        bar_c = [C["green"] if p < 0.05 else C["red"] for p in rb_df["p_upper"]]
        fig_rb = go.Figure(go.Bar(
            x=rb_df["Gamma"], y=rb_df["p_upper"].clip(upper=1.0),
            marker_color=bar_c, width=0.08,
            text=[f"p={p:.4f}" if p < 0.5 else "p>0.5" for p in rb_df["p_upper"]],
            textposition="outside"
        ))
        fig_rb.add_hline(y=0.05, line_dash="dash", line_color="#f5a623",
                          line_width=2, annotation_text="α=0.05")
        fig_rb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="Γ (숨겨진 편향 크기)",
            yaxis_title="p값 상한", yaxis_range=[0, 1.15],
            title="Γ≈1.15에서 결과 번복 → PSM 결과는 보수적 해석 필요",
            height=280
        )
        st.plotly_chart(fig_rb, use_container_width=True)

    st.markdown("""<div class="info-box">
    <b>SCI 수준 해석 요약</b><br>
    • <b>효과크기</b>: Medical Pro g=-0.269 (small 수준) — 통계적으로 유의하나 실용적 크기는 제한적<br>
    • <b>PSM 균형</b>: 5개 공변량 중 3개 균형 달성, 2개(Log_Ch_Med, Has_Number) 불균형 → 한계로 명시 필요<br>
    • <b>평행추세</b>: p=0.941 → DiD 핵심 가정 강하게 충족<br>
    • <b>민감도</b>: Γ≈1.15에서 결과 번복 → 숨겨진 혼란변수 15% 이상이면 인과 해석 위험 → 관찰 연구의 본질적 한계
    </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: CHANNEL EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("## 📡 채널 탐색기")

    if not ch.empty:
        df_ch = df.merge(ch[["Channel","Subscribers","Total_Views"]], on="Channel", how="left")
        df_ch = df_ch[df_ch["Subscribers"].notna() & (df_ch["Subscribers"] > 0)]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 구독자 수 분포")
            fig = px.histogram(df_ch, x=np.log10(df_ch["Subscribers"]+1),
                                nbins=40, color_discrete_sequence=[C["blue"]],
                                labels={"x": "log₁₀(구독자 수)"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 구독자 규모 × 조회수")
            sample = df_ch.sample(min(500, len(df_ch)), random_state=42)
            fig2 = px.scatter(sample,
                              x=np.log10(sample["Subscribers"]+1),
                              y="Log_Views",
                              color="Has_Fear",
                              color_discrete_map={0: C["blue"], 1: C["red"]},
                              opacity=0.5, size_max=6,
                              labels={"x": "log₁₀(구독자)", "y": "log(조회수)",
                                      "Has_Fear": "공포 키워드"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 시장 집중도")
        ch_total = df.groupby("Channel")["Views"].sum().sort_values(ascending=False)
        top10 = ch_total.head(10).sum() / ch_total.sum() * 100
        top50 = ch_total.head(50).sum() / ch_total.sum() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("상위 10개 채널 점유율", f"{top10:.1f}%")
        c2.metric("상위 50개 채널 점유율", f"{top50:.1f}%")
        c3.metric("나머지 채널 점유율", f"{100-top50:.1f}%")

        # 누적 조회수 Lorenz curve
        st.markdown("### 조회수 집중 곡선 (Lorenz Curve)")
        sorted_views = np.sort(ch_total.values)
        cum_views = np.cumsum(sorted_views) / sorted_views.sum()
        cum_ch    = np.arange(1, len(sorted_views)+1) / len(sorted_views)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cum_ch*100, y=cum_views*100, fill="tozeroy",
                                   name="실제 분포", line=dict(color=C["red"], width=2)))
        fig3.add_trace(go.Scatter(x=[0,100], y=[0,100], name="완전 평등",
                                   line=dict(color="white", dash="dash", width=1)))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", xaxis_title="채널 누적 비율 (%)",
                           yaxis_title="조회수 누적 비율 (%)")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""<div class="warn-box">
        상위 10개 채널이 전체 조회수의 33%를 독식 — 극심한 불평등 구조<br>
        신규 채널이 제목 전략만으로 조회수를 늘리기 어렵다는 구조적 근거
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 채널별 탐색")
        top_channels = ch_total.head(30).index.tolist()
        sel_ch = st.selectbox("채널 선택 (조회수 상위 30)", top_channels)
        ch_data = df[df["Channel"] == sel_ch].sort_values("Views", ascending=False)
        st.dataframe(ch_data[["Title","Keyword","Views","Likes","Published_At","Has_Fear","Type"]].head(10),
                     use_container_width=True)
    else:
        st.warning("channel_stats.csv 파일이 없습니다.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: MULTIMODAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("## 🎨 멀티모달 분석")

    subtab1, subtab2, subtab3 = st.tabs(["썸네일 CV", "댓글 감성", "KoBERT 클러스터"])

    with subtab1:
        st.markdown("### 썸네일 시각적 피처 × 조회수")
        if not tf.empty:
            tf2 = tf.copy()
            tf2["Log_Views"] = np.log1p(tf2["Views"])
            metrics_t = {
                "red_ratio": "빨간색 비율",
                "brightness": "밝기",
                "contrast": "대비",
                "saturation": "채도",
                "edge_density": "엣지 밀도",
                "text_ratio": "텍스트 비율",
            }
            avail = [k for k in metrics_t if k in tf2.columns]
            results = []
            for m in avail:
                r, p = stats.spearmanr(tf2[m], tf2["Log_Views"])
                results.append({"피처": metrics_t[m], "Spearman r": r, "p값": p,
                                 "유의": p < 0.05})
            res_df = pd.DataFrame(results).sort_values("Spearman r", ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(res_df, x="Spearman r", y="피처", orientation="h",
                             color="유의",
                             color_discrete_map={True: C["red"], False: C["gray"]},
                             labels={"Spearman r": "Spearman r (조회수와 상관)"},
                             title="썸네일 피처-조회수 상관관계")
                fig.add_vline(x=0, line_dash="dash", line_color="white")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if avail:
                    sel_feat = st.selectbox("피처 선택", avail,
                                            format_func=lambda x: metrics_t.get(x, x))
                    fig2 = px.scatter(tf2.sample(min(300, len(tf2)), random_state=42),
                                      x=sel_feat, y="Log_Views", opacity=0.4,
                                      color_discrete_sequence=[C["red"]],
                                      trendline="ols",
                                      labels={"Log_Views": "log(조회수)",
                                              sel_feat: metrics_t.get(sel_feat, sel_feat)})
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="white")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("thumbnail_features.csv 없음")

    with subtab2:
        st.markdown("### 댓글 감성 분석")
        if not com.empty:
            POS_WORDS = ["감사","도움","좋아요","최고","덕분","완치","나았","효과","좋은","고마워"]
            NEG_WORDS = ["무서워","걱정","겁","불안","두려","충격","슬프","힘들","병원","아파"]
            FEAR_COM  = ["무서워","겁나","두려","충격","공포","경악"]

            com2 = com.copy()
            c_text = com2["Comment"].apply(lambda x: str(x) if x is not None else "")
            com2["pos"] = c_text.apply(lambda x: any(w in x for w in POS_WORDS)).astype(int)
            com2["neg"] = c_text.apply(lambda x: any(w in x for w in NEG_WORDS)).astype(int)
            com2["fear_reaction"] = c_text.apply(lambda x: any(w in x for w in FEAR_COM)).astype(int)

            col1, col2 = st.columns(2)
            with col1:
                grp = com2.groupby("Has_Fear")[["pos","neg","fear_reaction"]].mean() * 100
                fig = go.Figure()
                cats = ["긍정 반응 (%)", "부정 반응 (%)", "공포 반응 (%)"]
                for i, (lbl, row) in enumerate(grp.iterrows()):
                    fig.add_trace(go.Bar(
                        name=f"{'공포 포함' if lbl == 1 else '공포 미포함'}",
                        x=cats, y=[row["pos"], row["neg"], row["fear_reaction"]],
                        marker_color=C["red"] if lbl == 1 else C["blue"]
                    ))
                fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                                  yaxis_title="비율 (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("분석 댓글 수", f"{len(com2):,}개")
                st.metric("공포 영상 댓글", f"{com2['Has_Fear'].sum():,}개")
                st.metric("공포 반응 비율", f"{com2['fear_reaction'].mean()*100:.1f}%")

                kw_sent = com2.groupby("Keyword")["neg"].mean().sort_values(ascending=False).head(10) * 100
                fig2 = px.bar(x=kw_sent.values, y=kw_sent.index, orientation="h",
                              color=kw_sent.values, color_continuous_scale="Reds",
                              labels={"x": "부정 댓글 비율 (%)", "y": ""},
                              title="카테고리별 부정 댓글 비율")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="white", coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            # ── 채널 유형 × 공포 교차 분석 (심화)
            st.markdown("---")
            st.markdown("### 📊 채널 유형 × 공포 키워드 — 댓글 반응 교차 분석")

            vid_key = "Video_ID" if "Video_ID" in com2.columns else None
            if vid_key and "Video_ID" in df.columns:
                com_m = com2.merge(df[["Video_ID","Type"]].drop_duplicates(), on="Video_ID", how="left")
            else:
                com_m = com2.copy()
                com_m["Type"] = com_m.get("Type", "Unknown")

            com_m = com_m[com_m["Type"].isin(["General","Medical Pro"])]

            if len(com_m) > 0:
                cross = com_m.groupby(["Type","Has_Fear"])[["pos","neg","fear_reaction"]].mean() * 100
                cross = cross.reset_index()
                cross["그룹"] = cross["Type"] + " · " + cross["Has_Fear"].map({0:"공포 없음", 1:"공포 포함"})

                col_a, col_b = st.columns(2)
                with col_a:
                    fig3 = go.Figure()
                    colors_grp = {
                        "General · 공포 없음":   "#4a90d9",
                        "General · 공포 포함":   "#1a5fa8",
                        "Medical Pro · 공포 없음": "#7ed56f",
                        "Medical Pro · 공포 포함": "#c0392b",
                    }
                    for _, row in cross.iterrows():
                        grp_name = row["그룹"]
                        fig3.add_trace(go.Bar(
                            name=grp_name,
                            x=["긍정 (%)", "부정 (%)", "공포 반응 (%)"],
                            y=[row["pos"], row["neg"], row["fear_reaction"]],
                            marker_color=colors_grp.get(grp_name, C["gray"])
                        ))
                    fig3.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white", yaxis_title="비율 (%)",
                        title="채널 유형 × 공포 키워드별 댓글 감성",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.5)
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                with col_b:
                    st.markdown("#### 통계 검정 — 공포 반응률 차이")
                    for ch_type in ["General", "Medical Pro"]:
                        sub = com_m[com_m["Type"] == ch_type]
                        f_  = sub[sub["Has_Fear"]==1]["fear_reaction"]
                        nf_ = sub[sub["Has_Fear"]==0]["fear_reaction"]
                        if len(f_) >= 5 and len(nf_) >= 5:
                            _, pv = stats.mannwhitneyu(f_, nf_, alternative="two-sided")
                            fr_ = f_.mean() * 100
                            nfr_ = nf_.mean() * 100
                            sig = "✅ 유의" if pv < 0.05 else "❌ 비유의"
                            box_cls = "insight-box" if pv < 0.05 else "info-box"
                            st.markdown(f"""<div class="{box_cls}">
                            <b>{ch_type}</b><br>
                            공포 포함 공포반응: <b>{fr_:.1f}%</b><br>
                            공포 미포함 공포반응: <b>{nfr_:.1f}%</b><br>
                            p={pv:.4f} → {sig}
                            </div>""", unsafe_allow_html=True)

                    st.markdown("""<div class="warn-box">
                    <b>해석 포인트</b><br>
                    공포 키워드 영상에 실제 공포 반응 댓글이 더 많은가?<br>
                    채널 유형에 따라 반응 패턴이 다른가?<br>
                    → 제목의 공포 소구가 시청자 감정에 미치는 실제 영향 검증
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("comments.csv 없음")

    with subtab3:
        st.markdown("### KoBERT 제목 클러스터 탐색기")
        emb_path = os.path.join(BASE, "src", "title_embeddings.npy")
        if os.path.exists(emb_path):
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            import numpy as np

            @st.cache_data
            def get_cluster_data():
                emb = np.load(emb_path)
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(emb)
                km = KMeans(n_clusters=9, random_state=42, n_init=10)
                labels = km.fit_predict(emb)
                return coords, labels

            coords, labels = get_cluster_data()
            df_emb = df[df["Source_Type"]=="Description"].reset_index(drop=True)
            n = min(len(coords), len(df_emb))
            plot_df = pd.DataFrame({
                "x": coords[:n, 0], "y": coords[:n, 1],
                "cluster": labels[:n].astype(str),
                "title": df_emb["Title"].values[:n],
                "views": df_emb["Views"].values[:n],
                "has_fear": df_emb["Has_Fear"].values[:n],
            })

            col1, col2 = st.columns([2, 1])
            with col1:
                color_by = st.radio("색상 기준", ["클러스터", "공포 키워드", "조회수"], horizontal=True)
                if color_by == "클러스터":
                    fig = px.scatter(plot_df.sample(min(1000, len(plot_df)), random_state=42),
                                     x="x", y="y", color="cluster", hover_data=["title"],
                                     opacity=0.6, title="KoBERT 제목 임베딩 (PCA 2D)")
                elif color_by == "공포 키워드":
                    fig = px.scatter(plot_df.sample(min(1000, len(plot_df)), random_state=42),
                                     x="x", y="y", color="has_fear",
                                     color_discrete_map={0: C["blue"], 1: C["red"]},
                                     hover_data=["title"], opacity=0.6,
                                     title="공포 키워드 분포 in 임베딩 공간")
                else:
                    fig = px.scatter(plot_df.sample(min(1000, len(plot_df)), random_state=42),
                                     x="x", y="y", color=np.log1p(plot_df["views"].sample(min(1000, len(plot_df)), random_state=42)),
                                     hover_data=["title"], opacity=0.6,
                                     color_continuous_scale="Reds",
                                     title="조회수 분포 in 임베딩 공간")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 클러스터별 중앙 조회수")
                cl_stats = plot_df.groupby("cluster")["views"].median().sort_values(ascending=False)
                for cl, v in cl_stats.items():
                    st.markdown(f"**Cluster {cl}**: {v/1e4:.1f}만회")
        else:
            st.info("임베딩 파일(title_embeddings.npy) 없음 — kobert_analysis.py 먼저 실행")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: STRATEGY GUIDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("## 💡 전략 가이드 — So What?")
    st.markdown("*분석 결과를 실제 채널 운영 전략으로 변환합니다.*")

    # ── 핵심 결론 3-box
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="warn-box">
        <h4 style="color:#e94560;margin-top:0">❌ 검증된 착시</h4>
        단순 비교 <b>+2.1배</b> 효과는<br>채널 규모 교란변수의 산물<br><br>
        대형 채널이 공포 키워드를 자주 사용하고,<br>대형 채널은 원래 조회수가 높음
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="warn-box">
        <h4 style="color:#e94560;margin-top:0">⚠️ 실제 효과 (PSM)</h4>
        채널 규모 통제 시<br><b>오히려 0.63배 감소</b> (p=0.019)<br><br>
        Bootstrap CI: [0.44, 0.85]<br>
        공포 키워드 = 조회수 감소 요인
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="insight-box">
        <h4 style="color:#27ae60;margin-top:0">✅ 채널 유형이 핵심</h4>
        <b>Medical Pro</b>: 0.83배 감소 (유의)<br>
        <b>General</b>: 1.14배 비유의<br><br>
        전문성 채널일수록 공포 소구가<br>신뢰도를 깎아먹음
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 채널 유형별 권장 전략
    st.markdown("### 채널 유형별 제목 전략 권장사항")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="warn-box">
        <h4 style="margin-top:0">🏥 의료 전문 채널 (Medical Pro)</h4>
        <b>공포 키워드 사용 자제</b> 강력 권장<br><br>
        ✅ <b>추천 패턴</b><br>
        &nbsp;&nbsp;• "고혈압 관리 3가지 핵심 전략"<br>
        &nbsp;&nbsp;• "당뇨 환자가 꼭 알아야 할 식단"<br>
        &nbsp;&nbsp;• "암 검진 주기와 검사 종류 총정리"<br><br>
        ❌ <b>피할 패턴</b><br>
        &nbsp;&nbsp;• "고혈압 방치하면 사망합니다"<br>
        &nbsp;&nbsp;• "암 증상 무시하면 절대 안 되는 이유"<br><br>
        시청자는 전문가에게 <b>정보·신뢰</b>를 기대<br>공포 자극은 권위 손상으로 이어짐
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown("""<div class="info-box">
        <h4 style="margin-top:0">📱 일반 채널 (General)</h4>
        <b>공포 키워드 효과 없음</b> — 중립적<br><br>
        📌 <b>더 중요한 것들</b><br>
        &nbsp;&nbsp;• <b>채널 규모</b>가 조회수를 결정 (구독자 증가 우선)<br>
        &nbsp;&nbsp;• <b>콘텐츠 발행 빈도</b>와 품질<br>
        &nbsp;&nbsp;• <b>썸네일 품질</b>: 빨간 배경·대비·엣지 밀도가 유의미<br>
        &nbsp;&nbsp;• <b>제목 길이</b>: 15~25자 범위 최적<br><br>
        📊 <b>카테고리 선택 전략</b><br>
        조회수 중앙값이 높은 카테고리(암·심장)에서<br>시작해 채널 성장 기반 확보
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 조회수 결정 요인 순위
    st.markdown("### 조회수를 실제로 결정하는 요인 (영향력 순위)")
    factors = [
        ("채널 구독자 수", 85, "채널 규모가 압도적 1위"),
        ("영상 발행 연도/나이", 60, "오래된 영상이 누적 조회수 많음 (착시 원인)"),
        ("썸네일 시각 피처", 35, "빨간색·대비·엣지 밀도 모두 유의 (p<0.05)"),
        ("질환 카테고리", 30, "암·심장 카테고리가 구조적으로 높음"),
        ("Shorts 여부", 20, "별도 알고리즘 — 일반 영상과 분리 분석 필요"),
        ("공포 키워드 (PSM)", -15, "채널 규모 통제 시 오히려 감소 (역효과)"),
    ]
    f_df = pd.DataFrame(factors, columns=["요인","영향력 점수","설명"])
    fig_f = go.Figure(go.Bar(
        x=f_df["영향력 점수"],
        y=f_df["요인"],
        orientation="h",
        marker_color=[C["red"] if v < 0 else C["blue"] for v in f_df["영향력 점수"]],
        text=f_df["설명"],
        textposition="outside",
    ))
    fig_f.add_vline(x=0, line_dash="dash", line_color="white")
    fig_f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis_title="상대적 영향력 (정성 추정)",
        margin=dict(l=160, r=20), height=350
    )
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")

    # ── 제목 분석기
    st.markdown("### 🔍 제목 분석기")
    st.markdown("제목을 입력하면 공포 점수와 채널 유형별 권장사항을 알려드립니다.")

    col_inp, col_type = st.columns([3, 1])
    with col_inp:
        title_input = st.text_input("제목 입력", placeholder="예: 당뇨 방치하면 실명·투석·사망까지 갑니다")
    with col_type:
        my_type = st.selectbox("내 채널 유형", ["Medical Pro", "General"])

    if title_input:
        found = [w for w in FEAR_WORDS if w in title_input]
        score = len(found)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("공포 점수", f"{score}점 / {len(FEAR_WORDS)}점")
        col_b.metric("감지 키워드", f"{len(found)}개")
        with col_c:
            if score == 0:
                st.markdown("""<div class="insight-box" style="padding:10px">
                ✅ <b>공포 키워드 없음</b><br>모든 채널 유형에 적합</div>""", unsafe_allow_html=True)
            elif score <= 1:
                st.markdown("""<div class="info-box" style="padding:10px">
                ⚠️ <b>낮은 공포 강도</b><br>Medical Pro는 주의 권장</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="warn-box" style="padding:10px">
                ❌ <b>높은 공포 강도</b><br>Medical Pro 채널 위험</div>""", unsafe_allow_html=True)

        if found:
            st.markdown(f"**감지된 공포 키워드**: {', '.join([f'`{w}`' for w in found])}")

        if my_type == "Medical Pro" and score >= 1:
            st.markdown("""<div class="warn-box">
            🏥 <b>Medical Pro 채널 경고</b><br>
            공포 키워드가 포함된 제목은 조회수 <b>0.83배 감소</b> 효과 (p=0.017, 유의)<br>
            전문성·신뢰 기반 제목으로 변경을 권장합니다.
            </div>""", unsafe_allow_html=True)
        elif my_type == "General" and score >= 1:
            st.markdown("""<div class="info-box">
            📱 <b>General 채널 참고</b><br>
            공포 키워드 효과는 통계적으로 유의하지 않습니다 (1.14배, p=0.558)<br>
            조회수 향상에 특별한 효과가 없으므로 채널 품질과 발행 빈도에 집중하세요.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="insight-box">
            ✅ 공포 키워드가 없는 좋은 제목입니다. Medical Pro 채널에 이상적입니다.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 공포 없이 고조회수 달성한 제목 TOP 20
    st.markdown("### 🏆 공포 키워드 없이 고조회수 달성한 제목 TOP 20")
    st.markdown("*벤치마킹 참고용 — 전문성·정보성 제목의 성공 사례*")
    top_nofear = df_f[df_f["Has_Fear"] == 0].nlargest(20, "Views")[
        ["Title", "Keyword", "Views", "Type", "Channel"]
    ].copy()
    top_nofear["조회수"] = top_nofear["Views"].apply(lambda x: f"{x/1e4:.1f}만")
    top_nofear = top_nofear[["Title", "Keyword", "조회수", "Type"]].reset_index(drop=True)
    top_nofear.index += 1
    st.dataframe(top_nofear, use_container_width=True)

    st.markdown("""<div class="insight-box">
    💡 <b>패턴 관찰</b>: 상위 영상들은 "방법", "원인", "증상", "총정리", "완벽 가이드" 등
    <b>정보 제공형 키워드</b>를 사용. 공포 없이도 높은 조회수 달성 가능.
    </div>""", unsafe_allow_html=True)

# ── 푸터
st.markdown("---")
st.markdown(
    "<center style='color:#666;font-size:.8em'>의료 유튜브 조회수 결정 요인 분석 · 데이터 기준일: 2026-04-19 · Python 3.13 · YouTube Data API v3</center>",
    unsafe_allow_html=True
)
