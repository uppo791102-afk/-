import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="생육 데이터 대시보드", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

def add_trendline(x, y, name):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None
    try:
        xv = df["x"].astype(float)
    except (ValueError, TypeError):
        xv = pd.Series(range(len(df)), index=df.index)
    m, b = np.polyfit(xv, df["y"], 1)
    fitted = m * xv + b
    return go.Scatter(x=df["x"], y=fitted, mode="lines", name=name, showlegend=False)

def show_summary(growth_df, env_df):
    dfs = []
    if growth_df is not None and len(growth_df.columns) > 0:
        dfs.append(("생육", growth_df))
    if env_df is not None and len(env_df.columns) > 0:
        dfs.append(("환경", env_df))
    if not dfs:
        st.info("요약할 데이터가 없어요.")
        return
    for label, df in dfs:
        numeric = [c for c in df.columns if c not in ("time", "개체번호") and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            st.info(f"'{label}' 데이터에서 숫자 열을 찾지 못했어요.")
            continue
        st.subheader(f"{label} 열 통계")
        rows = []
        for c in numeric:
            vals = df[c].dropna()
            if len(vals) == 0:
                continue
            rows.append({"열": c, "평균": vals.mean(), "최대": vals.max(), "최소": vals.min(), "개수": len(vals)})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        corr_cols = [c for c in numeric if df[c].notna().sum() >= 2]
        if len(corr_cols) >= 2:
            st.dataframe(df[corr_cols].corr(), width="stretch")

def growth_charts(growth_df):
    if growth_df is None or len(growth_df.columns) == 0:
        st.info("생육 데이터가 없어요.")
        return
    numeric = [c for c in growth_df.columns if c not in ("time", "개체번호") and pd.api.types.is_numeric_dtype(growth_df[c])]
    if not numeric:
        st.info("생육 데이터에서 숫자 열을 찾지 못했어요.")
        return
    n = len(numeric)
    cols = 2
    rows = (n + 1) // 2
    x_col = "time" if "time" in numeric else ("개체번호" if "개체번호" in numeric else numeric[0])
    if x_col not in ("time", "개체번호"):
        st.info("x축에 사용할 시간 또는 개체 열이 없어 숫자 열을 임의로 사용했어요.")
    titles = [f"{c} vs {x_col}" for c in numeric]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for i, c in enumerate(numeric):
        row = i // cols + 1
        col = i % cols + 1
        df = growth_df[[x_col, c]].dropna()
        if len(df) < 2:
            continue
        fig.add_trace(go.Scatter(x=df[x_col], y=df[c], mode="markers", name=c), row=row, col=col)
        trend = add_trendline(df[x_col], df[c], f"{c} 추세선")
        if trend is not None:
            fig.add_trace(trend, row=row, col=col)
    fig.update_layout(height=800, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"), title_text="생육 지표 추이")
    st.plotly_chart(fig, width="stretch")

def env_charts(env_df):
    if env_df is None or len(env_df.columns) == 0:
        st.info("환경 데이터가 없어요.")
        return
    numeric = [c for c in env_df.columns if c not in ("time", "개체번호") and pd.api.types.is_numeric_dtype(env_df[c])]
    if not numeric:
        st.info("환경 데이터에서 숫자 열을 찾지 못했어요.")
        return
    if "time" not in env_df.columns:
        st.info("환경 데이터에 시간 열이 없어요.")
        return
    n = len(numeric)
    cols = 2
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=numeric)
    for i, c in enumerate(numeric):
        row = i // cols + 1
        col = i % cols + 1
        df = env_df[["time", c]].dropna()
        if len(df) < 2:
            continue
        fig.add_trace(go.Scatter(x=df["time"], y=df[c], mode="lines+markers", name=c), row=row, col=col)
    fig.update_layout(height=800, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"), title_text="환경 지표 추이")
    st.plotly_chart(fig, width="stretch")

def main():
    st.title("식물 생육 데이터 대시보드")
    st.caption("왼쪽 사이드바에서 환경 CSV와 생육 CSV를 업로드해 주세요.")
    
    env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")
    
    env_df = load_file(env_file)
    growth_df = load_file(growth_file)
    
    env_df = std_cols(env_df) if env_df is not None else None
    growth_df = std_cols(growth_df) if growth_df is not None else None
    
    if env_df is None and growth_df is None:
        st.info("왼쪽 사이드바에서 환경 CSV와 생육 CSV를 업로드해 주세요.")
        return
    
    tab1, tab2, tab3 = st.tabs(["통계 요약", "생육 그래프", "환경 그래프"])
    
    with tab1:
        try:
            show_summary(growth_df, env_df)
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
    
    with tab2:
        try:
            growth_charts(growth_df)
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
    
    with tab3:
        try:
            env_charts(env_df)
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
    
    st.divider()
    for label, df in (("생육 CSV", growth_df), ("환경 CSV", env_df)):
        if df is not None and len(df.columns) > 0:
            data = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(f"{label} 다운로드", data, file_name=f"{label.split()[0]}.csv", mime="text/csv")

if __name__ == "__main__":
    main()
