
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Pandas Styler 렌더링 제한 해제
pd.set_option("styler.render.max_elements", 2000000)

# ---------------------- Page config ----------------------
st.set_page_config(
    page_title="GGP Report System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* 상단 여백 확보 - 탭이 보이도록 */
      .block-container { padding-top: 2.5rem; padding-bottom: 2rem; }

      /* 탭 영역 상단 고정 */
      .stTabs { margin-top: 1rem; }

      /* Watermark - 30% 크기로 축소 */
      .khs-watermark {
        position: fixed;
        right: 24px;
        bottom: 18px;
        opacity: 0.10;
        font-size: 14px;
        font-weight: 800;
        z-index: 0;
        pointer-events: none;
        user-select: none;
      }
    </style>
    <div class="khs-watermark">Written by KHS</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Constants ----------------------
# 로컬/클라우드 모두 지원
if os.path.exists("D:/PGSv2/selection_list.dat"):
    DATA_DEFAULT_PATH = Path("D:/PGSv2") / "selection_list.dat"
else:
    DATA_DEFAULT_PATH = Path("selection_list.dat")  # repo 루트

DESIRED_COLUMN_ORDER = [
    "animal_renum", "ani_notch", "animal",
    "sire_notch", "sire", "dam_notch", "dam",
    "brd", "sex", "birth_date", "end_date", "test_week",
    "icoef", "test_age",
    "bf_bv", "D90_bv", "loin_bv", "nba_bv", "std_bv",
    "birth_wt", "end_wt", "adg", "a_bf", "loin", "age90",
    "Index", "type_A", "type_B", "type_C"
]

ROUND_3_COLS = ["bf_bv", "D90_bv", "loin_bv", "nba_bv",
                "birth_wt", "end_wt", "adg", "a_bf", "loin", "age90", "Index"]
ROUND_6_COLS = ["std_bv"]


# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def load_selection_list(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p.resolve()}\n\n"
            f"Expected default path: {DATA_DEFAULT_PATH.as_posix()}\n"
            "Put your selection_list.dat under ./data/ or set a custom path in the sidebar."
        )

    df = pd.read_table(p, sep="\t", header=0)

    # Parse date columns if present (날짜만, 시간 제거)
    for c in ["birth_date", "end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    # Reorder columns (keep any extra columns at the end)
    ordered = [c for c in DESIRED_COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras].copy()

    # Round numeric columns
    for c in ROUND_3_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    for c in ROUND_6_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(6)

    return df


def safe_metric_value(x):
    if x is None:
        return 0
    try:
        if np.isnan(x) or np.isinf(x):
            return 0
    except Exception:
        pass
    return x


def month_floor(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()


def empty_plot_note(text="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def format_number(x):
    """숫자에 천 단위 콤마 추가"""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float)):
        if x == int(x):
            return f"{int(x):,}"
        else:
            return f"{x:,.3f}"
    return x


def style_selection_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # 숫자 컬럼에 천 단위 콤마 포맷 적용
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    format_dict = {col: format_number for col in numeric_cols}

    styler = df.style.format(format_dict)

    # Highlight BV columns (dark-ish) + phenotypic columns (green-ish) + Index (red-ish)
    bv_cols = [c for c in ["bf_bv", "D90_bv", "loin_bv", "nba_bv", "std_bv"] if c in df.columns]
    ph_cols = [c for c in ["birth_wt", "end_wt", "adg", "a_bf", "loin", "age90"] if c in df.columns]
    idx_cols = [c for c in ["Index"] if c in df.columns]

    if bv_cols:
        styler = styler.set_properties(subset=bv_cols, **{"background-color": "#0b2e4a", "color": "white"})
    if ph_cols:
        styler = styler.set_properties(subset=ph_cols, **{"background-color": "#0b4a2e", "color": "white"})
    if idx_cols:
        styler = styler.set_properties(subset=idx_cols, **{"background-color": "#b30000", "color": "white", "font-weight": "700"})

    return styler


# ---------------------- Sidebar: data source + filters ----------------------
# 회사 로고 (로컬/클라우드 모두 지원)
logo_path = "cj.jpg"  # repo 루트에 있다고 가정
if os.path.exists("D:/PGSv2/cj.jpg"):
    logo_path = "D:/PGSv2/cj.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=120)
st.sidebar.title("GGP Report System")
st.sidebar.caption("Selection dashboard (Streamlit)")

data_path = st.sidebar.text_input("Data path", value=str(DATA_DEFAULT_PATH), help="Path to selection_list.dat (tab-separated).")

try:
    data = load_selection_list(data_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# Filters
st.sidebar.divider()
st.sidebar.subheader("Filters (Bộ lọc)")

gender = st.sidebar.selectbox("Gender (Giới tính)", options=["All", "F", "M"], index=0)

brd_values = ["All"]
if "brd" in data.columns:
    brd_values += sorted([x for x in data["brd"].dropna().unique().tolist()])
brd = st.sidebar.selectbox("Breed (Giống)", options=brd_values, index=0)

# Date range
if "end_date" in data.columns and data["end_date"].notna().any():
    min_d = min(data["end_date"].dropna())
    max_d = max(data["end_date"].dropna())
else:
    min_d = pd.Timestamp.today().date()
    max_d = pd.Timestamp.today().date()

date_range = st.sidebar.date_input(
    "Date Range (Khoảng thời gian)",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)

st.sidebar.markdown("<p style='color: rgba(150,150,150,0.5); font-size: 11px; text-align: center; margin-top: 20px;'>Written by KHS</p>", unsafe_allow_html=True)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
else:
    # Streamlit sometimes returns a single date if user clears one end
    d0, d1 = min_d, max_d

# Apply filters
df = data.copy()

if "end_date" in df.columns:
    df = df[(df["end_date"] >= d0) & (df["end_date"] <= d1)]

if gender != "All" and "sex" in df.columns:
    df = df[df["sex"] == gender]

if brd != "All" and "brd" in df.columns:
    df = df[df["brd"] == brd]


# ---------------------- Main UI: Tabs ----------------------
tab_overview, tab_selection, tab_analytics, tab_about = st.tabs(
    ["Overview (Tổng quan)", "Selection List (Danh sách)", "Analytics (Phân tích)", "About (Giới thiệu)"]
)

# ---------------------- Tab 1: Overview ----------------------
with tab_overview:
    st.markdown("## *CJ Feed&Care / GGP Selection Overview*")

    c1, c2, c3, c4 = st.columns(4)

    total_animals = int(len(df))
    avg_index = 0.0
    if "Index" in df.columns and df["Index"].notna().any():
        avg_index = float(df["Index"].mean(skipna=True))
    avg_index = safe_metric_value(avg_index)

    breeds_count = 0
    if "brd" in df.columns:
        breeds_count = int(df["brd"].dropna().nunique())

    recent_tests = 0
    if "end_date" in df.columns and df["end_date"].notna().any():
        max_end = df["end_date"].max()
        recent_tests = int(df[df["end_date"] >= (max_end - pd.Timedelta(days=30))].shape[0])

    c1.metric("Total Animals (Tổng số)", f"{total_animals:,}")
    c2.metric("Average Index (Chỉ số TB)", f"{avg_index:,.2f}")
    c3.metric("Breeds (Số giống)", f"{breeds_count:,}")
    c4.metric("Recent Tests (30 days)", f"{recent_tests:,}")

    st.divider()

    left, right = st.columns(2)

    # Breed distribution pie
    with left:
        st.subheader("Breed Distribution (Phân bố giống)")
        if len(df) == 0 or "brd" not in df.columns or df["brd"].dropna().empty:
            st.plotly_chart(empty_plot_note(), use_container_width=True)
        else:
            breed_counts = (
                df.dropna(subset=["brd"])
                  .groupby("brd", as_index=False)
                  .size()
                  .rename(columns={"size": "count"})
                  .sort_values("count", ascending=False)
            )
            fig = px.pie(breed_counts, names="brd", values="count", hole=0.0)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Index trend
    with right:
        st.subheader("Index Trend Over Time (Xu hướng chỉ số)")
        if len(df) == 0 or "end_date" not in df.columns or "Index" not in df.columns:
            st.plotly_chart(empty_plot_note(), use_container_width=True)
        else:
            trend = (
                df.dropna(subset=["end_date", "Index"])
                  .assign(month=month_floor(df["end_date"]))
                  .groupby("month", as_index=False)["Index"]
                  .mean()
                  .rename(columns={"Index": "avg_index"})
                  .sort_values("month")
            )
            if len(trend) == 0:
                st.plotly_chart(empty_plot_note(), use_container_width=True)
            else:
                fig = px.line(trend, x="month", y="avg_index", markers=True)
                fig.update_layout(
                    height=320,
                    xaxis_title="Month (Tháng)",
                    yaxis_title="Average Index (Chỉ số TB)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Top 10 Animals by Index (10 con giống tốt nhất)")
    if len(df) == 0 or "Index" not in df.columns:
        st.info("No data available")
    else:
        top_cols = [c for c in ["ani_notch", "animal", "brd", "sex", "end_date",
                                "bf_bv", "D90_bv", "loin_bv", "nba_bv", "std_bv",
                                "Index", "type_A", "type_B", "type_C"] if c in df.columns]
        top10 = (
            df.dropna(subset=["Index"])
              .sort_values("Index", ascending=False)
              .head(10)[top_cols]
        )
        if len(top10) == 0:
            st.info("No data available")
        else:
            st.dataframe(style_selection_df(top10), use_container_width=True, hide_index=True)


# ---------------------- Tab 2: Selection List ----------------------
with tab_selection:
    st.markdown("## *Selection List (Danh sách lựa chọn)*")

    st.caption("Apply filters in the sidebar. Download will export the filtered result.")
    if len(df) == 0:
        st.info("No data available for the current filters.")
    else:
        # 데이터가 너무 크면 스타일링 없이 표시 (성능 최적화)
        max_styled_cells = 500000
        if len(df) * len(df.columns) > max_styled_cells:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(style_selection_df(df), use_container_width=True, hide_index=True)

        # Download
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download Selection List (Tải danh sách)",
            data=csv_buf.getvalue(),
            file_name=f"Selection_List_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ---------------------- Tab 3: Analytics ----------------------
with tab_analytics:
    st.markdown("## *Trait Analytics (Phân tích tính trạng)*")

    if len(df) == 0:
        st.info("No data available for the current filters.")
    else:
        left, right = st.columns([1, 2])

        with left:
            data_type = st.radio(
                "Data Type (Loại dữ liệu)",
                options=["Phenotypic Data (Dữ liệu kiểu hình)", "Breeding Value Data (Dữ liệu giá trị giống)"],
                index=0,
            )

            if data_type.startswith("Phenotypic"):
                trait_options = [c for c in ["end_wt", "adg", "a_bf", "loin", "age90"] if c in df.columns]
                trait = st.selectbox("Trait (Tính trạng)", options=trait_options, index=0 if trait_options else None)
            else:
                trait_options = [c for c in ["bf_bv", "D90_bv", "loin_bv", "nba_bv", "std_bv"] if c in df.columns]
                trait = st.selectbox("Breeding Value (Giá trị giống)", options=trait_options, index=0 if trait_options else None)

        if trait is None or "end_date" not in df.columns:
            st.warning("Trait or end_date column missing in the dataset.")
        else:
            # Monthly average
            monthly = (
                df.dropna(subset=["end_date"])
                  .assign(month=month_floor(df["end_date"]))
                  .groupby("month", as_index=False)[trait]
                  .mean()
                  .rename(columns={trait: "mean_value"})
                  .sort_values("month")
            )
            monthly["mean_value"] = pd.to_numeric(monthly["mean_value"], errors="coerce").round(3)

            with right:
                st.subheader("Monthly Trend (Xu hướng theo tháng)")
                if len(monthly) == 0 or monthly["mean_value"].dropna().empty:
                    st.plotly_chart(empty_plot_note(), use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["mean_value"], name=trait))
                    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["mean_value"], mode="lines+markers", name="Trend"))
                    fig.update_layout(
                        height=420,
                        title=f"{trait} Monthly Average (Trung bình theo tháng)",
                        xaxis_title="Month (Tháng)",
                        yaxis_title="Average Value (Giá trị trung bình)",
                        margin=dict(l=10, r=10, t=60, b=10),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Monthly Statistics (Thống kê theo tháng)")
            monthly_display = monthly.copy()
            monthly_display["month"] = monthly_display["month"].dt.strftime("%Y-%m")
            st.dataframe(monthly_display.rename(columns={"month": "Month", "mean_value": "Mean"}), use_container_width=True, hide_index=True)


# ---------------------- Tab 4: About ----------------------
with tab_about:
    st.markdown("## *About GGP Report System*")
    st.markdown(
        """
        **GGP (Great-Grand Parent) Selection System**

        This app provides:
        - Breeding Value visualization (Phân tích giá trị giống)
        - Selection Index overview (Tính toán/hiển thị chỉ số lựa chọn)
        - Interactive filtering + downloads (Bộ lọc + tải xuống)
        - Monthly trend analytics (Phân tích xu hướng theo tháng)

        **Developer:** Written by KHS  
        © 2026 CJ Feed&Care - All Rights Reserved
        """
    )
