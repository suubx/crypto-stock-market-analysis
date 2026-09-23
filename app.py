import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Crypto vs Stock Market Analysis",
    layout="wide"
)

# ============================================================
# VISUAL READABILITY / RESPONSIVE SIZING
# ============================================================

st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       Global sizing / overflow protection
       -------------------------------------------------------- */
    *, *::before, *::after {
        box-sizing: border-box;
    }

    html,
    body,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"] {
        max-width: 100%;
        overflow-x: hidden;
    }

    .block-container {
        max-width: 100%;
        padding-top: 1.6rem;
        padding-right: 2.1rem;
        padding-bottom: 2rem;
        padding-left: 2.1rem;
    }

    /* --------------------------------------------------------
       Main title and headings
       -------------------------------------------------------- */
    div[data-testid="stAppViewContainer"] h1 {
        font-size: clamp(36px, 2.3vw, 40px) !important;
        font-weight: 700 !important;
        line-height: 1.16 !important;
        margin-bottom: 0.4rem !important;
        white-space: nowrap;
        max-width: 100%;
    }

    div[data-testid="stAppViewContainer"] h2 {
        font-size: clamp(24px, 1.5vw, 27px) !important;
        font-weight: 700 !important;
        line-height: 1.24 !important;
        margin-top: 1.15rem !important;
        margin-bottom: 0.7rem !important;
        max-width: 100%;
    }

    div[data-testid="stAppViewContainer"] h3 {
        font-size: clamp(22px, 1.35vw, 25px) !important;
        font-weight: 680 !important;
        line-height: 1.25 !important;
        margin-top: 0.95rem !important;
        margin-bottom: 0.7rem !important;
        max-width: 100%;
    }

    /* --------------------------------------------------------
       General markdown/body text
       -------------------------------------------------------- */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: clamp(16px, 0.95vw, 18px) !important;
        line-height: 1.5 !important;
    }

    div[data-testid="stMarkdownContainer"] strong {
        font-weight: 700;
    }

    /* Captions: title caption + footer */
    div[data-testid="stCaptionContainer"] p {
        font-size: clamp(15px, 0.88vw, 16px) !important;
        line-height: 1.42 !important;
    }

    /* --------------------------------------------------------
       Sidebar
       -------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        max-width: 100%;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: clamp(20px, 1.25vw, 22px) !important;
        font-weight: 700 !important;
        line-height: 1.24 !important;
        margin-bottom: 0.6rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: clamp(16px, 0.92vw, 17px) !important;
        line-height: 1.35 !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] {
        min-height: 44px;
        max-width: 100%;
        min-width: 0;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        font-size: clamp(16px, 0.92vw, 17px) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        min-width: 0;
        max-width: 100%;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* --------------------------------------------------------
       File uploader
       -------------------------------------------------------- */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        max-width: 100%;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        font-size: clamp(15px, 0.86vw, 16px) !important;
        line-height: 1.35 !important;
    }

    /* --------------------------------------------------------
       Navigation tabs
       -------------------------------------------------------- */
    div[data-baseweb="tab-list"] {
        max-width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
    }

    button[data-baseweb="tab"] {
        padding: 0.58rem 0.9rem !important;
        min-height: 44px;
        white-space: nowrap;
        min-width: 0;
    }

    button[data-baseweb="tab"] p {
        font-size: clamp(16px, 0.95vw, 18px) !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        white-space: nowrap;
    }

    /* --------------------------------------------------------
       Buttons, sliders, inputs, selectboxes
       -------------------------------------------------------- */
    .stButton button,
    [data-testid="stFileUploaderDropzone"] button {
        font-size: clamp(15px, 0.9vw, 17px) !important;
        min-height: 42px;
        padding: 0.48rem 0.88rem;
    }

    [data-testid="stSlider"] p,
    [data-testid="stSelectbox"] p,
    [data-testid="stFileUploader"] p {
        font-size: clamp(16px, 0.92vw, 17px) !important;
        line-height: 1.35 !important;
    }

    [data-testid="stSlider"] [role="slider"],
    [data-testid="stSlider"] [data-baseweb="slider"] * {
        font-size: clamp(15px, 0.88vw, 17px) !important;
    }

    /* --------------------------------------------------------
       Alerts / recommendation / information boxes
       -------------------------------------------------------- */
    div[data-testid="stAlert"] {
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] li {
        font-size: clamp(15px, 0.9vw, 16px) !important;
        line-height: 1.48 !important;
    }

    /* --------------------------------------------------------
       Expanders
       -------------------------------------------------------- */
    details[data-testid="stExpander"] summary p {
        font-size: clamp(16px, 0.92vw, 17px) !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
    }

    details[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p,
    details[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li {
        font-size: clamp(16px, 0.92vw, 17px) !important;
        line-height: 1.5 !important;
    }

    /* --------------------------------------------------------
       Custom research metric cards
       -------------------------------------------------------- */
    .research-metric-card {
        width: 100%;
        min-width: 0;
        max-width: 100%;
        min-height: 108px;
        padding: 18px 14px;
        border: 1px solid #ddd;
        border-radius: 11px;
        background-color: #f9f9f9;
        text-align: center;
        margin-bottom: 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }

    .research-metric-label {
        width: 100%;
        min-width: 0;
        color: #555;
        font-size: clamp(16px, 0.96vw, 18px);
        font-weight: 550;
        line-height: 1.22;
        margin-bottom: 7px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .research-metric-value {
        width: 100%;
        min-width: 0;
        color: #111;
        font-size: clamp(30px, 1.65vw, 34px);
        font-weight: 700;
        line-height: 1.12;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-variant-numeric: tabular-nums;
    }

    div[data-testid="stVerticalBlock"] > div:has(.research-metric-card) {
        min-width: 0;
        max-width: 100%;
    }

    div[data-testid="stHorizontalBlock"] {
        max-width: 100%;
    }

    div[data-testid="stHorizontalBlock"] > div {
        min-width: 0;
    }

    /* --------------------------------------------------------
       Dataframes / tables
       Keep dataframe's own horizontal scrolling for wide tables.
       -------------------------------------------------------- */
    div[data-testid="stDataFrame"] {
        max-width: 100% !important;
        min-width: 0 !important;
        font-size: clamp(14px, 0.82vw, 16px) !important;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="rowheader"],
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td {
        font-size: clamp(14px, 0.82vw, 16px) !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] canvas {
        max-width: 100% !important;
    }

    /* --------------------------------------------------------
       Classification report / code / plain text
       -------------------------------------------------------- */
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code,
    pre code {
        font-size: clamp(14px, 0.82vw, 16px) !important;
        line-height: 1.45 !important;
    }

    div[data-testid="stText"] {
        font-size: clamp(16px, 0.92vw, 17px) !important;
        line-height: 1.45 !important;
    }

    /* --------------------------------------------------------
       Responsive behavior
       -------------------------------------------------------- */
    @media (max-width: 1200px) {
        .block-container {
            padding-right: 1.5rem;
            padding-left: 1.5rem;
        }

        div[data-testid="stAppViewContainer"] h1 {
            font-size: 34px !important;
            white-space: normal;
        }

        .research-metric-value {
            font-size: 29px;
        }

        .research-metric-label {
            font-size: 15.5px;
        }

        button[data-baseweb="tab"] {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }

        button[data-baseweb="tab"] p {
            font-size: 15.5px !important;
        }
    }

    @media (max-width: 900px) {
        .block-container {
            padding-right: 1rem;
            padding-left: 1rem;
        }

        div[data-testid="stAppViewContainer"] h1 {
            font-size: 30px !important;
            white-space: normal;
        }

        .research-metric-value {
            font-size: 26px;
        }

        .research-metric-label {
            font-size: 14.5px;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 15px !important;
        }

        button[data-baseweb="tab"] p {
            font-size: 14.5px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Crypto vs. Stock Markets: Trends, Volatility & Risk Analysis")
st.caption("Interactive financial analytics and machine-learning dashboard for Bitcoin, Ethereum, NASDAQ, and S&P 500.")


# ============================================================
# FILE SETTINGS
# ============================================================

DEFAULT_FILES = {
    "Daily Analysis": "FINAL_Daily_Analysis.csv",
    "4H Analysis": "FINAL_4H_Aligned.csv",
    "1H Crypto Analysis": "FINAL_1H_Crypto_Aligned.csv",
}

ASSET_CLOSES = {
    "Bitcoin": "btc_close",
    "Ethereum": "eth_close",
    "NASDAQ": "nasdaq_close",
    "S&P 500": "sp500_close",
}

RETURN_COLS = {
    "Bitcoin": "btc_return",
    "Ethereum": "eth_return",
    "NASDAQ": "nasdaq_return",
}

VOL_COLS = {
    "Bitcoin": "btc_volatility_30d",
    "Ethereum": "eth_volatility_30d",
    "NASDAQ": "nasdaq_volatility_30d",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_display_text(value):
    """Convert values safely for display."""
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).strftime("%Y-%m-%d")

    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return str(value)

    if isinstance(value, (int, np.integer)):
        return f"{value:,}"

    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"

    return str(value)


def metric_box(label, value):
    """Custom metric card. This avoids Streamlit datetime.date metric errors."""
    st.markdown(
        f"""
        <div class="research-metric-card">
            <div class="research-metric-label">{label}</div>
            <div class="research-metric-value">{to_display_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def load_csv(uploaded_file, fallback_name):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = fallback_name

        if not os.path.exists(path):
            path = os.path.join(os.getcwd(), fallback_name)

        df = pd.read_csv(path)

    df.columns = [str(col).strip() for col in df.columns]

    first_col = df.columns[0]
    lower_cols = {col.lower(): col for col in df.columns}

    if first_col.startswith("Unnamed") or first_col == "":
        df = df.rename(columns={first_col: "datetime"})
    elif first_col.lower() in ["date", "datetime", "time", "timestamp"]:
        df = df.rename(columns={first_col: "datetime"})
    elif "datetime" in lower_cols:
        df = df.rename(columns={lower_cols["datetime"]: "datetime"})
    elif "date" in lower_cols:
        df = df.rename(columns={lower_cols["date"]: "datetime"})
    elif "time" in lower_cols:
        df = df.rename(columns={lower_cols["time"]: "datetime"})
    elif "timestamp" in lower_cols:
        df = df.rename(columns={lower_cols["timestamp"]: "datetime"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime")
        df = df.reset_index(drop=True)

    return df


def missing_summary(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    available_cells = total_cells - missing_cells
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    available_percentage = 100 - missing_percentage

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "available_cells": available_cells,
        "missing_percentage": missing_percentage,
        "available_percentage": available_percentage,
    }


def handle_missing_values(df, method):
    """
    Handles missing values using selected method.
    Raw data remains unchanged.
    Cleaned dataset is used for graphs and model.
    """
    cleaned = df.copy()

    if method == "No handling":
        return cleaned

    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = [
        col for col in cleaned.columns
        if col not in numeric_cols and col != "datetime"
    ]

    if method == "Forward fill only":
        if len(numeric_cols) > 0:
            cleaned[numeric_cols] = cleaned[numeric_cols].ffill()

        if len(non_numeric_cols) > 0:
            cleaned[non_numeric_cols] = cleaned[non_numeric_cols].ffill()

    elif method == "Forward fill + backward fill":
        if len(numeric_cols) > 0:
            cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()

        if len(non_numeric_cols) > 0:
            cleaned[non_numeric_cols] = cleaned[non_numeric_cols].ffill().bfill()

    elif method == "Interpolate numeric + fill remaining":
        if len(numeric_cols) > 0:
            cleaned[numeric_cols] = cleaned[numeric_cols].interpolate(
                method="linear",
                limit_direction="both"
            )
            cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()

        if len(non_numeric_cols) > 0:
            cleaned[non_numeric_cols] = cleaned[non_numeric_cols].ffill().bfill()

    elif method == "Drop rows with missing values":
        cleaned = cleaned.dropna().reset_index(drop=True)

    return cleaned


def quality_table(df):
    ms = missing_summary(df)

    if "datetime" in df.columns:
        duplicate_timestamps = int(df["datetime"].duplicated().sum())
        start_date = to_display_text(df["datetime"].min())
        end_date = to_display_text(df["datetime"].max())
    else:
        duplicate_timestamps = "No datetime column"
        start_date = "N/A"
        end_date = "N/A"

    return pd.DataFrame({
        "Check": [
            "Rows",
            "Columns",
            "Total cells",
            "Available cells",
            "Missing cells",
            "Missing percentage",
            "Available percentage",
            "Duplicate timestamps",
            "Date range start",
            "Date range end",
        ],
        "Result": [
            to_display_text(len(df)),
            to_display_text(len(df.columns)),
            to_display_text(ms["total_cells"]),
            to_display_text(ms["available_cells"]),
            to_display_text(ms["missing_cells"]),
            f"{ms['missing_percentage']:.2f}%",
            f"{ms['available_percentage']:.2f}%",
            to_display_text(duplicate_timestamps),
            start_date,
            end_date,
        ],
    })


def find_available_cols(df, col_dict):
    available = {}

    for display_name, column_name in col_dict.items():
        if column_name in df.columns and df[column_name].notna().sum() > 0:
            available[display_name] = column_name

    return available


def normalize_for_plot(df, cols):
    temp = df[["datetime"] + list(cols.values())].copy()

    for col in cols.values():
        valid_values = temp[col].dropna()

        if len(valid_values) > 0:
            first_valid = valid_values.iloc[0]

            if pd.notna(first_valid) and first_valid != 0:
                temp[col] = temp[col] / first_valid * 100

    rename_map = {v: k for k, v in cols.items()}
    temp = temp.rename(columns=rename_map)

    return temp


def explain_box(title, what, finding, why, rationale):
    with st.expander(f"Detailed explanation: {title}", expanded=False):
        st.markdown(f"**What the graph shows:** {what}")
        st.markdown(f"**Finding:** {finding}")
        st.markdown(f"**Why this happened:** {why}")
        st.markdown(f"**Rationale:** {rationale}")


def safe_plotly_chart(fig):
    """Render Plotly charts with larger, presentation-friendly text."""
    fig.update_layout(
        title_font=dict(size=22),
        font=dict(size=16),
        legend=dict(font=dict(size=16)),
        hoverlabel=dict(font_size=15),
    )
    fig.update_xaxes(
        title_font=dict(size=17),
        tickfont=dict(size=15),
    )
    fig.update_yaxes(
        title_font=dict(size=17),
        tickfont=dict(size=15),
    )

    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def safe_dataframe(data):
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Dataset Input")

    selected = st.selectbox(
        "Choose analysis dataset",
        list(DEFAULT_FILES.keys())
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    st.header("Missing Value Handling")

    missing_method = st.selectbox(
        "Choose preprocessing method",
        [
            "Interpolate numeric + fill remaining",
            "Forward fill + backward fill",
            "Forward fill only",
            "Drop rows with missing values",
            "No handling",
        ]
    )

    st.info(
        "Recommended: Interpolate numeric + fill remaining. "
        "It is suitable for time-series financial data and keeps dataset size stable."
    )


# ============================================================
# LOAD RAW DATA
# ============================================================

try:
    raw_df = load_csv(uploaded, DEFAULT_FILES[selected])
except Exception as e:
    st.error(
        f"Could not load dataset. Put {DEFAULT_FILES[selected]} in the same folder as app.py "
        f"or upload it manually. Error: {e}"
    )
    st.stop()


# Cleaned dataset used for charts and model
df = handle_missing_values(raw_df, missing_method)


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Input",
    "2. Preprocess",
    "3. Model & Analysis",
    "4. Output & Analysis",
])


# ============================================================
# TAB 1: INPUT
# ============================================================

with tab1:
    st.subheader("Input: Dataset Preview")

    raw_missing = missing_summary(raw_df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_box("Rows", len(raw_df))

    with c2:
        metric_box("Columns", len(raw_df.columns))

    with c3:
        if "datetime" in raw_df.columns:
            metric_box("Start Date", raw_df["datetime"].min())
        else:
            metric_box("Start Date", "N/A")

    with c4:
        if "datetime" in raw_df.columns:
            metric_box("End Date", raw_df["datetime"].max())
        else:
            metric_box("End Date", "N/A")

    st.markdown("### Dataset Completeness")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_box("Total Cells", raw_missing["total_cells"])

    with c6:
        metric_box("Available Cells", raw_missing["available_cells"])

    with c7:
        metric_box("Missing Cells", raw_missing["missing_cells"])

    with c8:
        metric_box("Missing Percentage", f"{raw_missing['missing_percentage']:.2f}%")

    st.markdown("### Raw Dataset Preview")
    safe_dataframe(raw_df.head(50))

    st.markdown("""
    **Dataset rationale:**  
    The project uses three aligned datasets. The daily dataset is the main dataset for long-term crypto-stock comparison.
    The 4-hour dataset supports trend-structure analysis. The 1-hour crypto dataset supports volatility spike and event-timing analysis.
    """)


# ============================================================
# TAB 2: PREPROCESS
# ============================================================

with tab2:
    st.subheader("Preprocess: Missing Values, Data Quality, and Feature Engineering")

    st.markdown("## Before Missing Value Handling")
    safe_dataframe(quality_table(raw_df))

    st.markdown("## After Missing Value Handling")
    safe_dataframe(quality_table(df))

    before_missing = missing_summary(raw_df)
    after_missing = missing_summary(df)

    st.markdown("## Missing Value Handling Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_box("Missing Before", before_missing["missing_cells"])

    with c2:
        metric_box("Missing % Before", f"{before_missing['missing_percentage']:.2f}%")

    with c3:
        metric_box("Missing After", after_missing["missing_cells"])

    with c4:
        metric_box("Missing % After", f"{after_missing['missing_percentage']:.2f}%")

    st.markdown(f"""
    **Selected missing value handling method:** `{missing_method}`

    **Rationale:**  
    Missing values are expected in this project because the dataset contains rolling-window financial indicators and multiple market calendars.
    Rolling features such as 30-day volatility, 30-day return, and rolling correlation naturally create missing values at the beginning of the dataset.
    Also, cryptocurrency markets trade 24/7, while stock markets close on weekends and holidays.

    The selected preprocessing method creates a cleaned dataset for graphing and model training while keeping the raw dataset visible for transparency.
    This demonstrates that the project identifies missing values, reports their percentage, and handles them before analysis.
    """)

    st.markdown("""
    **Other preprocessing steps demonstrated in this project:**

    1. Converted date/time columns into a standard datetime format.
    2. Sorted observations chronologically.
    3. Checked duplicate timestamps.
    4. Handled missing values using the selected method.
    5. Resampled ETH 1-hour data into daily ETH data for the daily master dataset.
    6. Merged datasets using a common time index.
    7. Created engineered features such as returns, 30-day returns, rolling volatility, drawdown, crash/boom flags, convexity, liquidity-cost proxy, rolling correlation, and risk ratio.

    **Outlier decision:**  
    Extreme values were not removed because they represent real financial events such as crashes and booms.
    Removing them would reduce the validity of the volatility and risk analysis.
    """)

    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("Missing Values by Numeric Column Before Handling")

        missing_before_df = (
            raw_df[numeric_cols]
            .isna()
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        missing_before_df.columns = ["Column", "Missing Values Before"]

        safe_dataframe(missing_before_df.head(30))

    cleaned_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if cleaned_numeric_cols:
        st.subheader("Missing Values by Numeric Column After Handling")

        missing_after_df = (
            df[cleaned_numeric_cols]
            .isna()
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        missing_after_df.columns = ["Column", "Missing Values After"]

        safe_dataframe(missing_after_df.head(30))

    st.subheader("Cleaned Dataset Preview")
    safe_dataframe(df.head(50))


# ============================================================
# TAB 3: PROCESS / MODEL
# ============================================================

with tab3:
    st.subheader("Process: Market-State Classification Model")

    st.markdown("""
    The model classifies **historical market conditions** into three classes using engineered financial indicators:

    - **Class 0 = Normal**
    - **Class 1 = Boom**
    - **Class 2 = Crash**
    """)

    st.info(
        "This implementation performs historical market-state classification. "
        "It does not claim to forecast a crash or boom a fixed number of minutes, "
        "hours, or days into the future."
    )

    st.caption(
        "For academic demonstration only. Model outputs are based on historical "
        "data and are not financial or investment advice."
    )

    if selected != "Daily Analysis":
        st.warning(
            "The market-state model is designed for the Daily Analysis dataset "
            "because it contains the complete engineered crypto-stock feature set."
        )

    required_target_cols = ["btc_boom_flag", "btc_crash_flag"]

    if not all(col in df.columns for col in required_target_cols):
        st.info(
            "The selected dataset does not include the required BTC boom and crash "
            "labels. Please switch to the Daily Analysis dataset."
        )
    else:
        model_choice = st.selectbox(
            "Choose model",
            ["Random Forest", "Logistic Regression"]
        )

        model_df = df.copy()

        # ------------------------------------------------------------
        # Multiclass target:
        # 0 = Normal, 1 = Boom, 2 = Crash
        # Crash receives priority if both flags are ever present.
        # ------------------------------------------------------------
        model_df["market_state"] = 0
        model_df.loc[model_df["btc_boom_flag"] == 1, "market_state"] = 1
        model_df.loc[model_df["btc_crash_flag"] == 1, "market_state"] = 2

        class_labels = {
            0: "Normal",
            1: "Boom",
            2: "Crash",
        }

        # Paper-reproduction feature set. Target flag columns are excluded,
        # while btc_return is retained because it is part of the accepted paper's
        # reported experimental feature set.
        candidate_features = [
            "btc_return",
            "btc_return_30d",
            "btc_volatility_30d",
            "btc_drawdown",
            "btc_convexity",
            "btc_liquidity_cost",

            "eth_return",
            "eth_return_30d",
            "eth_volatility_30d",
            "eth_drawdown",
            "eth_convexity",
            "eth_liquidity_cost",

            "nasdaq_return",
            "nasdaq_return_30d",
            "nasdaq_volatility_30d",
            "nasdaq_drawdown",
            "nasdaq_convexity",
            "nasdaq_liquidity_cost",

            "corr_btc_nasdaq_30d",
            "corr_eth_nasdaq_30d",
            "risk_ratio_btc_vs_nasdaq",
        ]

        features = [
            col for col in candidate_features
            if col in model_df.columns
        ]

        if len(features) == 0:
            st.error("No usable model features were found in this dataset.")
        else:
            model_df = model_df.dropna(
                subset=features + ["market_state"]
            ).copy()

            if "datetime" in model_df.columns:
                model_df = model_df.sort_values("datetime").reset_index(drop=True)
            else:
                model_df = model_df.reset_index(drop=True)

            class_counts = model_df["market_state"].value_counts().to_dict()

            if len(model_df) < 50:
                st.error("Not enough valid rows for reliable model training.")
            else:
                split_index = int(len(model_df) * 0.75)

                X = model_df[features]
                y = model_df["market_state"]

                X_train = X.iloc[:split_index]
                X_test = X.iloc[split_index:]

                y_train = y.iloc[:split_index]
                y_test = y.iloc[split_index:]

                train_classes = set(int(v) for v in y_train.unique())

                st.markdown("### Model Dataset Summary")

                s1, s2, s3, s4 = st.columns(4)

                with s1:
                    metric_box("Usable Rows", len(model_df))
                with s2:
                    metric_box("Training Rows", len(X_train))
                with s3:
                    metric_box("Testing Rows", len(X_test))
                with s4:
                    metric_box("Features Used", len(features))

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_box("Normal Observations", class_counts.get(0, 0))
                with c2:
                    metric_box("Boom Observations", class_counts.get(1, 0))
                with c3:
                    metric_box("Crash Observations", class_counts.get(2, 0))

                if train_classes != {0, 1, 2}:
                    st.error(
                        "The training period does not contain all three market-state "
                        "classes. The model cannot be evaluated reliably with the "
                        "current dataset."
                    )
                elif len(X_test) == 0:
                    st.error("No testing observations are available after the chronological split.")
                else:
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                            max_depth=6,
                        )
                    else:
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(
                                    max_iter=1000,
                                    class_weight="balanced"
                                )
                            ),
                        ])

                    try:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    except Exception as e:
                        st.error(f"Model training or evaluation failed: {e}")
                    else:
                        accuracy = accuracy_score(y_test, predictions)
                        weighted_precision = precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                        weighted_recall = recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                        weighted_f1 = f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )

                        m1, m2, m3, m4 = st.columns(4)

                        with m1:
                            metric_box("Accuracy", accuracy)

                        with m2:
                            metric_box("Weighted Precision", weighted_precision)

                        with m3:
                            metric_box("Weighted Recall", weighted_recall)

                        with m4:
                            metric_box("Weighted F1", weighted_f1)

                        actual_crash_mask = (y_test.to_numpy() == 2)

                        if actual_crash_mask.sum() > 0:
                            crash_recall = (
                                predictions[actual_crash_mask] == 2
                            ).sum() / actual_crash_mask.sum()
                        else:
                            crash_recall = 0.0
                            st.warning(
                                "The testing period contains no actual Crash observations, "
                                "so Crash Recall cannot be meaningfully estimated."
                            )

                        st.markdown("### Crash-Class Recall")

                        crash_col, crash_text_col = st.columns([1, 3])

                        with crash_col:
                            metric_box("Crash Recall", crash_recall)

                        with crash_text_col:
                            st.markdown(
                                "**Crash recall** indicates how many actual crash-state "
                                "observations were successfully identified by the model."
                            )

                        cm = confusion_matrix(
                            y_test,
                            predictions,
                            labels=[0, 1, 2]
                        )

                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(
                                x="Predicted",
                                y="Actual",
                                color="Count"
                            ),
                            x=["Normal", "Boom", "Crash"],
                            y=["Normal", "Boom", "Crash"],
                            title="Market-State Confusion Matrix",
                        )

                        safe_plotly_chart(fig_cm)

                        st.markdown("### Classification Report")

                        st.code(
                            classification_report(
                                y_test,
                                predictions,
                                labels=[0, 1, 2],
                                target_names=["Normal", "Boom", "Crash"],
                                zero_division=0
                            )
                        )

                        if model_choice == "Random Forest":
                            importance_df = pd.DataFrame({
                                "Feature": features,
                                "Importance": model.feature_importances_,
                            }).sort_values(
                                by="Importance",
                                ascending=False
                            )

                            fig_importance = px.bar(
                                importance_df.head(12),
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                title="Top Feature Importance for Market-State Classification",
                            )

                            fig_importance.update_layout(
                                yaxis=dict(autorange="reversed")
                            )

                            safe_plotly_chart(fig_importance)

                        st.markdown("""
                        **Model rationale:**  
                        Random Forest is used because crypto-stock market relationships can be nonlinear and regime-sensitive.
                        Logistic Regression provides a simpler and more interpretable baseline.
                        The target variable represents three historical market states—Normal, Boom, and Crash—derived from the existing BTC boom and crash indicators.
                        A chronological 75/25 split is used to preserve the temporal structure of the financial data.

                        The model should be interpreted as a **historical market-state classifier** rather than a fixed-horizon forecasting system.
                        """)

                        # ------------------------------------------------------------
                        # Interactive scenario classification
                        # ------------------------------------------------------------
                        st.markdown("### Interactive Market-State Prediction")

                        st.info(
                            "Choose an observation from the testing period and optionally "
                            "adjust a small set of indicators. The output is an interactive "
                            "scenario classification, not a future forecast."
                        )

                        test_rows = model_df.iloc[split_index:].copy().reset_index(drop=True)

                        if len(test_rows) == 0:
                            st.warning("No testing observations are available for interactive classification.")
                        else:
                            if "datetime" in test_rows.columns:
                                date_options = [
                                    f"{to_display_text(row['datetime'])} | row {idx + 1}"
                                    for idx, row in test_rows.iterrows()
                                ]
                            else:
                                date_options = [
                                    f"Testing observation {idx + 1}"
                                    for idx in range(len(test_rows))
                                ]

                            selected_option = st.selectbox(
                                "Choose a historical testing observation",
                                date_options
                            )

                            selected_pos = date_options.index(selected_option)
                            selected_row = test_rows.iloc[selected_pos]

                            if "datetime" in test_rows.columns:
                                st.write(
                                    f"Selected date: **{to_display_text(selected_row['datetime'])}**"
                                )

                            adjustable_features = [
                                "btc_return",
                                "eth_return",
                                "nasdaq_return",
                                "btc_volatility_30d",
                                "btc_drawdown",
                                "eth_volatility_30d",
                                "nasdaq_volatility_30d",
                                "corr_btc_nasdaq_30d",
                                "risk_ratio_btc_vs_nasdaq",
                            ]

                            adjustable_features = [
                                col for col in adjustable_features
                                if col in features
                                and pd.api.types.is_numeric_dtype(model_df[col])
                                and model_df[col].notna().any()
                            ]

                            user_input = {}
                            slider_columns = st.columns(2)

                            for i, feature in enumerate(adjustable_features):
                                feature_series = model_df[feature].dropna()

                                feature_min = float(feature_series.min())
                                feature_max = float(feature_series.max())
                                current_value = float(selected_row[feature])

                                if feature_min == feature_max:
                                    feature_min = current_value - 1.0
                                    feature_max = current_value + 1.0

                                step = (feature_max - feature_min) / 100.0
                                if step <= 0 or not np.isfinite(step):
                                    step = 0.01

                                with slider_columns[i % 2]:
                                    user_input[feature] = st.slider(
                                        feature,
                                        min_value=feature_min,
                                        max_value=feature_max,
                                        value=min(max(current_value, feature_min), feature_max),
                                        step=step,
                                        format="%.5f"
                                    )

                            final_input = {}

                            for feature in features:
                                if feature in user_input:
                                    final_input[feature] = user_input[feature]
                                else:
                                    value = selected_row[feature]

                                    if pd.isna(value):
                                        value = model_df[feature].median()

                                    final_input[feature] = float(value)

                            input_df = pd.DataFrame(
                                [[final_input[feature] for feature in features]],
                                columns=features
                            )

                            if st.button("Predict Market Condition"):
                                try:
                                    predicted_class = int(model.predict(input_df)[0])
                                    predicted_label = class_labels.get(
                                        predicted_class,
                                        "Unknown"
                                    )
                                except Exception as e:
                                    st.error(f"Interactive classification failed: {e}")
                                else:
                                    st.markdown("#### Scenario Classification Result")

                                    result_col1, result_col2 = st.columns(2)

                                    with result_col1:
                                        metric_box(
                                            "Predicted Class",
                                            predicted_class
                                        )

                                    with result_col2:
                                        metric_box(
                                            "Market State",
                                            predicted_label
                                        )

                                    if predicted_class == 2:
                                        st.error(
                                            "Scenario classification: CRASH-like historical market state."
                                        )
                                    elif predicted_class == 1:
                                        st.warning(
                                            "Scenario classification: BOOM-like historical market state."
                                        )
                                    else:
                                        st.success(
                                            "Scenario classification: NORMAL historical market state."
                                        )

                                    if hasattr(model, "predict_proba"):
                                        try:
                                            probability_values = model.predict_proba(input_df)[0]
                                            model_classes = getattr(
                                                model,
                                                "classes_",
                                                np.array([0, 1, 2])
                                            )

                                            probability_map = {
                                                int(class_id): float(probability)
                                                for class_id, probability
                                                in zip(model_classes, probability_values)
                                            }

                                            probability_df = pd.DataFrame({
                                                "Market State": ["Normal", "Boom", "Crash"],
                                                "Probability": [
                                                    probability_map.get(0, 0.0),
                                                    probability_map.get(1, 0.0),
                                                    probability_map.get(2, 0.0),
                                                ],
                                            })

                                            fig_probability = px.bar(
                                                probability_df,
                                                x="Market State",
                                                y="Probability",
                                                text_auto=".3f",
                                                title="Scenario Classification Probabilities",
                                            )

                                            fig_probability.update_yaxes(
                                                range=[0, 1]
                                            )

                                            safe_plotly_chart(fig_probability)
                                        except Exception as e:
                                            st.warning(
                                                f"Class probabilities could not be displayed: {e}"
                                            )

                                    st.caption(
                                        "Interactive scenario classification only. "
                                        "This output is not financial advice and does not predict "
                                        "a fixed future time horizon."
                                    )


# ============================================================
# TAB 4: OUTPUT & ANALYSIS
# ============================================================

with tab4:
    st.subheader("Output: Charts, Findings, and Rationale")

    if "datetime" not in df.columns:
        st.error("No datetime column found, so charts cannot be produced.")
    else:
        close_cols = find_available_cols(df, ASSET_CLOSES)
        return_cols = find_available_cols(df, RETURN_COLS)
        volatility_cols = find_available_cols(df, VOL_COLS)

        # ============================================================
        # FEATURE 1: PRICE TREND ANALYSIS
        # ============================================================

        st.markdown("## Feature 1: Price Trend Analysis")

        if close_cols:
            normalized_df = normalize_for_plot(df, close_cols)

            melted_df = normalized_df.melt(
                id_vars="datetime",
                var_name="Asset",
                value_name="Indexed Price"
            )

            fig_price = px.line(
                melted_df,
                x="datetime",
                y="Indexed Price",
                color="Asset",
                title="Indexed Price Trend Comparison (First Valid Price = 100)",
            )

            safe_plotly_chart(fig_price)

            explain_box(
                "Price Trend Analysis",
                "It compares the price movement of available assets after normalizing their first valid price to 100.",
                "Crypto assets usually show sharper upward and downward movements than stock-market indices.",
                "Crypto markets are strongly influenced by speculative demand, liquidity changes, investor sentiment, and sudden market shocks.",
                "The graph supports the idea that crypto behaves as a higher-risk and more unstable asset class than traditional stock indices.",
            )
        else:
            st.warning("Price columns were not found.")

        # ============================================================
        # FEATURE 2: DAILY AND PERIODIC RETURNS
        # ============================================================

        st.markdown("## Feature 2: Daily and Periodic Returns")

        if return_cols:
            returns_df = df[
                ["datetime"] + list(return_cols.values())
            ].rename(
                columns={v: k for k, v in return_cols.items()}
            )

            melted_returns = returns_df.melt(
                id_vars="datetime",
                var_name="Asset",
                value_name="Return"
            )

            fig_returns = px.line(
                melted_returns,
                x="datetime",
                y="Return",
                color="Asset",
                title="Return Comparison",
            )

            safe_plotly_chart(fig_returns)

            explain_box(
                "Daily and Periodic Returns",
                "It shows percentage changes in prices over time.",
                "BTC and ETH returns fluctuate more sharply than NASDAQ returns.",
                "Crypto trades continuously and reacts quickly to market news, investor emotion, exchange activity, and speculative pressure.",
                "Higher return fluctuation means greater short-term risk and less predictable price behavior.",
            )
        else:
            st.warning("Return columns were not found.")

        # ============================================================
        # FEATURE 3: VOLATILITY MEASUREMENT
        # ============================================================

        st.markdown("## Feature 3: Volatility Measurement")

        if volatility_cols:
            volatility_df = df[
                ["datetime"] + list(volatility_cols.values())
            ].rename(
                columns={v: k for k, v in volatility_cols.items()}
            )

            melted_volatility = volatility_df.melt(
                id_vars="datetime",
                var_name="Asset",
                value_name="30D Volatility"
            )

            fig_volatility = px.line(
                melted_volatility,
                x="datetime",
                y="30D Volatility",
                color="Asset",
                title="30-Day Rolling Volatility",
            )

            safe_plotly_chart(fig_volatility)

            explain_box(
                "Volatility Measurement",
                "It shows how unstable returns are within a rolling 30-day window.",
                "Crypto volatility is generally higher than stock-market volatility.",
                "Crypto markets are less mature, more sentiment-driven, and exposed to sudden liquidation and liquidity events.",
                "This confirms that crypto carries higher risk and needs stronger risk-monitoring than traditional stock indices.",
            )
        else:
            st.warning("Volatility columns were not found.")

        # ============================================================
        # FEATURE 4: CRASH PERIOD DETECTION
        # ============================================================

        st.markdown("## Feature 4: Crash Period Detection")

        if "btc_close" in df.columns and "btc_crash_flag" in df.columns:
            fig_crash = go.Figure()

            fig_crash.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["btc_close"],
                    mode="lines",
                    name="BTC Close"
                )
            )

            crash_df = df[df["btc_crash_flag"] == 1]

            fig_crash.add_trace(
                go.Scatter(
                    x=crash_df["datetime"],
                    y=crash_df["btc_close"],
                    mode="markers",
                    name="BTC Crash Flag"
                )
            )

            fig_crash.update_layout(
                title="Bitcoin Crash Period Detection",
                xaxis_title="Date",
                yaxis_title="BTC Close",
            )

            safe_plotly_chart(fig_crash)

            explain_box(
                "Crash Period Detection",
                "It marks periods where BTC experienced extreme negative movement according to the crash flag.",
                "Crash points appear during sharp price decline periods.",
                "Crypto crashes often happen because of panic selling, liquidity shocks, regulatory news, exchange issues, or broad risk-off market sentiment.",
                "The crash flag helps convert visual price drops into measurable risk events.",
            )
        else:
            st.warning("Crash columns were not found.")

        # ============================================================
        # FEATURE 5: BOOM PERIOD IDENTIFICATION
        # ============================================================

        st.markdown("## Feature 5: Boom Period Identification")

        if "btc_close" in df.columns and "btc_boom_flag" in df.columns:
            fig_boom = go.Figure()

            fig_boom.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["btc_close"],
                    mode="lines",
                    name="BTC Close"
                )
            )

            boom_df = df[df["btc_boom_flag"] == 1]

            fig_boom.add_trace(
                go.Scatter(
                    x=boom_df["datetime"],
                    y=boom_df["btc_close"],
                    mode="markers",
                    name="BTC Boom Flag"
                )
            )

            fig_boom.update_layout(
                title="Bitcoin Boom Period Identification",
                xaxis_title="Date",
                yaxis_title="BTC Close",
            )

            safe_plotly_chart(fig_boom)

            explain_box(
                "Boom Period Identification",
                "It marks periods where BTC experienced unusually strong positive movement according to the boom flag.",
                "Boom points appear during rapid upward price movements.",
                "Crypto booms are often related to speculative buying, FOMO, liquidity inflow, institutional interest, and positive market narratives.",
                "Separating boom periods from normal growth helps explain why crypto markets can rise very quickly but also reverse sharply.",
            )
        else:
            st.warning("Boom columns were not found.")

        # ============================================================
        # FEATURE 6: CORRELATION / CONNECTEDNESS
        # ============================================================

        st.markdown("## Feature 6: Correlation / Connectedness")

        corr_candidates = [
            col for col in [
                "corr_btc_nasdaq_30d",
                "corr_eth_nasdaq_30d",
            ]
            if col in df.columns
        ]

        if corr_candidates:
            corr_df = df[
                ["datetime"] + corr_candidates
            ].copy()

            corr_df = corr_df.rename(columns={
                "corr_btc_nasdaq_30d": "BTC-NASDAQ",
                "corr_eth_nasdaq_30d": "ETH-NASDAQ",
            })

            melted_corr = corr_df.melt(
                id_vars="datetime",
                var_name="Pair",
                value_name="30D Rolling Correlation"
            )

            fig_corr = px.line(
                melted_corr,
                x="datetime",
                y="30D Rolling Correlation",
                color="Pair",
                title="Crypto vs NASDAQ Rolling Correlation",
            )

            safe_plotly_chart(fig_corr)

            explain_box(
                "Correlation / Connectedness",
                "It shows whether crypto and NASDAQ returns move together over time.",
                "Correlation changes across time and may increase during stress periods.",
                "During uncertainty, investors may treat both crypto and stocks as risky assets and sell them together.",
                "This means diversification benefits are not constant and can weaken during crisis periods.",
            )
        else:
            st.warning("Correlation columns were not found.")

        # ============================================================
        # FEATURE 7: CRYPTO VS STOCK MARKET COMPARISON
        # ============================================================

        st.markdown("## Feature 7: Crypto vs. Stock Market Comparison")

        comparison_assets = {
            "Bitcoin": "btc_return",
            "Ethereum": "eth_return",
            "NASDAQ": "nasdaq_return",
        }

        available_comparison = {
            asset: col for asset, col in comparison_assets.items()
            if col in df.columns
        }

        if len(available_comparison) == 0:
            st.warning(
                "Feature 7 cannot be displayed because return columns were not found. "
                "Expected columns: btc_return, eth_return, nasdaq_return."
            )
        else:
            comparison_data = []

            for asset, col in available_comparison.items():
                asset_type = "Crypto" if asset in ["Bitcoin", "Ethereum"] else "Stock Market"

                comparison_data.append({
                    "Asset": asset,
                    "Market Type": asset_type,
                    "Average Return": df[col].mean(),
                    "Return Volatility": df[col].std(),
                    "Minimum Return": df[col].min(),
                    "Maximum Return": df[col].max(),
                })

            comparison_df = pd.DataFrame(comparison_data)

            fig_comparison = px.bar(
                comparison_df,
                x="Asset",
                y="Return Volatility",
                color="Market Type",
                title="Crypto vs Stock Market Comparison Based on Return Volatility",
                text_auto=True,
            )

            safe_plotly_chart(fig_comparison)
            safe_dataframe(comparison_df)

            explain_box(
                "Crypto vs Stock Market Comparison",
                "This graph compares Bitcoin, Ethereum, and NASDAQ using return volatility.",
                "Cryptocurrency assets usually show higher return volatility than NASDAQ.",
                "Crypto markets are more speculative, trade continuously, and react strongly to investor sentiment and market shocks.",
                "This comparison supports the main objective of the project: identifying how crypto differs from traditional stock markets in risk and instability.",
            )

        # ============================================================
        # FEATURE 8: RISK ANALYSIS
        # ============================================================

        st.markdown("## Feature 8: Risk Analysis")

        risk_features = {
            "BTC Volatility": "btc_volatility_30d",
            "ETH Volatility": "eth_volatility_30d",
            "NASDAQ Volatility": "nasdaq_volatility_30d",
            "BTC Drawdown": "btc_drawdown",
            "ETH Drawdown": "eth_drawdown",
            "NASDAQ Drawdown": "nasdaq_drawdown",
            "BTC vs NASDAQ Risk Ratio": "risk_ratio_btc_vs_nasdaq",
        }

        available_risk = {
            name: col for name, col in risk_features.items()
            if col in df.columns
        }

        if len(available_risk) == 0:
            st.warning(
                "Feature 8 cannot be displayed because risk-related columns were not found. "
                "Expected columns include volatility, drawdown, or risk ratio columns."
            )
        else:
            risk_data = []

            for name, col in available_risk.items():
                risk_data.append({
                    "Risk Indicator": name,
                    "Average Value": df[col].mean(),
                    "Maximum Value": df[col].max(),
                    "Minimum Value": df[col].min(),
                })

            risk_df = pd.DataFrame(risk_data)

            fig_risk = px.bar(
                risk_df,
                x="Risk Indicator",
                y="Average Value",
                title="Average Risk Indicators Across Assets",
                text_auto=True,
            )

            fig_risk.update_layout(
                xaxis_tickangle=-45
            )

            safe_plotly_chart(fig_risk)
            safe_dataframe(risk_df)

            explain_box(
                "Risk Analysis",
                "This graph summarizes risk indicators such as rolling volatility, drawdown, and BTC-NASDAQ risk ratio.",
                "Crypto assets generally show higher risk values compared to NASDAQ.",
                "This happens because crypto prices are more sensitive to liquidity shocks, investor sentiment, rapid selling pressure, and speculative trading.",
                "Risk analysis helps investors and analysts understand which assets are more unstable and require stronger risk management.",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
