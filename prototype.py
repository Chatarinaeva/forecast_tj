import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from tabs.visualization import render_visualization_tab
from tabs.prediction import render_prediction_tab

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dashboard Penumpang Bus Trans Jogja",
    layout="wide"
)

sns.set_style("darkgrid")

st.title(
    "🚌 Prototipe Sistem Visualisasi Data dan Prediksi Jumlah Penumpang Trans Jogja"
)

# KONFIGURASI UMUM
DATA_PATH = Path("transjogja_passengers.csv")

ARTIFACT_DIR = Path("model_artifacts")
MODELS_DIR = ARTIFACT_DIR / "models"
SCALERS_DIR = ARTIFACT_DIR / "scalers"
FORECAST_DIR = ARTIFACT_DIR / "forecast"

AUTO_SELECTION_PATH = ARTIFACT_DIR / "auto_selection_result.csv"
MODEL_REGISTRY_PATH = ARTIFACT_DIR / "model_registry.csv"
MODEL_PARAMS_PATH = ARTIFACT_DIR / "model_params.pkl"
REG_FEATURES_PATH = ARTIFACT_DIR / "reg_features.pkl"
XGB_CONFIG_PATH = ARTIFACT_DIR / "xgb_config.pkl"
HISTORICAL_CALENDAR_PATH = ARTIFACT_DIR / "historical_calendar_features.csv"
REGRESSION_TRAINING_PATH = ARTIFACT_DIR / "regression_training_features.csv"
TIME_INDEX_MAP_PATH = ARTIFACT_DIR / "time_index_map.csv"
FINAL_XGB_PARAMS_PATH = ARTIFACT_DIR / "final_xgb_params.pkl"

ENABLE_VISUALIZATION_TAB = True
SHOW_ARTIFACTS_STATUS = False

# DAFTAR BULAN
month_cols = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december"
]


month_alias = {
    "januari": 1,
    "jan": 1,
    "january": 1,

    "februari": 2,
    "feb": 2,
    "february": 2,

    "maret": 3,
    "mar": 3,
    "march": 3,

    "april": 4,
    "apr": 4,

    "mei": 5,
    "may": 5,

    "juni": 6,
    "jun": 6,
    "june": 6,

    "juli": 7,
    "jul": 7,
    "july": 7,

    "agustus": 8,
    "agu": 8,
    "aug": 8,
    "august": 8,

    "september": 9,
    "sep": 9,
    "sept": 9,

    "oktober": 10,
    "okt": 10,
    "oct": 10,
    "october": 10,

    "november": 11,
    "nov": 11,

    "desember": 12,
    "des": 12,
    "dec": 12,
    "december": 12,
}


month_num_to_en = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december"
}


month_num_to_id = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}

# URUTAN JALUR
route_order_viz = [
    "1A",
    "1B",
    "2A",
    "2B",
    "3A",
    "3B",
    "4A",
    "4B",
    "5A",
    "5B",
    "6A",
    "6B",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]


route_order_model = [
    "1B",
    "2B",
    "3A",
    "3B",
    "4A",
    "4B",
    "5A",
    "5B",
    "6",
    "8",
    "9",
    "10",
    "11",
    "13",
    "14",
    "15",
]

# HELPER DASAR
def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)

    return col.strip("_")


def find_column(columns, candidates):
    cleaned_map = {
        clean_column_name(col): col
        for col in columns
    }

    for candidate in candidates:
        if candidate in cleaned_map:
            return cleaned_map[candidate]

    for col in columns:
        cleaned = clean_column_name(col)

        for candidate in candidates:
            if candidate in cleaned:
                return col

    return None


def normalize_month(value):
    if pd.isna(value):
        return np.nan

    raw = str(value).strip()
    raw_lower = raw.lower()

    if re.fullmatch(r"\d{1,2}", raw_lower):
        month_num = int(raw_lower)

        if 1 <= month_num <= 12:
            return month_num

    if re.fullmatch(r"\d{4}", raw_lower):
        return np.nan

    if "-" in raw_lower or "/" in raw_lower:
        parsed_date = pd.to_datetime(
            raw_lower,
            errors="coerce",
            dayfirst=True
        )

        if not pd.isna(parsed_date):
            return parsed_date.month

    key = re.sub(r"[^a-z]", "", raw_lower)

    return month_alias.get(key, np.nan)


def normalize_route_historical(value):
    if pd.isna(value):
        return np.nan

    route = str(value).strip()
    route_clean = route.upper().replace(" ", "")

    route_alias_map = {
        "6A/6": "6A/6",
        "6/6A": "6A/6",
        "6A": "6A",
        "6B": "6B",
        "6": "6",
    }

    return route_alias_map.get(
        route_clean,
        route_clean
    )


def normalize_route_model(value):
    if pd.isna(value):
        return np.nan

    route = str(value).strip()
    route_clean = route.upper().replace(" ", "")

    route_alias_map = {
        "6": "6",
        "6A": "6",
        "6B": "6",
        "6A/6": "6",
        "6/6A": "6",
    }

    return route_alias_map.get(
        route_clean,
        route_clean
    )


def clean_number(value):
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        return value

    value = str(value).strip()

    if value == "" or value == "-":
        return np.nan

    value = value.replace(" ", "")
    value = re.sub(
        r"[^0-9,.\-]",
        "",
        value
    )

    if value == "":
        return np.nan

    if "," in value and "." in value:

        if value.rfind(",") > value.rfind("."):
            value = (
                value
                .replace(".", "")
                .replace(",", ".")
            )

        else:
            value = value.replace(",", "")

    elif "," in value:

        parts = value.split(",")

        if len(parts[-1]) == 3:
            value = value.replace(",", "")

        else:
            value = value.replace(",", ".")

    elif "." in value:

        parts = value.split(".")

        if len(parts[-1]) == 3:
            value = value.replace(".", "")

    return pd.to_numeric(
        value,
        errors="coerce"
    )


def period_label_id(period):
    return (
        f"{month_num_to_id[period.month]} "
        f"{period.year}"
    )

# LOAD DATA UTAMA
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    data.columns = [
        str(col).strip().lower()
        for col in data.columns
    ]

    required_cols = [
        "year",
        "route"
    ] + month_cols

    missing_cols = [
        col
        for col in required_cols
        if col not in data.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Kolom berikut belum ditemukan pada dataset: "
            f"{missing_cols}"
        )

    data = data[
        required_cols
    ].copy()

    data["year"] = pd.to_numeric(
        data["year"],
        errors="coerce"
    )

    data["route"] = (
        data["route"]
        .apply(normalize_route_historical)
    )

    for month in month_cols:
        data[month] = (
            data[month]
            .apply(clean_number)
        )

    data = data.dropna(
        subset=[
            "year",
            "route"
        ]
    )

    data["year"] = (
        data["year"]
        .astype(int)
    )

    data = data[
        data["route"]
        .astype(str)
        .str
        .upper()
        != "TOTAL"
    ]

    return data

# NORMALISASI DATA INPUT
def normalize_input_data(raw_df):
    data = raw_df.copy()

    data = data.dropna(
        how="all"
    )

    data.columns = [
        str(col).strip().lower()
        for col in data.columns
    ]

    year_col = find_column(
        data.columns,
        [
            "year",
            "tahun"
        ]
    )

    route_col = find_column(
        data.columns,
        [
            "route",
            "jalur",
            "rute"
        ]
    )

    month_col = find_column(
        data.columns,
        [
            "month",
            "bulan",
            "periode",
            "period"
        ]
    )

    value_col = find_column(
        data.columns,
        [
            "passengers",
            "jumlah_penumpang",
            "penumpang",
            "jumlah",
            "total"
        ]
    )

    if (
        year_col
        and route_col
        and month_col
        and value_col
    ):

        long_data = data[
            [
                year_col,
                month_col,
                route_col,
                value_col
            ]
        ].copy()

        long_data.columns = [
            "year",
            "month",
            "route",
            "passengers"
        ]

    else:

        if not year_col or not route_col:
            raise ValueError(
                "File harus memiliki kolom year dan route."
            )

        found_month_cols = []

        for col in data.columns:

            month_num = normalize_month(col)

            if pd.notna(month_num):
                found_month_cols.append(col)

        if len(found_month_cols) == 0:
            raise ValueError(
                "Kolom bulan tidak ditemukan. "
                "Gunakan nama bulan seperti january/february."
            )

        long_data = data.melt(
            id_vars=[
                year_col,
                route_col
            ],
            value_vars=found_month_cols,
            var_name="month",
            value_name="passengers"
        )

        long_data = long_data.rename(
            columns={
                year_col: "year",
                route_col: "route"
            }
        )

    long_data["year"] = pd.to_numeric(
        long_data["year"],
        errors="coerce"
    )

    long_data["month_num"] = (
        long_data["month"]
        .apply(normalize_month)
    )

    long_data["route"] = (
        long_data["route"]
        .apply(normalize_route_historical)
    )

    long_data["passengers"] = (
        long_data["passengers"]
        .apply(clean_number)
    )

    long_data = long_data.dropna(
        subset=[
            "year",
            "month_num",
            "route",
            "passengers"
        ]
    )

    long_data["year"] = (
        long_data["year"]
        .astype(int)
    )

    long_data["month_num"] = (
        long_data["month_num"]
        .astype(int)
    )

    long_data["passengers"] = (
        long_data["passengers"]
        .astype(float)
    )

    long_data = long_data[
        long_data["route"]
        .astype(str)
        .str
        .upper()
        != "TOTAL"
    ]

    long_data["period"] = pd.to_datetime(
        dict(
            year=long_data["year"],
            month=long_data["month_num"],
            day=1
        )
    )

    long_data["month"] = (
        long_data["month_num"]
        .map(month_num_to_en)
    )

    return long_data[
        [
            "year",
            "month",
            "month_num",
            "route",
            "passengers",
            "period"
        ]
    ].copy()

# PREPARE HISTORICAL LONG
def prepare_historical_long(df_source):
    available_month_cols = [
        month
        for month in month_cols
        if month in df_source.columns
    ]

    historical_raw = df_source.melt(
        id_vars=[
            "year",
            "route"
        ],
        value_vars=available_month_cols,
        var_name="month",
        value_name="passengers"
    )

    historical_long = normalize_input_data(
        historical_raw
    )

    return historical_long

# PREPARE DATA MODELING
def prepare_modeling_long(long_data):
    modeling_data = long_data.copy()

    modeling_data["route"] = (
        modeling_data["route"]
        .apply(normalize_route_model)
    )

    inactive_routes = [
        "1C",
        "7"
    ]

    modeling_data = modeling_data[
        ~modeling_data["route"]
        .isin(inactive_routes)
    ].copy()

    modeling_data = (
        modeling_data
        .groupby(
            [
                "route",
                "period",
                "year",
                "month_num"
            ],
            as_index=False
        )
        .agg(
            passengers=(
                "passengers",
                "sum"
            )
        )
    )

    modeling_data["month"] = (
        modeling_data["month_num"]
        .map(month_num_to_en)
    )

    return modeling_data[
        [
            "year",
            "month",
            "month_num",
            "route",
            "passengers",
            "period"
        ]
    ].copy()

# LOAD DATA
try:
    df = load_data()

except Exception as error:

    st.error(
        f"Data utama gagal dibaca: {error}"
    )

    st.stop()


# Menyimpan semua tahun
df_all_year = df.copy()


# # SIDEBAR - FILTER TAHUN
# st.sidebar.header(
#     "📅 Filter Tahun"
# )

# year_list = sorted(
#     df["year"]
#     .dropna()
#     .astype(int)
#     .unique()
# )

# selected_year = st.sidebar.selectbox(
#     "Pilih Tahun",
#     year_list,
#     index=len(year_list) - 1
# )

# df_selected_year = df[
#     df["year"] == selected_year
# ].copy()

# # SIDEBAR - FILTER JALUR
# st.sidebar.header(
#     "🛣️ Filter Jalur"
# )

# route_available = [
#     route
#     for route in route_order_viz
#     if route
#     in df_selected_year["route"].unique()
# ]

# selected_route_viz = st.sidebar.multiselect(
#     "Pilih Jalur",
#     options=route_available,
#     default=route_available
# )

# # SIDEBAR - SKALA PENUMPANG
# st.sidebar.markdown(
#     "### Skala Jumlah Penumpang"
# )

# unit = st.sidebar.selectbox(
#     "Pilih Skala",
#     [
#         "Jiwa",
#         "Ribu Jiwa",
#         "Juta Jiwa"
#     ],
#     index=1
# )

# scale_map = {
#     "Jiwa": 1,
#     "Ribu Jiwa": 1_000,
#     "Juta Jiwa": 1_000_000
# }

# scale = scale_map[unit]


# TABS
tab_viz, tab_pred = st.tabs(
    [
        "📊 Visualisasi",
        "🔮 Prediksi"
    ]
)

# # TAB VISUALISASI
# with tab_viz:

#     render_visualization_tab(
#         df_selected_year=df_selected_year,
#         df_all_year=df_all_year,
#         selected_year=selected_year,
#         selected_route_viz=selected_route_viz,
#         route_available=route_available,
#         month_cols=month_cols,
#         route_order_viz=route_order_viz,
#         scale=scale,
#         unit=unit,
#         enable_visualization_tab=ENABLE_VISUALIZATION_TAB,
#     )

# TAB VISUALISASI
with tab_viz:

    render_visualization_tab(
        df_all_year=df_all_year,
        month_cols=month_cols,
        route_order_viz=route_order_viz,
        enable_visualization_tab=ENABLE_VISUALIZATION_TAB,
    )

# TAB PREDIKSI
with tab_pred:

    render_prediction_tab(
        df_all_year=df_all_year,

        prepare_historical_long=prepare_historical_long,
        prepare_modeling_long=prepare_modeling_long,
        normalize_input_data=normalize_input_data,
        normalize_route_model=normalize_route_model,
        period_label_id=period_label_id,
        find_column=find_column,

        route_order_model=route_order_model,

        auto_selection_path=AUTO_SELECTION_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        model_params_path=MODEL_PARAMS_PATH,
        reg_features_path=REG_FEATURES_PATH,
        xgb_config_path=XGB_CONFIG_PATH,
        historical_calendar_path=HISTORICAL_CALENDAR_PATH,
        regression_training_path=REGRESSION_TRAINING_PATH,
        time_index_map_path=TIME_INDEX_MAP_PATH,
        final_xgb_params_path=FINAL_XGB_PARAMS_PATH,

        models_dir=MODELS_DIR,
        scalers_dir=SCALERS_DIR,

        show_artifacts_status=SHOW_ARTIFACTS_STATUS,
    )

# FOOTER
st.caption(
    "🚌 Dashboard Penumpang Bus | Data Tahunan"
)