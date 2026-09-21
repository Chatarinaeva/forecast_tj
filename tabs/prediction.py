import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def render_prediction_tab(
    df_all_year,
    prepare_historical_long,
    prepare_modeling_long,
    normalize_input_data,
    normalize_route_model,
    period_label_id,
    find_column,
    route_order_model,
    auto_selection_path,
    model_registry_path,
    model_params_path,
    reg_features_path,
    xgb_config_path,
    historical_calendar_path,
    regression_training_path,
    time_index_map_path,
    final_xgb_params_path,
    models_dir,
    scalers_dir,
    show_artifacts_status,
):
    # =========================================================
    # ALIAS PATH/KONFIGURASI DARI prototype.py
    # =========================================================
    AUTO_SELECTION_PATH = Path(auto_selection_path)
    MODEL_REGISTRY_PATH = Path(model_registry_path)
    MODEL_PARAMS_PATH = Path(model_params_path)
    REG_FEATURES_PATH = Path(reg_features_path)
    XGB_CONFIG_PATH = Path(xgb_config_path)
    HISTORICAL_CALENDAR_PATH = Path(historical_calendar_path)
    REGRESSION_TRAINING_PATH = Path(regression_training_path)
    TIME_INDEX_MAP_PATH = Path(time_index_map_path)
    FINAL_XGB_PARAMS_PATH = Path(final_xgb_params_path)

    MODELS_DIR = Path(models_dir)
    SCALERS_DIR = Path(scalers_dir)

    SHOW_ARTIFACTS_STATUS = show_artifacts_status

    st.subheader("Prediksi Jumlah Penumpang")

    st.markdown(
        """
        Pada tab ini, pengguna dapat mengunggah data terbaru, memilih jalur,
        menentukan jumlah bulan prediksi, serta menandai jenis periode libur
        untuk bulan yang akan diprediksi.
        """
    )
    st.divider()

    st.markdown("### Input Data untuk Prediksi")

    # =========================================================
    # 1. TEMPLATE UPLOAD
    # =========================================================
    def create_template_data():
        template = pd.DataFrame({
            "year": [2026, 2026, 2026, 2026, 2026, 2026],
            "month": ["january", "february", "march", "january", "february", "march"],
            "route": ["1B", "1B", "1B", "2B", "2B", "2B"],
            "passengers": [0, 0, 0, 0, 0, 0]
        })

        return template

    template_df = create_template_data()
    template_csv = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Template Data Upload",
        data=template_csv,
        file_name="template_data_penumpang.csv",
        mime="text/csv"
    )

    # =========================================================
    # 2. UPLOAD DATA TERBARU
    # =========================================================
    uploaded_file = st.file_uploader(
        "Upload data terbaru dalam format CSV atau Excel",
        type=["csv", "xlsx"]
    )

    historical_long_detail = prepare_historical_long(df_all_year)
    historical_long = prepare_modeling_long(historical_long_detail)

    uploaded_long = pd.DataFrame()

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_upload = pd.read_csv(uploaded_file)
            else:
                raw_upload = pd.read_excel(uploaded_file)

            uploaded_long_detail = normalize_input_data(raw_upload)
            uploaded_long = prepare_modeling_long(uploaded_long_detail)

            st.success("Data terbaru berhasil dibaca dan divalidasi.")

            with st.expander("Lihat preview data terbaru"):
                preview_upload = uploaded_long.copy()
                preview_upload["periode"] = preview_upload["period"].apply(period_label_id)

                st.dataframe(
                    preview_upload[
                        ["periode", "year", "month", "route", "passengers"]
                    ],
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Data gagal dibaca: {e}")
            st.stop()

    if uploaded_file is not None:
        combined_data = pd.concat(
            [historical_long, uploaded_long],
            ignore_index=True
        )
    else:
        combined_data = historical_long.copy()

    combined_data = (
        combined_data
        .sort_values(["route", "period"])
        .drop_duplicates(subset=["route", "period"], keep="last")
        .reset_index(drop=True)
    )

    st.info(
        f"Data yang digunakan mencakup periode "
        f"{period_label_id(combined_data['period'].min())} sampai "
        f"{period_label_id(combined_data['period'].max())}."
    )

    # =========================================================
    # 3. LOAD MODEL ARTIFACT DAN MAPPING MODEL TERBAIK
    # =========================================================
    @st.cache_data
    def load_auto_selection_result():
        if not AUTO_SELECTION_PATH.exists():
            return pd.DataFrame()

        data = pd.read_csv(AUTO_SELECTION_PATH)
        data.columns = [str(col).strip() for col in data.columns]

        route_col = find_column(data.columns, ["route", "jalur", "rute"])
        model_col = find_column(data.columns, ["best_model", "model_terbaik", "model"])

        if route_col is None or model_col is None:
            return pd.DataFrame()

        data = data.copy()
        data["route"] = data[route_col].apply(normalize_route_model)
        data["best_model"] = data[model_col].astype(str)

        return data


    @st.cache_data
    def load_model_registry():
        if not MODEL_REGISTRY_PATH.exists():
            return pd.DataFrame()

        registry = pd.read_csv(MODEL_REGISTRY_PATH)
        registry.columns = [str(col).strip() for col in registry.columns]

        route_col = find_column(registry.columns, ["route", "jalur", "rute"])
        if route_col is None:
            return pd.DataFrame()

        registry = registry.copy()
        registry["route"] = registry[route_col].apply(normalize_route_model)

        return registry


    @st.cache_resource
    def load_model_params():
        if not MODEL_PARAMS_PATH.exists():
            return {}

        return joblib.load(MODEL_PARAMS_PATH)

    @st.cache_resource
    def load_reg_features():
        if not REG_FEATURES_PATH.exists():
            return []

        return joblib.load(REG_FEATURES_PATH)

    @st.cache_resource
    def load_xgb_config():
        if not XGB_CONFIG_PATH.exists():
            return {}

        return joblib.load(XGB_CONFIG_PATH)

    @st.cache_data
    def load_historical_calendar_features():
        if not HISTORICAL_CALENDAR_PATH.exists():
            return pd.DataFrame()

        data = pd.read_csv(HISTORICAL_CALENDAR_PATH)
        data["period"] = pd.to_datetime(data["period"]).dt.to_period("M").dt.to_timestamp()

        return data

    @st.cache_data
    def load_regression_training_features():
        if not REGRESSION_TRAINING_PATH.exists():
            return pd.DataFrame()

        data = pd.read_csv(REGRESSION_TRAINING_PATH)

        data["route"] = data["route"].apply(normalize_route_model)
        data["period"] = pd.to_datetime(data["period"]).dt.to_period("M").dt.to_timestamp()

        numeric_cols = [
            col for col in data.columns
            if col not in ["route", "period"]
        ]

        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    @st.cache_data
    def load_time_index_map():
        if not TIME_INDEX_MAP_PATH.exists():
            return {}

        data = pd.read_csv(TIME_INDEX_MAP_PATH)
        data["route"] = data["route"].apply(normalize_route_model)
        data["last_time_index"] = pd.to_numeric(
            data["last_time_index"],
            errors="coerce"
        )

        data = data.dropna(subset=["route", "last_time_index"])
        data["last_time_index"] = data["last_time_index"].astype(int)

        return dict(zip(data["route"], data["last_time_index"]))

    @st.cache_resource
    def load_final_xgb_params():
        if not FINAL_XGB_PARAMS_PATH.exists():
            return {}

        return joblib.load(FINAL_XGB_PARAMS_PATH)

    def resolve_artifact_path(saved_path, fallback_folder):
        saved_path = Path(str(saved_path))

        if saved_path.exists():
            return saved_path

        fallback_path = fallback_folder / saved_path.name

        if fallback_path.exists():
            return fallback_path

        return saved_path


    df_auto_selection_streamlit = load_auto_selection_result()
    model_registry = load_model_registry()
    model_params_streamlit = load_model_params()
    reg_features_streamlit = load_reg_features()
    xgb_config_streamlit = load_xgb_config()
    historical_calendar_streamlit = load_historical_calendar_features()
    regression_training_streamlit = load_regression_training_features()
    time_index_map_streamlit = load_time_index_map()
    final_xgb_params_streamlit = load_final_xgb_params()

    if df_auto_selection_streamlit.empty:
        model_mapping = {}
    else:
        model_mapping = dict(
            zip(
                df_auto_selection_streamlit["route"],
                df_auto_selection_streamlit["best_model"]
            )
        )


    if SHOW_ARTIFACTS_STATUS:
        if df_auto_selection_streamlit.empty:
            st.warning(
                "File auto_selection_result.csv belum ditemukan di folder model_artifacts. "
                "Sistem akan menggunakan model regresi sederhana sebagai fallback."
            )
        else: 
            st.success("Mapping model terbaik berhasil dimuat dari model_artifacts/auto_selection_result.csv")

        if not model_registry.empty:
            st.success("Model registry berhasil dimuat dari model_artifacts/model_registry.csv")
        else:
            st.warning("model_registry.csv belum ditemukan atau belum dapat dibaca.")

        if len(model_params_streamlit) > 0:
            st.success("Parameter model terbaik berhasil dimuat dari model_artifacts/model_params.pkl")
        else:
            st.warning("model_params.pkl belum ditemukan. XGBoost akan memakai parameter default dashboard.")

        if len(reg_features_streamlit) > 0:
            st.success("reg_features berhasil dimuat dari model_artifacts/reg_features.pkl")
        else:
            st.warning("reg_features.pkl belum ditemukan. Dashboard akan memakai urutan fitur default.")

        if len(xgb_config_streamlit) > 0:
            st.success("Konfigurasi XGBoost berhasil dimuat dari model_artifacts/xgb_config.pkl")
        else:
            st.warning("xgb_config.pkl belum ditemukan. Dashboard akan memakai konfigurasi XGBoost default.")

        if not historical_calendar_streamlit.empty:
            st.success("Fitur kalender historis berhasil dimuat dari model_artifacts/historical_calendar_features.csv")
        else:
            st.warning("historical_calendar_features.csv belum ditemukan. Fitur libur historis akan dianggap Normal.")

        if not regression_training_streamlit.empty:
            st.success("Data training regresi berhasil dimuat dari model_artifacts/regression_training_features.csv")
        else:
            st.warning("regression_training_features.csv belum ditemukan. XGBoost akan membentuk fitur training dari data dashboard.")

        if len(time_index_map_streamlit) > 0:
            st.success("time_index_map berhasil dimuat dari model_artifacts/time_index_map.csv")
        else:
            st.warning("time_index_map.csv belum ditemukan. Dashboard akan memakai time_index dari data aktif.")

        if len(final_xgb_params_streamlit) > 0:
            st.success("Parameter final XGBoost berhasil dimuat dari model_artifacts/final_xgb_params.pkl")
        else:
            st.warning("final_xgb_params.pkl belum ditemukan. XGBoost akan memakai model_params.pkl.")

    # =========================================================
    # 4. INPUT JALUR DAN JUMLAH BULAN PREDIKSI
    # =========================================================
    route_unique_data = sorted(combined_data["route"].dropna().unique())

    route_options_all = (
        [route for route in route_order_model if route in route_unique_data] +
        [route for route in route_unique_data if route not in route_order_model]
    )

    if not model_registry.empty and "route" in model_registry.columns:
        route_unique_model = sorted(model_registry["route"].dropna().unique())
    else:
        route_unique_model = []

    route_options = [
        route for route in route_options_all
        if route in route_unique_model
    ]

    unmodeled_routes = [
        route for route in route_options_all
        if route not in route_unique_model
    ]

    if len(route_options) == 0:
        st.error(
            "Tidak ada jalur pada dataset aktif yang memiliki model tersimpan. "
            "Periksa kembali file model_registry.csv."
        )
        st.stop()

    route_name_map = {
        "1A": "Prambanan - Malioboro",
        "1B": "Ngabean - Bandara Adisutjipto",
        "2A": "Condong Catur - Malioboro",
        "2B": "Condong Catur - Ngabean",
        "3A": "Condong Catur - Giwangan",
        "3B": "Giwangan - Condong Catur",
        "4A": "Giwangan - RSUP Sardjito",
        "4B": "Giwangan - UGM",
        "5A": "Jombor - Ambarukmo",
        "5B": "Jombor - Bandara Adisutjipto",
        "6": "Gamping - Malioboro",
        "8": "Jombor - Ngabean",
        "9": "Giwangan - Jombor",
        "10": "Gamping - Kusumanegara",
        "11": "Giwangan - RS Panti Rapih",
        "12": "Condong Catur - Pakem",
        "13": "Ngabean - Stadion TGP",
        "14": "Bandara Adisutjipto - Pakem",
        "15": "Malioboro - Palbapang"
    }

    route_info_pred = pd.DataFrame({
        "Jalur": route_options,
        "Rute": [
            route_name_map.get(route, "-")
            for route in route_options
        ]
    })

    with st.expander("Daftar Jalur dan Rute yang Dapat Diprediksi"):
        st.dataframe(
            route_info_pred,
            use_container_width=True,
            hide_index=True,
        )

    if len(unmodeled_routes) > 0:
        with st.expander("Jalur yang belum memiliki model tersimpan"):
            st.write(
                "Jalur berikut belum tersedia di `model_registry.csv` karena hanya memiliki 12 bulan data historis pada dataset, "
                "sedangkan pemodelan membutuhkan 36 bulan data."
            )

            st.dataframe(
                pd.DataFrame({
                    "route": unmodeled_routes,
                    "keterangan": "Belum memiliki model tersimpan"
                }),
                use_container_width=True
            )

    col_input_1, col_input_2 = st.columns(2)

    # with col_input_1:
    #     selected_routes_pred = st.multiselect(
    #         "Pilih jalur yang ingin diprediksi",
    #         options=route_options,
    #         default=route_options
    #     )

    with col_input_1:
        if "select_all_routes" not in st.session_state:
            st.session_state["select_all_routes"] = True

        if "selected_routes_pred" not in st.session_state:
            st.session_state["selected_routes_pred"] = route_options

        def update_route_selection():
            if st.session_state["select_all_routes"]:
                st.session_state["selected_routes_pred"] = route_options
            else:
                st.session_state["selected_routes_pred"] = []

        selected_routes_pred = st.multiselect(
            "Pilih jalur yang ingin diprediksi",
            options=route_options,
            key="selected_routes_pred",
            disabled=st.session_state["select_all_routes"],
            placeholder="Pilih satu atau beberapa jalur"
        )

        st.checkbox(
            "Pilih Semua Jalur",
            key="select_all_routes",
            on_change=update_route_selection
        )

    with col_input_2:
        forecast_horizon = st.slider(
            "Jumlah bulan prediksi",
            min_value=1,
            max_value=12,
            value=1
        )

    if len(selected_routes_pred) == 0:
        st.info("Pilih minimal 1 jalur untuk menjalankan prediksi.")
        st.stop()

    # =========================================================
    # 5. INPUT PERIODE PREDIKSI DAN PERIODE LIBUR
    # =========================================================
    # Membuat periode forecast berdasarkan bulan terakhir masing-masing jalur
    route_future_period_map = {}

    for route in selected_routes_pred:
        route_period_data = combined_data[
            combined_data["route"] == route
        ].copy()

        route_last_period = route_period_data["period"].max()
        route_start_period = route_last_period + pd.offsets.MonthBegin(1)

        route_future_periods = pd.date_range(
            start=route_start_period,
            periods=forecast_horizon,
            freq="MS"
        )

        route_future_period_map[route] = route_future_periods

    # Menggabungkan seluruh periode forecast unik dari semua jalur terpilih
    future_periods = pd.DatetimeIndex(
        sorted({
            pd.Timestamp(period).to_period("M").to_timestamp()
            for periods in route_future_period_map.values()
            for period in periods
        })
    )

    if uploaded_file is not None:
        st.info(
            "Dataset aktif menggunakan **data historis lama + data terbaru yang di-upload**. "
            "Periode prediksi ditentukan otomatis berdasarkan bulan terakhir masing-masing jalur."
        )
    else:
        st.info(
            "Dataset aktif hanya menggunakan **data historis bawaan dashboard**. "
            "Periode prediksi ditentukan otomatis berdasarkan bulan terakhir masing-masing jalur."
        )

    st.markdown("#### Jenis Periode Prediksi")

    st.caption(
        f"Kategori periode untuk **{forecast_horizon} bulan ke depan** "
        "diterapkan untuk seluruh jalur terpilih."
    )

    holiday_options = [
        "Normal",
        "Libur Lebaran",
        "Libur Akhir Tahun Ajaran",
        "Libur Akhir Tahun"
    ]

    col_head_1, col_head_2 = st.columns([1, 1.3])

    with col_head_1:
        st.markdown("**Periode Prediksi**")

    with col_head_2:
        st.markdown("**Jenis Periode**")

    holiday_rows = []

    for i, period in enumerate(future_periods):
        period_key = pd.Timestamp(period).to_period("M").to_timestamp()
        periode_label = period_label_id(period_key)

        col_period, col_holiday = st.columns([1, 1.3])

        with col_period:
            st.markdown(
                f"""
                <div style="
                    padding: 0.55rem 0.75rem;
                    border: 1px solid rgba(250,250,250,0.15);
                    border-radius: 0.4rem;
                    min-height: 38px;
                    display: flex;
                    align-items: center;
                ">
                    {periode_label}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_holiday:
            selected_holiday = st.selectbox(
                label=f"Jenis Periode {periode_label}",
                options=holiday_options,
                index=0,
                key=f"holiday_type_{period_key.strftime('%Y_%m')}",
                label_visibility="collapsed"
            )

        holiday_rows.append({
            "periode_forecast": periode_label,
            "period_category": selected_holiday
        })

    edited_holiday_df = pd.DataFrame(holiday_rows)


    def build_future_features(edited_df, periods):
        future_features = edited_df.copy()

        future_features["period"] = pd.to_datetime(list(periods))
        future_features["period"] = (
            future_features["period"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        future_features["month_num"] = future_features["period"].dt.month

        future_features["is_eid_holiday"] = 0
        future_features["is_school_year_end"] = 0
        future_features["is_year_end_holiday"] = 0

        future_features.loc[
            future_features["period_category"] == "Libur Lebaran",
            "is_eid_holiday"
        ] = 1

        future_features.loc[
            future_features["period_category"] == "Libur Akhir Tahun Ajaran",
            "is_school_year_end"
        ] = 1

        future_features.loc[
            future_features["period_category"] == "Libur Akhir Tahun",
            "is_year_end_holiday"
        ] = 1

        return future_features


    def build_route_future_features(periods, holiday_category_map):
        route_holiday_rows = []

        for period in periods:
            period_key = pd.Timestamp(period).to_period("M").to_timestamp()

            route_holiday_rows.append({
                "periode_forecast": period_label_id(period_key),
                "period_category": holiday_category_map.get(period_key, "Normal")
            })

        return build_future_features(
            pd.DataFrame(route_holiday_rows),
            periods
        )


    future_features_all = build_future_features(
        edited_holiday_df,
        future_periods
    )

    holiday_category_map = dict(
        zip(
            future_features_all["period"],
            future_features_all["period_category"]
        )
    )

    st.divider()

    # =========================================================
    # 6. HELPER FORECAST BERDASARKAN SAVED MODEL
    # =========================================================
    HOLIDAY_COLS = [
        "is_eid_holiday",
        "is_school_year_end",
        "is_year_end_holiday"
    ]

    DEFAULT_REG_FEATURES = [
        "month_num",
        "quarter",
        "time_index",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "month_sin",
        "month_cos",
        "is_eid_holiday",
        "is_school_year_end",
        "is_year_end_holiday"
    ]

    if len(reg_features_streamlit) > 0:
        REG_FEATURES = list(reg_features_streamlit)
    else:
        REG_FEATURES = DEFAULT_REG_FEATURES

    DEFAULT_DL_FEATURES = [
        "passengers_scaled",
        "month_sin",
        "month_cos",
        "is_eid_holiday",
        "is_school_year_end",
        "is_year_end_holiday"
    ]

    def add_time_features(data):
        data = data.copy()

        data["period"] = pd.to_datetime(data["period"]).dt.to_period("M").dt.to_timestamp()
        data["month_num"] = data["period"].dt.month
        data["quarter"] = data["period"].dt.quarter
        data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)

        if not historical_calendar_streamlit.empty:
            calendar_data = historical_calendar_streamlit.copy()
            calendar_data["period"] = (
                pd.to_datetime(calendar_data["period"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )

            calendar_cols = ["period"] + [
                col for col in HOLIDAY_COLS
                if col in calendar_data.columns
            ]

            calendar_data = calendar_data[calendar_cols].drop_duplicates(subset=["period"])

            data = data.merge(
                calendar_data,
                on="period",
                how="left",
                suffixes=("", "_calendar")
            )

            for col in HOLIDAY_COLS:
                calendar_col = f"{col}_calendar"

                if col in data.columns and calendar_col in data.columns:
                    data[col] = data[col].fillna(data[calendar_col])
                    data = data.drop(columns=[calendar_col])
                elif calendar_col in data.columns:
                    data[col] = data[calendar_col]
                    data = data.drop(columns=[calendar_col])
                elif col not in data.columns:
                    data[col] = 0

                data[col] = data[col].fillna(0).astype(int)

        else:
            for col in HOLIDAY_COLS:
                if col not in data.columns:
                    data[col] = 0

                data[col] = data[col].fillna(0).astype(int)

        return data

    def prepare_regression_route_data(route_df):
        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        route_df = add_time_features(route_df)

        route_df["passengers"] = route_df["passengers"].astype(float)
        route_df["time_index"] = np.arange(1, len(route_df) + 1)

        route_df["lag_1"] = route_df["passengers"].shift(1)
        route_df["lag_2"] = route_df["passengers"].shift(2)
        route_df["lag_3"] = route_df["passengers"].shift(3)

        route_df["rolling_mean_3"] = (
            route_df["passengers"]
            .shift(1)
            .rolling(window=3)
            .mean()
        )

        return route_df

    def build_future_reg_features_dashboard(route_df, future_df, route=None):
        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        future_df = future_df.copy().reset_index(drop=True)
        future_df = add_time_features(future_df)

        route_norm = normalize_route_model(route) if route is not None else None

        if (
            uploaded_file is None
            and route_norm is not None
            and route_norm in time_index_map_streamlit
        ):
            last_time_index = int(time_index_map_streamlit[route_norm])
        else:
            last_time_index = int(route_df["time_index"].max())

        future_reg = future_df.copy()

        future_reg["time_index"] = np.arange(
            last_time_index + 1,
            last_time_index + len(future_reg) + 1
        )

        for col in ["lag_1", "lag_2", "lag_3", "rolling_mean_3"]:
            future_reg[col] = np.nan

        missing_cols = [col for col in REG_FEATURES if col not in future_reg.columns]

        if missing_cols:
            raise ValueError(f"Kolom future belum lengkap: {missing_cols}")

        return future_reg[REG_FEATURES].copy()

    def forecast_recursive_regressor_dashboard(model, X_future, y_history, feature_names=None):
        if feature_names is None:
            feature_names = REG_FEATURES

        history = list(
            pd.Series(y_history)
            .dropna()
            .astype(float)
            .values
        )

        predictions = []
        used_rows = []

        for i in range(len(X_future)):
            row = X_future.iloc[i].copy()

            row["lag_1"] = history[-1] if len(history) >= 1 else np.nan
            row["lag_2"] = history[-2] if len(history) >= 2 else np.nan
            row["lag_3"] = history[-3] if len(history) >= 3 else np.nan
            row["rolling_mean_3"] = np.mean(history[-3:]) if len(history) >= 3 else np.nan

            X_one = pd.DataFrame([row])

            missing_features = [
                feature for feature in feature_names
                if feature not in X_one.columns
            ]

            if missing_features:
                raise ValueError(
                    f"Fitur berikut belum tersedia untuk prediksi XGBoost: {missing_features}"
                )

            X_one = X_one[feature_names]
            X_one = X_one.apply(pd.to_numeric, errors="coerce")

            pred = float(model.predict(X_one)[0])

            predictions.append(pred)
            history.append(max(pred, 0))
            used_rows.append(row)

        X_future_used = pd.DataFrame(used_rows)

        return np.array(predictions), X_future_used

    XGB_DEFAULT_PARAMS_DASHBOARD = {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }

    def get_xgb_default_params_dashboard():
        if isinstance(xgb_config_streamlit, dict):
            params = xgb_config_streamlit.get(
                "xgb_default_params",
                XGB_DEFAULT_PARAMS_DASHBOARD
            )
            return params.copy()

        return XGB_DEFAULT_PARAMS_DASHBOARD.copy()


    def get_xgb_common_params_dashboard():
        if isinstance(xgb_config_streamlit, dict):
            params = xgb_config_streamlit.get("xgb_common_params", {})
            return params.copy()

        return {}

    def clean_model_params(params):
        cleaned = {}

        int_cols = [
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "random_state",
            "n_jobs"
        ]

        float_cols = [
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda"
        ]

        for key, value in params.items():
            if pd.isna(value):
                continue

            if key in int_cols:
                cleaned[key] = int(value)
            elif key in float_cols:
                cleaned[key] = float(value)
            else:
                cleaned[key] = value

        return cleaned


    def get_best_params_dashboard(route, algorithm):
        model_key = f"{route}_{algorithm}".replace(" ", "_")
        algorithm_lower = str(algorithm).lower()

        if (
            ("xgboost" in algorithm_lower or "xgb" in algorithm_lower)
            and model_key in final_xgb_params_streamlit
        ):
            params = final_xgb_params_streamlit.get(model_key, {}).copy()
            return clean_model_params(params)

        params = model_params_streamlit.get(model_key, {}).copy()
        return clean_model_params(params)

    def get_xgboost_training_data(route, route_df):
        route_norm = normalize_route_model(route)

        if uploaded_file is None and not regression_training_streamlit.empty:
            saved_route_df = regression_training_streamlit[
                regression_training_streamlit["route"] == route_norm
            ].copy()

            required_cols = ["period", "passengers"] + REG_FEATURES
            missing_cols = [
                col for col in required_cols
                if col not in saved_route_df.columns
            ]

            if not saved_route_df.empty and len(missing_cols) == 0:
                saved_route_df = (
                    saved_route_df
                    .sort_values("period")
                    .reset_index(drop=True)
                )

                return saved_route_df, "notebook_features"

        prepared_route_df = prepare_regression_route_data(route_df)

        return prepared_route_df, "dashboard_features"

    # def forecast_retrained_xgboost(route_df, future_df):
    #     from xgboost import XGBRegressor

    #     route = route_df["route"].iloc[0]

    #     route_df, feature_source = get_xgboost_training_data(
    #         route=route,
    #         route_df=route_df
    #     )

    #     train_df = route_df.dropna(subset=REG_FEATURES + ["passengers"]).copy()

    #     if len(train_df) < 12:
    #         raise ValueError("Data training terlalu sedikit untuk XGBoost.")

    #     X_full = train_df[REG_FEATURES].apply(pd.to_numeric, errors="coerce")
    #     y_full = train_df["passengers"].astype(float)

    #     X_future = build_future_reg_features_dashboard(
    #         route_df=route_df,
    #         future_df=future_df,
    #         route=route
    #     )

    #     best_params = get_best_params_dashboard(route, "XGBoost")

    #     final_params = get_xgb_default_params_dashboard()
    #     final_params.update(get_xgb_common_params_dashboard())
    #     final_params.update(best_params)
    #     final_params.pop("early_stopping_rounds", None)

    #     final_params = clean_model_params(final_params)

    #     model = XGBRegressor(**final_params)
    #     model.fit(X_full, y_full)

    #     pred, X_future_used = forecast_recursive_regressor_dashboard(
    #         model=model,
    #         X_future=X_future,
    #         y_history=route_df["passengers"]
    #     )

    #     return pred, "XGBoost", X_future_used

    def forecast_retrained_xgboost(route_df, future_df):
        from xgboost import XGBRegressor

        route = route_df["route"].iloc[0]

        route_df, feature_source = get_xgboost_training_data(
            route=route,
            route_df=route_df
        )

        train_df = route_df.dropna(
            subset=REG_FEATURES + ["passengers"]
        ).copy()

        if len(train_df) < 12:
            raise ValueError(
                "Data training terlalu sedikit untuk XGBoost."
            )

        X_full = train_df[
            REG_FEATURES
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y_full = train_df[
            "passengers"
        ].astype(float)

        X_future = build_future_reg_features_dashboard(
            route_df=route_df,
            future_df=future_df,
            route=route
        )

        best_params = get_best_params_dashboard(
            route,
            "XGBoost"
        )

        final_params = get_xgb_default_params_dashboard()
        final_params.update(
            get_xgb_common_params_dashboard()
        )
        final_params.update(
            best_params
        )

        final_params.pop(
            "early_stopping_rounds",
            None
        )

        final_params = clean_model_params(
            final_params
        )

        model = XGBRegressor(
            **final_params
        )

        model.fit(
            X_full,
            y_full
        )

        pred, X_future_used = (
            forecast_recursive_regressor_dashboard(
                model=model,
                X_future=X_future,
                y_history=route_df["passengers"]
            )
        )
        
        return (
            pred,
            "XGBoost",
            X_future_used
        )

    def forecast_saved_xgboost_model(model_object, route_df, future_df):
        feature_names = get_model_feature_names(model_object)

        route_df = prepare_regression_route_data(route_df)

        X_future = build_future_reg_features_dashboard(
            route_df=route_df,
            future_df=future_df
        )

        missing_features = [
            feature for feature in feature_names
            if feature not in X_future.columns
        ]

        if missing_features:
            raise ValueError(
                f"Fitur saved XGBoost tidak cocok dengan fitur dashboard: {missing_features}"
            )

        pred, X_future_used = forecast_recursive_regressor_dashboard(
            model=model_object,
            X_future=X_future,
            y_history=route_df["passengers"],
            feature_names=feature_names
        )

        return pred, "XGBoost", X_future_used

    def get_model_registry_row(route):
        if model_registry.empty:
            return None

        registry_data = model_registry.copy()

        if "route" not in registry_data.columns:
            return None

        registry_data["route"] = registry_data["route"].apply(normalize_route_model)
        route_norm = normalize_route_model(route)

        matched = registry_data[
            registry_data["route"] == route_norm
        ]

        if matched.empty:
            return None

        return matched.iloc[0]


    @st.cache_resource
    def load_saved_pickle_model(model_path_text):
        return joblib.load(model_path_text)

    @st.cache_resource
    def load_saved_xgboost_native_model(model_path_text):
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(model_path_text)

        return model


    @st.cache_resource
    def load_saved_keras_model(model_path_text):
        import tensorflow as tf
        return tf.keras.models.load_model(model_path_text)


    @st.cache_resource
    def load_saved_scaler(scaler_path_text):
        return joblib.load(scaler_path_text)


    def is_generic_feature_names(feature_names):
        if feature_names is None:
            return True

        feature_names = [str(feature) for feature in feature_names]

        return all(
            re.fullmatch(r"f\d+", feature) is not None
            for feature in feature_names
        )


    def get_model_feature_names(model_object):
        n_features = getattr(model_object, "n_features_in_", None)

        if hasattr(model_object, "feature_names_in_"):
            feature_names = list(model_object.feature_names_in_)

            if not is_generic_feature_names(feature_names):
                return feature_names

        try:
            booster_features = model_object.get_booster().feature_names

            if booster_features is not None and not is_generic_feature_names(booster_features):
                return list(booster_features)

        except Exception:
            pass

        if n_features is None:
            n_features = len(DEFAULT_REG_FEATURES)

        n_features = int(n_features)

        if n_features > len(DEFAULT_REG_FEATURES):
            raise ValueError(
                f"Model membutuhkan {n_features} fitur, tetapi DEFAULT_REG_FEATURES "
                f"hanya berisi {len(DEFAULT_REG_FEATURES)} fitur."
            )

        return DEFAULT_REG_FEATURES[:n_features]


    def forecast_saved_regression_model(model_object, route_df, future_df):
        feature_names = get_model_feature_names(model_object)

        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        route_df = add_time_features(route_df)
        route_df["time_index"] = np.arange(1, len(route_df) + 1)

        history_values = list(route_df["passengers"].astype(float).values)
        predictions = []

        for _, future_row in future_df.reset_index(drop=True).iterrows():
            row = {
                "month_num": future_row["month_num"],
                "quarter": future_row.get("quarter", future_row["period"].quarter),
                "time_index": len(history_values) + 1,
                "lag_1": history_values[-1] if len(history_values) >= 1 else 0,
                "lag_2": history_values[-2] if len(history_values) >= 2 else history_values[-1],
                "lag_3": history_values[-3] if len(history_values) >= 3 else history_values[-1],
                "rolling_mean_3": (
                    np.mean(history_values[-3:])
                    if len(history_values) >= 3
                    else np.mean(history_values)
                ),
                "month_sin": future_row.get(
                    "month_sin",
                    np.sin(2 * np.pi * future_row["month_num"] / 12)
                ),
                "month_cos": future_row.get(
                    "month_cos",
                    np.cos(2 * np.pi * future_row["month_num"] / 12)
                ),
                "is_eid_holiday": future_row.get("is_eid_holiday", 0),
                "is_school_year_end": future_row.get("is_school_year_end", 0),
                "is_year_end_holiday": future_row.get("is_year_end_holiday", 0),
            }

            missing_features = [
                feature for feature in feature_names
                if feature not in row
            ]

            if missing_features:
                raise ValueError(
                    f"Fitur berikut belum tersedia untuk prediksi regresi: {missing_features}"
                )

            X_future = pd.DataFrame([row])[feature_names]
            pred_value_raw = float(model_object.predict(X_future)[0])

            predictions.append(pred_value_raw)
            history_values.append(max(pred_value_raw, 0))

        return np.array(predictions)


    def forecast_saved_statistical_model(model_object, route_df, future_df, best_model):
        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        route_df = add_time_features(route_df)

        future_df = future_df.copy()
        future_df = add_time_features(future_df)

        try:
            needs_exog = getattr(model_object.model, "k_exog", 0) > 0
        except Exception:
            needs_exog = False

        # Perbarui state model jika ada data observasi baru
        working_model = model_object

        best_model_lower = str(best_model).strip().lower()

        is_holt_winters = (
            "holt" in best_model_lower
            or "winter" in best_model_lower
        )

        # Perbarui state Holt-Winters dengan observasi terbaru
        if is_holt_winters:
            model_nobs = int(len(model_object.model.endog))

            if len(route_df) > model_nobs:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                params = model_object.params

                updated_endog = model_object.data.orig_endog.copy()
                updated_endog = updated_endog.astype(float)

                new_data = route_df.iloc[model_nobs:].copy()

                for _, row in new_data.iterrows():
                    period_value = pd.Timestamp(row["period"])
                    passenger_value = float(row["passengers"])
                    updated_endog.loc[period_value] = passenger_value

                updated_endog = updated_endog.sort_index()
                updated_endog.index = pd.DatetimeIndex(updated_endog.index)
                updated_endog.index.name = "period"
                updated_endog.name = "passengers"

                # Validasi kontinuitas periode
                expected_index = pd.date_range(
                    start=updated_endog.index.min(),
                    end=updated_endog.index.max(),
                    freq="MS"
                )

                if not updated_endog.index.equals(expected_index):
                    raise ValueError(
                        "Data Holt-Winters harus memiliki periode bulanan yang berurutan "
                        "tanpa bulan yang hilang."
                    )

                model_kwargs = {
                    "trend": model_object.model.trend,
                    "damped_trend": model_object.model.damped_trend,
                    "seasonal": model_object.model.seasonal,
                    "seasonal_periods": model_object.model.seasonal_periods,
                    "initialization_method": "known",
                    "initial_level": float(params["initial_level"]),
                    "use_boxcox": params.get("use_boxcox", False),
                }

                if model_object.model.trend is not None:
                    model_kwargs["initial_trend"] = float(
                        params["initial_trend"]
                    )

                if model_object.model.seasonal is not None:
                    model_kwargs["initial_seasonal"] = np.asarray(
                        params["initial_seasons"],
                        dtype=float
                    )

                rebuilt_model = ExponentialSmoothing(
                    updated_endog,
                    **model_kwargs
                )

                fit_kwargs = {
                    "smoothing_level": float(params["smoothing_level"]),
                    "optimized": False,
                    "remove_bias": bool(params.get("remove_bias", False)),
                }

                if (
                    model_object.model.trend is not None
                    and not pd.isna(params.get("smoothing_trend"))
                ):
                    fit_kwargs["smoothing_trend"] = float(
                        params["smoothing_trend"]
                    )

                if (
                    model_object.model.seasonal is not None
                    and not pd.isna(params.get("smoothing_seasonal"))
                ):
                    fit_kwargs["smoothing_seasonal"] = float(
                        params["smoothing_seasonal"]
                    )

                if (
                    model_object.model.damped_trend
                    and not pd.isna(params.get("damping_trend"))
                ):
                    fit_kwargs["damping_trend"] = float(
                        params["damping_trend"]
                    )

                working_model = rebuilt_model.fit(
                    **fit_kwargs
                )

        # Perbarui state ARIMA/SARIMA/SARIMAX tanpa retraining
        elif hasattr(model_object, "append"):
            model_nobs = int(getattr(model_object, "nobs", len(route_df)))

            if len(route_df) > model_nobs:
                new_data = route_df.iloc[model_nobs:].copy()

                new_index = pd.DatetimeIndex(
                    pd.to_datetime(new_data["period"])
                )

                new_endog = pd.Series(
                    new_data["passengers"].astype(float).values,
                    index=new_index,
                    name="passengers"
                )

                new_endog.index.name = "period"

                if needs_exog:
                    new_exog = new_data[HOLIDAY_COLS].copy()
                    new_exog.index = new_index

                    working_model = model_object.append(
                        endog=new_endog,
                        exog=new_exog,
                        refit=False
                    )
                else:
                    working_model = model_object.append(
                        endog=new_endog,
                        refit=False
                    )

        # Siapkan variabel eksogen untuk periode prediksi
        exog_future = None

        if needs_exog:
            exog_future = future_df[HOLIDAY_COLS]

        try:
            if needs_exog:
                pred = working_model.forecast(
                    steps=len(future_df),
                    exog=exog_future
                )
            else:
                pred = working_model.forecast(
                    steps=len(future_df)
                )

            return np.array(pred)

        except Exception as e:
            raise ValueError(
                f"Model statistik gagal melakukan forecast: {e}"
            )


    def forecast_saved_deep_learning_model(model_object, scaler_object, route_df, future_df):
        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        route_df = add_time_features(route_df)
        future_df = add_time_features(future_df)

        route_df["passengers_scaled"] = scaler_object.transform(
            route_df[["passengers"]]
        ).ravel()

        input_shape = model_object.input_shape

        window_size_model = input_shape[1]
        n_features_model = input_shape[2]

        dl_features = DEFAULT_DL_FEATURES.copy()

        if n_features_model != len(dl_features):
            raise ValueError(
                f"Jumlah fitur DL tidak sesuai. Model membutuhkan {n_features_model} fitur, "
                f"tetapi dashboard menyiapkan {len(dl_features)} fitur."
            )

        working_features = route_df[dl_features].copy().reset_index(drop=True)

        if len(working_features) < window_size_model:
            raise ValueError(
                f"Data historis terlalu sedikit. Minimal membutuhkan {window_size_model} periode."
            )

        predictions = []

        for _, future_row in future_df.iterrows():
            X_window = working_features.tail(window_size_model).values.reshape(
                1,
                window_size_model,
                n_features_model
            )

            pred_scaled = float(model_object.predict(X_window, verbose=0).ravel()[0])

            pred_value = scaler_object.inverse_transform(
                np.array([[pred_scaled]])
            ).ravel()[0]

            pred_value = max(pred_value, 0)
            predictions.append(pred_value)

            next_scaled = scaler_object.transform(
                np.array([[pred_value]])
            ).ravel()[0]

            next_row = {
                "passengers_scaled": next_scaled,
                "month_sin": future_row["month_sin"],
                "month_cos": future_row["month_cos"],
                "is_eid_holiday": future_row["is_eid_holiday"],
                "is_school_year_end": future_row["is_school_year_end"],
                "is_year_end_holiday": future_row["is_year_end_holiday"]
            }

            working_features = pd.concat(
                [working_features, pd.DataFrame([next_row])],
                ignore_index=True
            )

        return np.array(predictions)

    def forecast_route_by_model(route_df, future_df, best_model_name):
        route = route_df["route"].iloc[0]
        registry_row = get_model_registry_row(route)

        if registry_row is None:
            raise ValueError(f"Model untuk jalur {route} tidak ditemukan di model_registry.csv.")

        best_model = str(registry_row["best_model"])
        best_model_lower = best_model.lower()

        route_df = route_df.sort_values("period").copy().reset_index(drop=True)
        future_df = future_df.copy().reset_index(drop=True)
        future_df = add_time_features(future_df)

        # if "xgboost" in best_model_lower or "xgb" in best_model_lower:
        #     pred, model_used, X_future_used = forecast_retrained_xgboost(
        #         route_df=route_df,
        #         future_df=future_df
        #     )

        #     return pred, model_used

        if "xgboost" in best_model_lower or "xgb" in best_model_lower:
            native_model_file = str(
                registry_row.get("native_model_file", "")
            ).strip()

            if native_model_file == "" or native_model_file.lower() == "nan":
                raise ValueError(
                    f"Native model XGBoost untuk jalur {route} tidak ditemukan di model_registry.csv."
                )

            native_model_path = resolve_artifact_path(
                native_model_file,
                MODELS_DIR
            )

            if not native_model_path.exists():
                raise ValueError(
                    f"File native XGBoost tidak ditemukan: {native_model_path}"
                )

            model_object = load_saved_xgboost_native_model(
                str(native_model_path)
            )

            pred, model_used, X_future_used = forecast_saved_xgboost_model(
                model_object=model_object,
                route_df=route_df,
                future_df=future_df
            )

            return pred, model_used

        model_path = resolve_artifact_path(
            registry_row["model_file"],
            MODELS_DIR
        )

        if not model_path.exists():
            raise ValueError(f"File model tidak ditemukan: {model_path}")


        if best_model_lower in ["lstm", "bilstm"]:
            scaler_file = str(registry_row.get("scaler_file", "")).strip()

            if scaler_file == "" or scaler_file.lower() == "nan":
                raise ValueError(f"Scaler untuk model {best_model} jalur {route} tidak ditemukan.")

            scaler_path = resolve_artifact_path(
                scaler_file,
                SCALERS_DIR
            )

            if not scaler_path.exists():
                raise ValueError(f"File scaler tidak ditemukan: {scaler_path}")

            model_object = load_saved_keras_model(str(model_path))
            scaler_object = load_saved_scaler(str(scaler_path))

            pred = forecast_saved_deep_learning_model(
                model_object=model_object,
                scaler_object=scaler_object,
                route_df=route_df,
                future_df=future_df
            )

            return pred, best_model

        elif "random" in best_model_lower or "forest" in best_model_lower:
            model_object = load_saved_pickle_model(str(model_path))

            pred = forecast_saved_regression_model(
                model_object=model_object,
                route_df=route_df,
                future_df=future_df
            )

            return pred, best_model

        elif (
            best_model_lower in ["arima", "sarima", "sarimax"]
            or "holt" in best_model_lower
            or "winter" in best_model_lower
        ):
            model_object = load_saved_pickle_model(str(model_path))

            pred = forecast_saved_statistical_model(
                model_object=model_object,
                route_df=route_df,
                future_df=future_df,
                best_model=best_model
            )

            return pred, best_model

        else:
            raise ValueError(f"Jenis model tidak dikenali: {best_model}")

    # =========================================================
    # 7. JALANKAN PREDIKSI
    # =========================================================
    run_prediction = st.button(
        "🚀 Jalankan Prediksi",
        type="primary"
    )

    if run_prediction:
        all_prediction_results = []
        skipped_routes = []

        progress_bar = st.progress(0)

        for idx, route in enumerate(selected_routes_pred):
            route_data = combined_data[
                combined_data["route"] == route
            ].copy()

            route_data = route_data.sort_values("period")
            n_period = route_data["period"].nunique()

            if n_period < 6:
                skipped_routes.append({
                    "route": route,
                    "jumlah_periode_data": n_period,
                    "keterangan": "Data terlalu sedikit untuk prediksi."
                })

                progress_bar.progress((idx + 1) / len(selected_routes_pred))
                continue

            best_model = model_mapping.get(route, "Tidak ditemukan")

            if get_model_registry_row(route) is None:
                skipped_routes.append({
                    "route": route,
                    "jumlah_periode_data": n_period,
                    "keterangan": "Belum memiliki model tersimpan di model_registry.csv."
                })

                progress_bar.progress((idx + 1) / len(selected_routes_pred))
                continue

            try:
                route_future_features = build_route_future_features(
                    periods=route_future_period_map[route],
                    holiday_category_map=holiday_category_map
                )

                pred_values, model_used = forecast_route_by_model(
                    route_data,
                    route_future_features,
                    best_model
                )

            except Exception as error:
                skipped_routes.append({
                    "route": route,
                    "jumlah_periode_data": n_period,
                    "keterangan": f"Gagal load/prediksi model: {error}"
                })

                progress_bar.progress((idx + 1) / len(selected_routes_pred))
                continue

            raw_pred_values = np.array(pred_values, dtype=float)
            pred_values_clipped = np.maximum(raw_pred_values, 0)
            pred_values_rounded = np.round(pred_values_clipped).astype(int)

            prediction_note = np.select(
                [
                    raw_pred_values < 0,
                    raw_pred_values == 0,
                    (raw_pred_values > 0) & (pred_values_rounded == 0)
                ],
                [
                    "Dikoreksi dari nilai negatif",
                    "Model menghasilkan nilai 0",
                    "Nilai sangat kecil, dibulatkan menjadi 0"
                ],
                default="-"
            )

            route_prediction = pd.DataFrame({
                "route": route,
                "period": route_future_features["period"],
                "periode": route_future_features["period"].apply(period_label_id),
                "period_category": route_future_features["period_category"],
                "best_model_mapping": best_model,
                "model_used": model_used,
                "raw_prediction": np.round(raw_pred_values, 2),
                "prediction_note": prediction_note,
                "prediction": pred_values_rounded
            })

            all_prediction_results.append(route_prediction)
            progress_bar.progress((idx + 1) / len(selected_routes_pred))

        progress_bar.empty()

        # =====================================================
        # 8. OUTPUT HASIL PREDIKSI
        # =====================================================
        if len(all_prediction_results) > 0:
            final_prediction_df = pd.concat(
                all_prediction_results,
                ignore_index=True
            )

            st.success("Prediksi berhasil dibuat.")
            st.markdown("### Hasil Prediksi")
            st.markdown("#### Tabel Hasil Prediksi")

            # Kolom yang ditampilkan pada prototipe
            display_df = final_prediction_df[
                [
                    "route",
                    "periode",
                    "period_category",
                    "model_used",
                    "prediction"
                ]
            ].rename(
                columns={
                    "route": "Jalur",
                    "periode": "Periode",
                    "period_category": "Jenis Periode",
                    "model_used": "Model",
                    "prediction": "Prediksi"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # Kolom yang disimpan pada CSV
            output_df = final_prediction_df[
                [
                    "route",
                    "periode",
                    "period_category",
                    "best_model_mapping",
                    "model_used",
                    "raw_prediction",
                    "prediction",
                    "prediction_note"
                ]
            ].rename(
                columns={
                    "route": "Jalur",
                    "periode": "Periode",
                    "period_category": "Jenis Periode",
                    "best_model_mapping": "Model Terbaik",
                    "model_used": "Model Digunakan",
                    "raw_prediction": "Prediksi Mentah",
                    "prediction": "Prediksi",
                    "prediction_note": "Keterangan Hasil Prediksi"
                }
            )

            result_csv = (
                output_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            # st.dataframe(
            #     final_prediction_df[display_columns],
            #     use_container_width=True
            # )

            # result_csv = (
            #     final_prediction_df[output_columns]
            #     .to_csv(index=False)
            #     .encode("utf-8")
            # )

            st.download_button(
                label="⬇️ Download Hasil Prediksi CSV",
                data=result_csv,
                file_name="final_forecast_result.csv",
                mime="text/csv"
            )

            st.divider()

        # =================================================
        # 9. GRAFIK AKTUAL VS PREDIKSI PER JALUR
        # =================================================
        st.markdown("#### Grafik Aktual vs Prediksi per Jalur")

        plot_routes = list(final_prediction_df["route"].dropna().unique())

        n_cols = 3 if len(plot_routes) > 1 else 1
        n_rows = int(np.ceil(len(plot_routes) / n_cols))

        fig_pred, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.4 * n_cols, 3.8 * n_rows),
            dpi=120
        )

        if len(plot_routes) == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).reshape(-1)

        for idx, route in enumerate(plot_routes):
            ax = axes[idx]

            hist_plot = combined_data[
                combined_data["route"] == route
            ].copy()

            hist_plot = hist_plot.sort_values("period").tail(12)

            pred_plot = final_prediction_df[
                final_prediction_df["route"] == route
            ].copy()

            pred_plot = pred_plot.sort_values("period")

            # Garis aktual
            ax.plot(
                hist_plot["period"],
                hist_plot["passengers"],
                marker="o",
                linewidth=1.8,
                label="Aktual"
            )

            # Membuat garis prediksi tersambung dari titik aktual terakhir
            last_actual_point = hist_plot.tail(1)[
                ["period", "passengers"]
            ].rename(
                columns={"passengers": "prediction"}
            )

            pred_line = pd.concat(
                [
                    last_actual_point,
                    pred_plot[["period", "prediction"]]
                ],
                ignore_index=True
            )

            pred_connector, = ax.plot(
                pred_line["period"],
                pred_line["prediction"],
                linestyle="--",
                linewidth=1.8,
                marker=None,
                label="Prediksi"
            )

            pred_color = pred_connector.get_color()

            ax.plot(
                pred_plot["period"],
                pred_plot["prediction"],
                marker="o",
                linestyle="None",
                color=pred_color,
                label="_nolegend_"
            )

            ax.set_title(f"Jalur {route}", fontsize=10)
            ax.set_xlabel("Periode", fontsize=8)
            ax.set_ylabel("Jumlah Penumpang", fontsize=8)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.ticklabel_format(style="plain", axis="y")
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7)

        for j in range(len(plot_routes), len(axes)):
            axes[j].axis("off")

        fig_pred.suptitle(
            "Aktual vs Prediksi Jumlah Penumpang per Jalur",
            fontsize=13,
            y=1.02
        )

        plt.tight_layout()
        st.pyplot(fig_pred, use_container_width=True)

        # =====================================================
        # 10. JALUR YANG TIDAK DIPREDIKSI
        # =====================================================
        if len(skipped_routes) > 0:
            st.warning("Beberapa jalur tidak diprediksi karena data belum mencukupi.")

            skipped_df = pd.DataFrame(skipped_routes)

            st.dataframe(
                skipped_df,
                use_container_width=True
            )
