import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def render_visualization_tab(
    df_all_year,
    month_cols,
    route_order_viz,
    enable_visualization_tab,
):
    # =========================
    # TAB VISUALISASI
    # =========================

    if not enable_visualization_tab:
        st.info(
            "Tab visualisasi sementara dinonaktifkan."
        )
        return

    # =========================
    # JUDUL DAN DESKRIPSI
    # =========================

    st.subheader("📊 Visualisasi Data Historis Jumlah Penumpang")

    st.write(
        "Pada tab ini, pengguna dapat melihat pola historis jumlah penumpang "
        "Trans Jogja berdasarkan tahun, jalur, dan skala jumlah penumpang."
    )

    route_info = pd.DataFrame({
        "Jalur": [
            "1A", "1B", "2A", "2B", "3A", "3B",
            "4A", "4B", "5A", "5B", "6", "8",
            "9", "10", "11", "12", "13", "14", "15"
        ],
        "Rute": [
            "Prambanan - Malioboro",
            "Ngabean - Bandara Adisutjipto",
            "Condong Catur - Malioboro",
            "Condong Catur - Ngabean",
            "Condong Catur - Giwangan",
            "Giwangan - Condong Catur",
            "Giwangan - RSUP Sardjito",
            "Giwangan - UGM",
            "Jombor - Ambarukmo",
            "Jombor - Bandara Adisutjipto",
            "Gamping - Malioboro",
            "Jombor - Ngabean",
            "Giwangan - Jombor",
            "Gamping - Kusumanegara",
            "Giwangan - RS Panti Rapih",
            "Condong Catur - Pakem",
            "Ngabean - Stadion TGP",
            "Bandara Adisutjipto - Pakem",
            "Malioboro - Palbapang"
        ]
    })

    with st.expander("Daftar Jalur dan Rute Trans Jogja"):
        st.dataframe(
            route_info,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # =========================
    # FILTER VISUALISASI
    # =========================

    st.markdown("### 🔎 Filter Visualisasi")

    year_list = sorted(
        df_all_year["year"]
        .dropna()
        .astype(int)
        .unique()
    )

    col_year, col_scale = st.columns(2)

    with col_year:
        selected_year = st.selectbox(
            "Pilih Tahun",
            options=year_list,
            index=len(year_list) - 1,
            key="viz_selected_year",
        )

    with col_scale:
        unit = st.selectbox(
            "Pilih Skala",
            options=[
                "Jiwa",
                "Ribu Jiwa",
                "Juta Jiwa",
            ],
            index=1,
            key="viz_unit",
        )

    scale_map = {
        "Jiwa": 1,
        "Ribu Jiwa": 1_000,
        "Juta Jiwa": 1_000_000,
    }

    scale = scale_map[unit]

    # Data sesuai tahun terpilih
    df_selected_year = df_all_year[
        df_all_year["year"] == selected_year
    ].copy()

    # Jalur yang benar-benar tersedia pada tahun terpilih
    route_available = [
        route
        for route in route_order_viz
        if route in df_selected_year["route"].unique()
    ]

    selected_route_viz = st.multiselect(
        "Pilih Jalur",
        options=route_available,
        default=route_available,
        key="viz_selected_routes",
    )

    st.divider()

    # =========================
    # PREPARE DATA
    # =========================

    df_long_viz = df_selected_year.melt(
        id_vars=["year", "route"],
        value_vars=month_cols,
        var_name="month",
        value_name="passengers",
    )

    yearly_long = df_all_year.melt(
        id_vars=["year", "route"],
        value_vars=month_cols,
        var_name="month",
        value_name="passengers",
    )

    df_long_viz["passengers"] = pd.to_numeric(
        df_long_viz["passengers"],
        errors="coerce",
    )

    yearly_long["passengers"] = pd.to_numeric(
        yearly_long["passengers"],
        errors="coerce",
    )

    df_long_viz = df_long_viz.dropna(
        subset=[
            "year",
            "route",
            "month",
            "passengers",
        ]
    ).copy()

    yearly_long = yearly_long.dropna(
        subset=[
            "year",
            "route",
            "month",
            "passengers",
        ]
    ).copy()

    df_long_viz["route"] = (
        df_long_viz["route"]
        .astype(str)
    )

    yearly_long["route"] = (
        yearly_long["route"]
        .astype(str)
    )

    # =========================
    # TOTAL TAHUNAN
    # =========================

    yearly_total = (
        yearly_long
        .groupby(
            "year",
            as_index=False,
        )["passengers"]
        .sum()
        .sort_values("year")
    )

    yearly_total["nilai_plot"] = (
        yearly_total["passengers"] / scale
    )

    # Data sesuai filter jalur
    filtered_df = df_long_viz[
        df_long_viz["route"].isin(
            selected_route_viz
        )
    ].copy()

    # =========================
    # 1. TREN TAHUNAN
    # =========================

    st.subheader(
        "📈 Tren Total Penumpang per Tahun"
    )

    if yearly_total.empty:
        st.warning(
            "Data tahunan tidak tersedia."
        )

    else:
        tahun_awal = int(
            yearly_total["year"].min()
        )

        tahun_akhir = int(
            yearly_total["year"].max()
        )

        fig1, ax1 = plt.subplots(
            figsize=(6.8, 3.2),
            dpi=120,
        )

        sns.lineplot(
            data=yearly_total,
            x="year",
            y="nilai_plot",
            marker="o",
            linewidth=2.3,
            ax=ax1,
        )

        ax1.set_title(
            f"Tren Total Penumpang Trans Jogja "
            f"Periode {tahun_awal}–{tahun_akhir}",
            fontsize=10,
            pad=8,
        )

        ax1.set_xlabel(
            "Tahun",
            fontsize=9,
        )

        ax1.set_ylabel(
            f"Jumlah Penumpang ({unit})",
            fontsize=9,
        )

        ax1.set_xticks(
            yearly_total["year"]
            .astype(int)
        )

        ax1.tick_params(
            axis="both",
            labelsize=8,
        )

        ax1.ticklabel_format(
            style="plain",
            axis="y",
        )

        max_plot = (
            yearly_total["nilai_plot"]
            .max()
        )

        if (
            pd.notna(max_plot)
            and max_plot > 0
        ):
            y_offset = max_plot * 0.015

        else:
            y_offset = 0

        for x, y in zip(
            yearly_total["year"],
            yearly_total["nilai_plot"],
        ):
            ax1.text(
                x,
                y + y_offset,
                f"{y:,.0f}".replace(
                    ",",
                    ".",
                ),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax1.margins(
            y=0.15
        )

        plt.tight_layout()

        col_chart, col_empty = st.columns(
            [0.8, 0.2]
        )

        with col_chart:
            st.pyplot(
                fig1,
                use_container_width=True,
            )

        plt.close(fig1)

    st.divider()

    # =========================
    # VALIDASI FILTER JALUR
    # =========================

    if len(selected_route_viz) == 0:
        st.warning(
            "Silakan pilih minimal satu jalur "
            "untuk menampilkan visualisasi "
            "berdasarkan jalur."
        )
        return

    # =========================
    # 2. PENUMPANG PER JALUR
    #    SETIAP BULAN
    # =========================

    st.subheader(
        f"📊 Jumlah Penumpang per Jalur "
        f"Setiap Bulan Tahun {selected_year}"
    )

    group_df = filtered_df.copy()

    group_df["month"] = pd.Categorical(
        group_df["month"],
        categories=month_cols,
        ordered=True,
    )

    current_route = [
        route
        for route in route_order_viz
        if (
            route in selected_route_viz
            and route
            in group_df["route"].unique()
        )
    ]

    group_df["route"] = pd.Categorical(
        group_df["route"],
        categories=current_route,
        ordered=True,
    )

    group_df = (
        group_df
        .dropna(
            subset=["route"]
        )
        .sort_values(
            ["month", "route"]
        )
    )

    group_df["nilai_plot"] = (
        group_df["passengers"]
        / scale
    )

    if group_df.empty:
        st.warning(
            f"Data jalur yang dipilih tidak "
            f"tersedia pada tahun {selected_year}."
        )

    else:
        fig2, ax2 = plt.subplots(
            figsize=(13, 6),
            dpi=120,
        )

        sns.barplot(
            data=group_df,
            x="month",
            y="nilai_plot",
            hue="route",
            hue_order=current_route,
            errorbar=None,
            ax=ax2,
        )

        ax2.set_title(
            f"Jumlah Penumpang per Jalur "
            f"Setiap Bulan Tahun {selected_year}",
            fontsize=10,
            pad=8,
        )

        ax2.set_xlabel(
            "Bulan",
            fontsize=9,
        )

        ax2.set_ylabel(
            f"Jumlah Penumpang ({unit})",
            fontsize=9,
        )

        ax2.tick_params(
            axis="both",
            labelsize=8,
        )

        ax2.legend(
            title="Jalur",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True,
        )

        plt.close(fig2)

    st.divider()

    # =========================
    # 3. PERINGKAT JALUR
    # =========================

    st.subheader(
        f"📊 Peringkat Jalur Berdasarkan "
        f"Jumlah Penumpang Tahun {selected_year}"
    )

    all_routes_selected = (
        set(selected_route_viz)
        == set(route_available)
    )

    if (
        all_routes_selected
        and len(route_available) > 0
    ):

        ranking_df = (
            df_long_viz[
                df_long_viz["route"]
                .isin(route_available)
            ]
            .groupby(
                "route",
                as_index=False,
            )["passengers"]
            .sum()
        )

        col1, col2 = st.columns(2)

        # =========================
        # TOP 5 TERTINGGI
        # =========================

        with col1:
            st.markdown(
                "#### 📈 5 Jalur Tertinggi"
            )

            max5 = (
                ranking_df
                .sort_values(
                    "passengers",
                    ascending=False,
                )
                .head(5)
                .copy()
            )

            max5["nilai_plot"] = (
                max5["passengers"]
                / scale
            )

            fig3, ax3 = plt.subplots(
                figsize=(6, 3.5)
            )

            palette_max = (
                sns.color_palette(
                    "Blues_r",
                    n_colors=max(
                        len(max5),
                        1,
                    ),
                )
            )

            sns.barplot(
                data=max5,
                y="route",
                x="nilai_plot",
                hue="route",
                palette=palette_max,
                legend=False,
                ax=ax3,
            )

            for i, value in enumerate(
                max5["nilai_plot"]
            ):
                ax3.text(
                    value,
                    i,
                    f" {value:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    va="center",
                    fontsize=8,
                )

            if not max5.empty:
                max_value = (
                    max5["nilai_plot"]
                    .max()
                )

                if max_value > 0:
                    ax3.set_xlim(
                        0,
                        max_value * 1.12,
                    )

            ax3.set_title(
                f"Top 5 Jalur Tertinggi "
                f"Tahun {selected_year}",
                fontsize=11,
            )

            ax3.set_xlabel(
                f"Jumlah Penumpang ({unit})",
                fontsize=9,
            )

            ax3.set_ylabel(
                "Jalur",
                fontsize=9,
            )

            ax3.tick_params(
                axis="both",
                labelsize=8,
            )

            ax3.ticklabel_format(
                style="plain",
                axis="x",
            )

            plt.tight_layout()

            st.pyplot(
                fig3,
                use_container_width=True,
            )

            plt.close(fig3)

        # =========================
        # TOP 5 TERENDAH
        # =========================

        with col2:
            st.markdown(
                "#### 📉 5 Jalur Terendah"
            )

            min5 = (
                ranking_df
                .sort_values(
                    "passengers",
                    ascending=True,
                )
                .head(5)
                .copy()
            )

            min5["nilai_plot"] = (
                min5["passengers"]
                / scale
            )

            fig4, ax4 = plt.subplots(
                figsize=(6, 3.5)
            )

            palette_min = (
                sns.color_palette(
                    "Blues",
                    n_colors=max(
                        len(min5),
                        1,
                    ),
                )
            )

            sns.barplot(
                data=min5,
                y="route",
                x="nilai_plot",
                hue="route",
                palette=palette_min,
                legend=False,
                ax=ax4,
            )

            for i, value in enumerate(
                min5["nilai_plot"]
            ):
                ax4.text(
                    value,
                    i,
                    f" {value:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    va="center",
                    fontsize=8,
                )

            if not min5.empty:
                max_value = (
                    min5["nilai_plot"]
                    .max()
                )

                if max_value > 0:
                    ax4.set_xlim(
                        0,
                        max_value * 1.12,
                    )

            ax4.set_title(
                f"Top 5 Jalur Terendah "
                f"Tahun {selected_year}",
                fontsize=11,
            )

            ax4.set_xlabel(
                f"Jumlah Penumpang ({unit})",
                fontsize=9,
            )

            ax4.set_ylabel(
                "Jalur",
                fontsize=9,
            )

            ax4.tick_params(
                axis="both",
                labelsize=8,
            )

            ax4.ticklabel_format(
                style="plain",
                axis="x",
            )

            plt.tight_layout()

            st.pyplot(
                fig4,
                use_container_width=True,
            )

            plt.close(fig4)

    else:
        st.warning(
            "Grafik **Peringkat Jalur Tertinggi "
            "dan Terendah** hanya ditampilkan jika "
            "semua jalur dipilih pada filter jalur."
        )

    st.divider()

    # =========================
    # 4. TOTAL PER BULAN
    # =========================

    st.subheader(
        f"📊 Total Penumpang per Bulan "
        f"Tahun {selected_year}"
    )

    monthly_total = (
        filtered_df
        .groupby(
            "month",
            as_index=False,
        )["passengers"]
        .sum()
    )

    monthly_total["month"] = pd.Categorical(
        monthly_total["month"],
        categories=month_cols,
        ordered=True,
    )

    monthly_total = (
        monthly_total
        .sort_values("month")
    )

    monthly_total["nilai_plot"] = (
        monthly_total["passengers"]
        / scale
    )

    fig5, ax5 = plt.subplots(
        figsize=(10, 5)
    )

    sns.lineplot(
        data=monthly_total,
        x="month",
        y="nilai_plot",
        marker="o",
        ax=ax5,
    )

    ax5.set_title(
        f"Total Penumpang per Bulan "
        f"Tahun {selected_year}"
    )

    ax5.set_xlabel(
        "Bulan"
    )

    ax5.set_ylabel(
        f"Jumlah Penumpang ({unit})"
    )

    ax5.ticklabel_format(
        style="plain",
        axis="y",
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig5,
        use_container_width=True,
    )

    plt.close(fig5)

    st.divider()

    # =========================
    # 5. TOTAL PER JALUR
    # =========================

    st.subheader(
        f"📈 Perbandingan Jumlah Penumpang "
        f"per Jalur Tahun {selected_year}"
    )

    route_total = (
        filtered_df
        .groupby(
            "route",
            as_index=False,
        )["passengers"]
        .sum()
    )

    current_route_total = [
        route
        for route in route_order_viz
        if route
        in route_total["route"].unique()
    ]

    route_total["route"] = pd.Categorical(
        route_total["route"],
        categories=current_route_total,
        ordered=True,
    )

    route_total = (
        route_total
        .dropna(
            subset=["route"]
        )
        .sort_values("route")
    )

    route_total["nilai_plot"] = (
        route_total["passengers"]
        / scale
    )

    fig6, ax6 = plt.subplots(
        figsize=(12, 5)
    )

    sns.barplot(
        data=route_total,
        x="route",
        y="nilai_plot",
        errorbar=None,
        ax=ax6,
    )

    ax6.set_title(
        f"Total Penumpang per Jalur "
        f"Tahun {selected_year}"
    )

    ax6.set_xlabel(
        "Jalur"
    )

    ax6.set_ylabel(
        f"Jumlah Penumpang ({unit})"
    )

    ax6.ticklabel_format(
        style="plain",
        axis="y",
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig6,
        use_container_width=True,
    )

    plt.close(fig6)

    st.divider()

    # =========================
    # 6. HEATMAP
    # =========================

    st.subheader(
        f"🔥 Heatmap Penumpang Tahun "
        f"{selected_year} (Jalur vs Bulan)"
    )

    pivot_table = (
        filtered_df
        .pivot_table(
            index="route",
            columns="month",
            values="passengers",
            aggfunc="sum",
            observed=False,
        )
    )

    current_heatmap_routes = [
        route
        for route in route_order_viz
        if route
        in filtered_df["route"].unique()
    ]

    pivot_table = (
        pivot_table
        .reindex(
            index=current_heatmap_routes,
            columns=month_cols,
        )
    )

    if pivot_table.empty:
        st.warning(
            "Data tidak tersedia "
            "untuk membuat heatmap."
        )

    else:
        fig7, ax7 = plt.subplots(
            figsize=(12, 6)
        )

        sns.heatmap(
            pivot_table,
            cmap="YlOrRd",
            annot=False,
            ax=ax7,
        )

        ax7.set_title(
            f"Heatmap Jumlah Penumpang "
            f"Tahun {selected_year}"
        )

        ax7.set_xlabel(
            "Bulan"
        )

        ax7.set_ylabel(
            "Jalur"
        )

        plt.tight_layout()

        st.pyplot(
            fig7,
            use_container_width=True,
        )

        plt.close(fig7)
