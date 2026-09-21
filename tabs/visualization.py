import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


def render_visualization_tab(
    df_all_year,
    month_cols,
    route_order_viz,
    enable_visualization_tab,
):
    if not enable_visualization_tab:
        st.info("Tab visualisasi sementara dinonaktifkan.")
        return

    # Judul dan deskripsi
    st.subheader("Visualisasi Data Historis Jumlah Penumpang")

    st.write(
        "Pada tab ini, pengguna dapat melihat pola historis jumlah penumpang "
        "Trans Jogja berdasarkan tahun, jalur, dan skala jumlah penumpang."
    )

    route_info = pd.DataFrame(
        {
            "Jalur": [
                "1A", "1B", "2A", "2B", "3A", "3B",
                "4A", "4B", "5A", "5B", "6", "8",
                "9", "10", "11", "12", "13", "14", "15",
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
                "Malioboro - Palbapang",
            ],
        }
    )

    with st.expander("Daftar Jalur dan Rute Trans Jogja"):
        st.dataframe(
            route_info,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Filter visualisasi
    st.markdown("### Filter Visualisasi")

    year_list = sorted(
        df_all_year["year"].dropna().astype(int).unique()
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
                "Nilai Aktual",
                "Ribu",
                "Juta",
            ],
            index=1,
            key="viz_unit",
        )

    scale_map = {
        "Nilai Aktual": 1,
        "Ribu": 1_000,
        "Juta": 1_000_000,
    }
    scale = scale_map[unit]

    # Label skala untuk grafik
    scale_label = "" if unit == "Nilai Aktual" else f" ({unit})"

    # Data sesuai tahun terpilih
    df_selected_year = df_all_year[
        df_all_year["year"] == selected_year
    ].copy()

    # Jalur yang tersedia pada tahun terpilih
    route_available = [
        route
        for route in route_order_viz
        if route in df_selected_year["route"].unique()
    ]

    st.divider()

    # Persiapan data
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
        subset=["year", "route", "month", "passengers"]
    ).copy()

    yearly_long = yearly_long.dropna(
        subset=["year", "route", "month", "passengers"]
    ).copy()

    df_long_viz["route"] = df_long_viz["route"].astype(str)
    yearly_long["route"] = yearly_long["route"].astype(str)

    # Total tahunan
    yearly_total = (
        yearly_long
        .groupby("year", as_index=False)["passengers"]
        .sum()
        .sort_values("year")
    )

    yearly_total["nilai_plot"] = yearly_total["passengers"] / scale

    # Format angka
    def format_number(value, unit):
        if unit == "Juta":
            return f"{value:.2f}".replace(".", ",")
        return f"{value:,.0f}".replace(",", ".")

    # Label bulan untuk visualisasi
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
        "Jul", "Agu", "Sep", "Okt", "Nov", "Des",
    ]

# VISUALISASI DATA
    # 1. Tren tahunan
    st.subheader("Tren Total Penumpang dan Pertumbuhan per Tahun")

    if yearly_total.empty:
        st.warning("Data tahunan tidak tersedia.")

    else:
        tahun_awal = int(yearly_total["year"].min())
        tahun_akhir = int(yearly_total["year"].max())

        plot_data = yearly_total.reset_index(drop=True).copy()
        plot_data["growth"] = plot_data["passengers"].pct_change() * 100

        x = range(len(plot_data))
        fig1, ax1 = plt.subplots(figsize=(6.8, 3.5), dpi=120)

        bars = ax1.bar(
            x,
            plot_data["nilai_plot"],
            width=0.6,
            zorder=3,
        )

        bar_labels = [
            format_number(value, unit)
            for value in plot_data["nilai_plot"]
        ]

        ax1.bar_label(
            bars,
            labels=bar_labels,
            padding=4,
            fontsize=8,
        )

        max_value = plot_data["nilai_plot"].max()

        # Pertumbuhan antar tahun
        growth_color = "#F28E2B"

        bar_centers = [
            bar.get_x() + bar.get_width() / 2
            for bar in bars
        ]
        bar_heights = [
            bar.get_height()
            for bar in bars
        ]
        bar_width = bars[0].get_width()

        bracket_gap = max_value * 0.17
        arrow_gap = max_value * 0.090
        x_gap = bar_width * 0.025

        for i in range(1, len(plot_data)):
            growth = plot_data.loc[i, "growth"]

            prev_height = bar_heights[i - 1]
            curr_height = bar_heights[i]

            left_x = bar_centers[i - 1] + x_gap
            right_x = bar_centers[i] - x_gap
            bracket_y = max(prev_height, curr_height) + bracket_gap

            # Garis bracket
            ax1.plot(
                [left_x, left_x, right_x],
                [prev_height + arrow_gap, bracket_y, bracket_y],
                color=growth_color,
                linewidth=1.5,
                zorder=4,
            )

            # Panah menuju tahun berikutnya
            ax1.annotate(
                "",
                xy=(right_x, curr_height + arrow_gap),
                xytext=(right_x, bracket_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=growth_color,
                    linewidth=1.5,
                ),
            )

            # Label pertumbuhan
            ax1.text(
                (left_x + right_x) / 2,
                bracket_y + max_value * 0.015,
                f"{growth:.2f}%".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax1.set_title(
            f"Total Penumpang dan Pertumbuhan per Tahun "
            f"(Periode {tahun_awal}–{tahun_akhir})",
            fontsize=10,
        )
        ax1.set_xlabel("Tahun", fontsize=9)
        ax1.set_ylabel(f"Total Penumpang{scale_label}", fontsize=9)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(plot_data["year"].astype(int))
        ax1.set_ylim(0, max_value * 1.30)

        ax1.yaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: format_number(x, unit)
            )
        )

        ax1.grid(axis="x", visible=False)

        legend_items = [
            Patch(
                facecolor=bars[0].get_facecolor(),
                label=f"Total Penumpang{scale_label}",
            ),
            Line2D(
                [0],
                [0],
                color=growth_color,
                linewidth=1.5,
                marker=">",
                label="Pertumbuhan (%)",
            ),
        ]

        ax1.legend(
            handles=legend_items,
            loc="upper left",
            fontsize=8,
            frameon=True,
        )

        plt.tight_layout()

        col_chart, col_empty = st.columns([0.8, 0.2])

        with col_chart:
            st.pyplot(fig1, use_container_width=True)

        plt.close(fig1)

    st.divider()

    # 2. Total per bulan
    st.subheader(
        f"Total Penumpang per Bulan "
        f"Tahun {selected_year}"
    )

    monthly_total = (
        df_long_viz
        .groupby("month", as_index=False)["passengers"]
        .sum()
    )

    monthly_total["month"] = pd.Categorical(
        monthly_total["month"],
        categories=month_cols,
        ordered=True,
    )

    monthly_total = monthly_total.sort_values("month")
    monthly_total["nilai_plot"] = monthly_total["passengers"] / scale

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=monthly_total,
        x="month",
        y="nilai_plot",
        marker="o",
        ax=ax2,
    )

    ax2.set_title(
        f"Total Penumpang per Bulan "
        f"Tahun {selected_year}"
    )
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel(f"Total Penumpang{scale_label}")

    ax2.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: format_number(x, unit)
        )
    )

    ax2.set_xticklabels(month_labels, rotation=0)
    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=True,
    )

    plt.close(fig2)

    st.divider()

    # 3. Total per jalur
    st.subheader(
        f"Total Penumpang per Jalur "
        f"Tahun {selected_year}"
    )

    route_total = (
        df_long_viz
        .groupby("route", as_index=False)["passengers"]
        .sum()
    )

    current_route_total = [
        route
        for route in route_order_viz
        if route in route_total["route"].unique()
    ]

    route_total["route"] = pd.Categorical(
        route_total["route"],
        categories=current_route_total,
        ordered=True,
    )

    route_total = (
        route_total
        .dropna(subset=["route"])
        .sort_values("route")
    )

    route_total["nilai_plot"] = route_total["passengers"] / scale

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    sns.barplot(
        data=route_total,
        x="route",
        y="nilai_plot",
        errorbar=None,
        ax=ax3,
    )

    ax3.set_title(
        f"Total Penumpang per Jalur "
        f"Tahun {selected_year}"
    )
    ax3.set_xlabel("Jalur")
    ax3.set_ylabel(f"Total Penumpang{scale_label}")

    ax3.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: format_number(x, unit)
        )
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(
        fig3,
        use_container_width=True,
    )

    plt.close(fig3)

    st.divider()

    # 4. Peringkat jalur
    st.subheader(
        f"Peringkat Jalur Berdasarkan "
        f"Total Penumpang Tahun {selected_year}"
    )

    ranking_df = (
        df_long_viz[
            df_long_viz["route"].isin(route_available)
        ]
        .groupby("route", as_index=False)["passengers"]
        .sum()
    )

    col1, col2 = st.columns(2)

    # Top 5 tertinggi
    with col1:
        st.markdown("#### 5 Jalur Tertinggi")

        max5 = (
            ranking_df
            .sort_values(
                "passengers",
                ascending=False,
            )
            .head(5)
            .copy()
        )

        max5["nilai_plot"] = max5["passengers"] / scale

        fig4, ax4 = plt.subplots(figsize=(6, 3.5))

        palette_max = sns.color_palette(
            "Blues_r",
            n_colors=max(len(max5), 1),
        )

        sns.barplot(
            data=max5,
            y="route",
            x="nilai_plot",
            hue="route",
            palette=palette_max,
            legend=False,
            ax=ax4,
        )

        for i, value in enumerate(max5["nilai_plot"]):
            ax4.text(
                value,
                i,
                f" {format_number(value, unit)}",
                va="center",
                fontsize=8,
            )

        if not max5.empty:
            max_value = max5["nilai_plot"].max()

            if max_value > 0:
                ax4.set_xlim(
                    0,
                    max_value * 1.12,
                )

        ax4.set_title(
            f"Top 5 Jalur Tertinggi "
            f"Tahun {selected_year}",
            fontsize=11,
        )
        ax4.set_xlabel(
            f"Total Penumpang{scale_label}",
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

        ax4.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: format_number(x, unit)
            )
        )

        plt.tight_layout()

        st.pyplot(
            fig4,
            use_container_width=True,
        )

        plt.close(fig4)

    # Top 5 terendah
    with col2:
        st.markdown("#### 5 Jalur Terendah")

        min5 = (
            ranking_df
            .sort_values(
                "passengers",
                ascending=True,
            )
            .head(5)
            .copy()
        )

        min5["nilai_plot"] = min5["passengers"] / scale

        fig5, ax5 = plt.subplots(figsize=(6, 3.5))

        palette_min = sns.color_palette(
            "Blues",
            n_colors=max(len(min5), 1),
        )

        sns.barplot(
            data=min5,
            y="route",
            x="nilai_plot",
            hue="route",
            palette=palette_min,
            legend=False,
            ax=ax5,
        )

        for i, value in enumerate(min5["nilai_plot"]):
            ax5.text(
                value,
                i,
                f" {format_number(value, unit)}",
                va="center",
                fontsize=8,
            )

        if not min5.empty:
            max_value = min5["nilai_plot"].max()

            if max_value > 0:
                ax5.set_xlim(
                    0,
                    max_value * 1.12,
                )

        ax5.set_title(
            f"Top 5 Jalur Terendah "
            f"Tahun {selected_year}",
            fontsize=11,
        )
        ax5.set_xlabel(
            f"Total Penumpang{scale_label}",
            fontsize=9,
        )
        ax5.set_ylabel(
            "Jalur",
            fontsize=9,
        )
        ax5.tick_params(
            axis="both",
            labelsize=8,
        )

        ax5.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: format_number(x, unit)
            )
        )

        plt.tight_layout()

        st.pyplot(
            fig5,
            use_container_width=True,
        )

        plt.close(fig5)

    st.divider()

    # 5. Penumpang per jalur setiap bulan
    st.subheader(
        f"Jumlah Penumpang per Jalur "
        f"Setiap Bulan Tahun {selected_year}"
    )

    # Default 3 jalur dengan total penumpang tertinggi
    default_routes = max5["route"].head(3).tolist()

    selected_route_viz = st.multiselect(
        "Pilih Jalur",
        options=route_available,
        default=default_routes,
        placeholder="Pilih satu atau beberapa jalur",
        key="viz_selected_routes",
    )

    filtered_df = df_long_viz[
        df_long_viz["route"].isin(selected_route_viz)
    ].copy()

    if len(selected_route_viz) == 0:
        st.info(
            "Pilih minimal 1 jalur untuk menampilkan jumlah penumpang per bulan."
        )

    elif len(selected_route_viz) > 5:
        st.info(
            "Pilih maksimal 5 jalur agar grafik tidak terlalu padat "
            "dan perbandingan antarjalur setiap bulan tetap mudah dibaca."
        )

    else:
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
                and route in group_df["route"].unique()
            )
        ]

        group_df["route"] = pd.Categorical(
            group_df["route"],
            categories=current_route,
            ordered=True,
        )

        group_df = (
            group_df
            .dropna(subset=["route"])
            .sort_values(["month", "route"])
        )

        group_df["nilai_plot"] = group_df["passengers"] / scale

        fig6, ax6 = plt.subplots(
            figsize=(13, 6),
            dpi=120,
        )

        sns.lineplot(
            data=group_df,
            x="month",
            y="nilai_plot",
            hue="route",
            hue_order=current_route,
            marker="o",
            ax=ax6,
        )

        ax6.set_title(
            f"Jumlah Penumpang per Jalur "
            f"Setiap Bulan Tahun {selected_year}",
            fontsize=10,
            pad=8,
        )

        ax6.set_xlabel(
            "Bulan",
            fontsize=9,
        )

        ax6.set_ylabel(
            f"Jumlah Penumpang{scale_label}",
            fontsize=9,
        )

        ax6.yaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: format_number(x, unit)
            )
        )

        ax6.tick_params(
            axis="both",
            labelsize=8,
        )

        ax6.legend(
            title="Jalur",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
        )

        ax6.set_xticklabels(month_labels, rotation=0)

        plt.tight_layout()

        st.pyplot(
            fig6,
            use_container_width=True,
        )

        plt.close(fig6)