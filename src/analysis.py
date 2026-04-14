from __future__ import annotations

from calendar import month_abbr
import math
from datetime import datetime
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from src.dashboard_contracts import (
    build_top_trends_calendar,
    get_dashboard_layout_contract,
    select_reliable_platform_leader,
)
from src.preprocessing import build_analysis_assets, load_dashboard_assets


st.set_page_config(
    page_title="Social Media Trends Analyzer",
    page_icon="SM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

sns.set_theme(style="whitegrid")

PLATFORM_DISPLAY_MAP = {
    "Tiktok": "TikTok",
    "Youtube": "YouTube",
}

DARK_FIGURE = "#0a1020"
DARK_AXES = "#0f172a"
DARK_GRID = "#2a3550"
TEXT_PRIMARY = "#f5f7ff"
TEXT_MUTED = "#9caed3"
PANEL_BORDER = "#24314f"
CONTINUITY_START = "2017-01"
CONTINUITY_END = "2023-12"
FULL_TIMELINE_VIEW = "Full timeline view"
SOURCE_MONTHS_ONLY = "Source months only"
DASHBOARD_LAYOUT_CONTRACT = get_dashboard_layout_contract()


@st.cache_data(show_spinner=False)
def get_dashboard_assets() -> dict:
    return load_dashboard_assets()


def rebuild_dashboard_assets() -> dict:
    build_analysis_assets(verbose=True)
    get_dashboard_assets.clear()
    return get_dashboard_assets()


def normalize_display_labels(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "platform" in normalized.columns:
        normalized["platform"] = normalized["platform"].replace(PLATFORM_DISPLAY_MAP)
    return normalized


def apply_filters(
    df: pd.DataFrame,
    datasets: list[str],
    platforms: list[str],
    record_types: list[str],
    regions: list[str],
    sentiments: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    for column, selected in [
        ("source_dataset", datasets),
        ("platform", platforms),
        ("record_type", record_types),
        ("region", regions),
        ("sentiment", sentiments),
    ]:
        if selected:
            if column not in filtered.columns:
                continue
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator)


def weighted_average(numerator: pd.Series, denominator: pd.Series) -> float:
    denominator_sum = denominator.sum()
    if denominator_sum == 0:
        return 0.0
    return float(numerator.sum() / denominator_sum)


def format_metric(value: float | int, precision: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "0"
    absolute = abs(float(value))
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B{suffix}"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M{suffix}"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K{suffix}"
    if suffix == "%":
        return f"{value:.2f}{suffix}"
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.{precision}f}{suffix}"


def compact_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _: format_metric(value))


def percent_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _: f"{value:.1f}%")


def build_palette(values: list[str]) -> dict[str, str]:
    colors = [
        "#52d1ff",
        "#7c5cff",
        "#1ce7b8",
        "#ff7b72",
        "#ffb84d",
        "#5eead4",
        "#c084fc",
        "#38bdf8",
    ]
    return {value: colors[index % len(colors)] for index, value in enumerate(values)}


def scale_marker_sizes(values: pd.Series, minimum: float = 170, maximum: float = 360) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    if numeric.empty:
        return numeric
    rooted = numeric.add(1).map(math.log10)
    low = float(rooted.min())
    high = float(rooted.max())
    if high == low:
        fallback = minimum if high == 0 else (minimum + maximum) / 2
        return pd.Series([fallback] * len(numeric), index=numeric.index)
    normalized = (rooted - low) / (high - low)
    return minimum + normalized * (maximum - minimum)


def donut_autopct(threshold: float = 4.0):
    def _formatter(value: float) -> str:
        return f"{value:.1f}%" if value >= threshold else ""

    return _formatter


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=11)
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color(PANEL_BORDER)
    ax.grid(color=DARK_GRID, alpha=0.28)


def style_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor("#111b30")
    legend.get_frame().set_edgecolor(PANEL_BORDER)
    legend.get_frame().set_alpha(0.9)
    legend.get_title().set_color(TEXT_PRIMARY)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)


def create_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_FIGURE)
    return fig, ax


def render_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def format_topic_label(topic: str, limit: int | None = None) -> str:
    if topic is None or pd.isna(topic):
        cleaned = "Unknown"
    else:
        normalized = unicodedata.normalize("NFKD", str(topic))
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = " ".join(ascii_only.replace("_", " ").replace("#", " ").split()).title()
        if not cleaned:
            cleaned = "Topic"
    if limit is not None and len(cleaned) > limit:
        return cleaned[: max(limit - 3, 1)].rstrip() + "..."
    return cleaned


def build_dataset_display_map(catalog: dict) -> dict[str, str]:
    explicit_map = {
        "viral_social_media_trends": "Viral Social Trends",
        "indian_youtube_trending": "India YouTube API Trending",
        "tweet_sentiment_selection": "Tweet Sentiment Selection",
        "twitter_train_binary_sentiment": "Twitter Sentiment Training",
        "youtube_comments_sentiment": "YouTube Comment Sentiment",
        "youtube_publish_country_snapshot": "Cross-country YouTube Snapshot",
    }
    region_short_map = {
        "United States": "US",
        "United Kingdom": "UK",
    }

    display_map: dict[str, str] = {}
    for dataset in catalog.get("datasets", []):
        name = dataset.get("name", "")
        title = dataset.get("title", name)
        if name in explicit_map:
            display_map[name] = explicit_map[name]
        elif name.startswith("youtube_trending_") and title.endswith(" YouTube Trending Videos"):
            region = title.replace(" YouTube Trending Videos", "")
            region_label = region_short_map.get(region, region)
            display_map[name] = f"{region_label} YouTube Trending"
        else:
            display_map[name] = title
    return display_map


def display_dataset_name(name: str, dataset_display_map: dict[str, str], limit: int | None = None) -> str:
    label = dataset_display_map.get(name, " ".join(str(name).replace("_", " ").split()).title())
    if limit is not None and len(label) > limit:
        return label[: max(limit - 3, 1)].rstrip() + "..."
    return label


def add_bar_end_labels(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    max_value = max(values) if max(values) > 0 else 1
    for patch, value in zip(ax.patches, values):
        ax.text(
            value + max_value * 0.018,
            patch.get_y() + patch.get_height() / 2,
            format_metric(value),
            va="center",
            ha="left",
            color=TEXT_PRIMARY,
            fontsize=9.5,
            fontweight="semibold",
        )


def build_month_frame(start: str = CONTINUITY_START, end: str = CONTINUITY_END) -> pd.DataFrame:
    periods = pd.period_range(start=start, end=end, freq="M")
    return pd.DataFrame(
        {
            "period": periods,
            "year_month": periods.astype(str),
            "year": periods.year.astype(int),
            "month_num": periods.month.astype(int),
        }
    )


def smooth_month_profile(month_profile: pd.Series) -> pd.Series:
    profile = pd.Series(index=range(1, 13), dtype=float)
    for month_num, value in month_profile.items():
        profile.loc[int(month_num)] = float(value)

    if profile.notna().sum() == 0:
        return pd.Series(1.0, index=range(1, 13))

    if profile.notna().sum() == 1:
        return pd.Series(1.0, index=range(1, 13))

    for month_num in profile.index[profile.isna()]:
        prior_months = [month for month in range(month_num - 1, 0, -1) if pd.notna(profile.get(month))]
        next_months = [month for month in range(month_num + 1, 13) if pd.notna(profile.get(month))]
        if prior_months and next_months:
            profile.loc[month_num] = (float(profile.loc[prior_months[0]]) + float(profile.loc[next_months[0]])) / 2
        elif prior_months:
            profile.loc[month_num] = float(profile.loc[prior_months[0]])
        elif next_months:
            profile.loc[month_num] = float(profile.loc[next_months[0]])

    profile = profile.ffill().bfill()
    if profile.mean() <= 0:
        return pd.Series(1.0, index=range(1, 13))

    wrapped_values = profile.iloc[-2:].tolist() + profile.tolist() + profile.iloc[:2].tolist()
    smoothed = pd.Series(wrapped_values).rolling(window=5, center=True, min_periods=1).mean().iloc[2:-2]
    smoothed.index = range(1, 13)
    if smoothed.mean() <= 0:
        return pd.Series(1.0, index=range(1, 13))
    return smoothed / smoothed.mean()


def build_dynamic_wave(
    years: pd.Series,
    months: pd.Series,
    series_key: str,
) -> pd.Series:
    phase_seed = sum(ord(character) for character in str(series_key))
    phase = math.radians(phase_seed % 360)
    dynamic_values: list[float] = []

    for year, month_num in zip(years.astype(int), months.astype(int)):
        month_angle = ((month_num - 1) / 12) * (2 * math.pi)
        year_offset = year - int(years.min())
        quarter_wave = 1 + 0.18 * math.sin(month_angle + phase) + 0.09 * math.cos((2 * month_angle) - phase / 2)
        year_wave = 1 + 0.07 * math.sin((year_offset + 1) * 0.95 + phase / 3 + month_num / 7)
        pulse_wave = 1 + 0.05 * math.cos((month_num / 2.2) + phase + year_offset * 0.35)
        dynamic_values.append(max(0.72, quarter_wave * year_wave * pulse_wave))

    return pd.Series(dynamic_values, index=years.index, dtype=float)


def apply_quarter_ticks(ax: plt.Axes, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(start_date, end_date)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")


def build_continuity_timeline(
    monthly_frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    start: str = CONTINUITY_START,
    end: str = CONTINUITY_END,
) -> pd.DataFrame:
    if monthly_frame.empty:
        return pd.DataFrame(columns=[group_column, "year_month", "year", "month_num", value_column, "value_origin"])

    month_frame = build_month_frame(start=start, end=end)
    output_frames: list[pd.DataFrame] = []

    for group_value, group_df in monthly_frame.groupby(group_column, sort=False):
        base = month_frame.copy()
        series = (
            group_df.assign(period=lambda frame: pd.PeriodIndex(frame["year_month"], freq="M"))
            .groupby("period", as_index=True)[value_column]
            .sum()
            .sort_index()
        )
        base = base.merge(series.rename("observed_value"), left_on="period", right_index=True, how="left")
        base["value_origin"] = base["observed_value"].notna().map({True: "Observed", False: "Modeled"})

        if base["observed_value"].notna().sum() == 0:
            continue

        observed_month_profile = (
            base.loc[base["observed_value"].notna(), ["month_num", "observed_value"]]
            .groupby("month_num")["observed_value"]
            .mean()
        )
        month_weights = smooth_month_profile(observed_month_profile)

        observed_year_profile = (
            base.loc[base["observed_value"].notna(), ["year", "observed_value"]]
            .groupby("year")["observed_value"]
            .median()
            .sort_index()
        )
        year_index = pd.Index(range(int(month_frame["year"].min()), int(month_frame["year"].max()) + 1), dtype=int)
        year_baseline = pd.Series(index=year_index, dtype=float)
        for year, value in observed_year_profile.items():
            year_baseline.loc[int(year)] = float(value)
        year_baseline = year_baseline.interpolate(limit_direction="both").ffill().bfill()
        if year_baseline.isna().all():
            year_baseline = pd.Series(float(base["observed_value"].dropna().median()), index=year_index)

        dynamic_wave = build_dynamic_wave(base["year"], base["month_num"], str(group_value))
        dynamic_wave = dynamic_wave.groupby(base["year"]).transform(
            lambda values: values / values.mean() if values.mean() else values
        )

        base["modeled_value"] = (
            base["year"].map(year_baseline).astype(float)
            * base["month_num"].map(month_weights).astype(float)
            * dynamic_wave.astype(float)
        )
        base[value_column] = base["observed_value"].fillna(base["modeled_value"]).fillna(0).clip(lower=0)

        base[group_column] = group_value
        output_frames.append(base[[group_column, "year_month", "year", "month_num", value_column, "value_origin"]])

    if not output_frames:
        return pd.DataFrame(columns=[group_column, "year_month", "year", "month_num", value_column, "value_origin"])

    return pd.concat(output_frames, ignore_index=True)


def prepare_yearly_month_view(monthly_frame: pd.DataFrame, metric_key: str, coverage_mode: str) -> pd.DataFrame:
    if monthly_frame.empty:
        return pd.DataFrame(columns=["year", "month_num", metric_key, "value_origin"])

    if coverage_mode == FULL_TIMELINE_VIEW:
        total_monthly = monthly_frame.groupby("year_month", as_index=False)[metric_key].sum()
        total_monthly["series_name"] = "overall"
        continuity = build_continuity_timeline(total_monthly, "series_name", metric_key)
        continuity["year"] = continuity["year"].astype(str)
        return continuity[["year", "month_num", metric_key, "value_origin"]]

    year_view = monthly_frame.copy()
    year_view["period"] = pd.to_datetime(year_view["year_month"] + "-01", errors="coerce")
    year_view = year_view.dropna(subset=["period"])
    if year_view.empty:
        return pd.DataFrame(columns=["year", "month_num", metric_key, "value_origin"])

    year_view["year"] = year_view["period"].dt.year.astype(int)
    year_view["month_num"] = year_view["period"].dt.month.astype(int)
    summary = year_view.groupby(["year", "month_num"], as_index=False)[metric_key].sum()
    summary["year"] = summary["year"].astype(str)
    summary["value_origin"] = "Observed"
    return summary


def build_topic_boom_calendar(
    topic_timeline_metrics: pd.DataFrame,
    metric_key: str,
    start: str = CONTINUITY_START,
    end: str = CONTINUITY_END,
) -> pd.DataFrame:
    ranked_trends = build_top_trends_calendar(
        topic_timeline_metrics,
        trend_lens=metric_key,
        top_n=3,
        fill_missing_months=True,
        start=start,
        end=end,
    )
    if ranked_trends.empty:
        return ranked_trends

    valid_months = set(pd.period_range(start=start, end=end, freq="M").astype(str))
    return ranked_trends[ranked_trends["year_month"].isin(valid_months)].copy()


def render_topic_cloud(topic_rollup: pd.DataFrame, metric_key: str, metric_title: str) -> None:
    cloud_frame = topic_rollup.copy()
    cloud_frame = cloud_frame[cloud_frame[metric_key] > 0].sort_values(metric_key, ascending=False).head(28)
    if cloud_frame.empty:
        st.info("No weighted topic data is available for the current filter and ranking selection.")
        return

    frequencies = {
        format_topic_label(row["primary_topic"]): float(row[metric_key])
        for _, row in cloud_frame.iterrows()
    }
    palette = ["#52d1ff", "#7c5cff", "#1ce7b8", "#ff7b72", "#ffb84d", "#5eead4", "#c084fc", "#38bdf8"]

    def color_func(word: str, *args, **kwargs) -> str:
        return palette[sum(ord(character) for character in word) % len(palette)]

    cloud = WordCloud(
        width=1500,
        height=860,
        background_color=None,
        mode="RGBA",
        prefer_horizontal=0.95,
        collocations=False,
        max_words=min(28, len(frequencies)),
        min_font_size=18,
        margin=10,
        color_func=color_func,
    ).generate_from_frequencies(frequencies)

    fig, ax = create_figure((8.8, 5.3))
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title(
        f"Topic cloud weighted by {metric_title.lower()}",
        color=TEXT_PRIMARY,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_facecolor(DARK_AXES)
    ax.axis("off")
    fig.tight_layout()
    render_figure(fig)


def render_platform_timeline_cards(timeline_view: pd.DataFrame, metric_key: str, metric_label: str) -> None:
    platform_totals = (
        timeline_view.groupby("platform", as_index=False)[metric_key]
        .sum()
        .sort_values(metric_key, ascending=False)
    )
    platform_order = platform_totals["platform"].tolist()
    platform_palette = build_palette(platform_order)
    timeline_cols = st.columns(2)

    for index, platform in enumerate(platform_order):
        platform_frame = (
            timeline_view[timeline_view["platform"] == platform]
            .sort_values("year_month")
            .reset_index(drop=True)
        )
        positions = list(range(len(platform_frame)))
        values = platform_frame[metric_key].tolist()
        labels = platform_frame["year_month"].tolist()
        tick_step = max(1, len(labels) // 6)
        tick_positions = positions[::tick_step]
        if positions and positions[-1] not in tick_positions:
            tick_positions.append(positions[-1])

        display_name = PLATFORM_DISPLAY_MAP.get(platform, platform)
        color = platform_palette.get(platform, "#52d1ff")

        with timeline_cols[index % 2]:
            fig, ax = create_figure((6.4, 3.6))
            ax.plot(
                positions,
                values,
                color=color,
                linewidth=2.7,
                solid_capstyle="round",
            )
            ax.fill_between(positions, values, 0, color=color, alpha=0.14)
            ax.scatter(
                positions,
                values,
                s=42,
                color=DARK_AXES,
                edgecolors=color,
                linewidths=1.8,
                zorder=3,
            )
            style_axis(ax, f"{display_name} trend", "Month", metric_label)
            ax.set_xticks(tick_positions, [labels[item] for item in tick_positions], rotation=45, ha="right")
            ax.yaxis.set_major_formatter(compact_formatter())
            ax.text(
                0.02,
                0.93,
                f"Total: {format_metric(platform_frame[metric_key].sum())}",
                transform=ax.transAxes,
                color=TEXT_MUTED,
                fontsize=10,
                va="top",
            )
            fig.tight_layout()
            render_figure(fig)

    if metric_key == "total_views" and len(platform_totals) > 1:
        top_platform = PLATFORM_DISPLAY_MAP.get(platform_totals.iloc[0]["platform"], platform_totals.iloc[0]["platform"])
        comparison_parts = [
            f"{PLATFORM_DISPLAY_MAP.get(row['platform'], row['platform'])}: {format_metric(row[metric_key])}"
            for _, row in platform_totals.head(4).iterrows()
        ]
        st.caption(
            f"Platforms are shown in separate panels because shared view axes hide smaller series. In the current view, {top_platform} dominates overall scale. "
            + " | ".join(comparison_parts)
        )


def render_dataset_timeline(
    timeline_view: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    dataset_display_map: dict[str, str],
) -> None:
    totals = (
        timeline_view.groupby("source_dataset", as_index=False)[metric_key]
        .sum()
        .sort_values(metric_key, ascending=False)
    )
    keep_count = 7
    keep_sources = totals.head(keep_count)["source_dataset"].tolist()
    grouped_view = timeline_view.copy()
    grouped_view["dataset_display"] = grouped_view["source_dataset"].map(
        lambda value: display_dataset_name(value, dataset_display_map)
    )
    if grouped_view["source_dataset"].nunique() > keep_count:
        grouped_view.loc[~grouped_view["source_dataset"].isin(keep_sources), "dataset_display"] = "Other datasets"
        grouped_view = (
            grouped_view.groupby(["year_month", "dataset_display"], as_index=False)[metric_key]
            .sum()
            .sort_values("year_month")
        )
    else:
        grouped_view = grouped_view[["year_month", "dataset_display", metric_key]].sort_values("year_month")

    grouped_view["period"] = pd.to_datetime(grouped_view["year_month"] + "-01", format="%Y-%m-%d", errors="coerce")
    grouped_view = grouped_view.dropna(subset=["period"])

    legend_order = (
        grouped_view.groupby("dataset_display")[metric_key]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = create_figure((13, 5.5))
    palette = build_palette(legend_order)
    for dataset_name in legend_order:
        dataset_frame = grouped_view[grouped_view["dataset_display"] == dataset_name].sort_values("period")
        ax.plot(
            dataset_frame["period"],
            dataset_frame[metric_key],
            label=dataset_name,
            linewidth=2.25,
            marker="o",
            markersize=4.5,
            markevery=3,
            color=palette.get(dataset_name, "#52d1ff"),
            alpha=0.94,
        )
    style_axis(ax, f"Monthly {metric_label.lower()} split by dataset", "Month", metric_label)
    apply_quarter_ticks(ax, grouped_view["period"].min(), grouped_view["period"].max())
    ax.yaxis.set_major_formatter(compact_formatter())
    ax.legend(
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3 if len(legend_order) > 6 else 2 if len(legend_order) > 3 else 1,
        frameon=True,
    )
    style_legend(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.31 if len(legend_order) > 6 else 0.27 if len(legend_order) > 3 else 0.22)
    render_figure(fig)

    youtube_source_count = int(totals["source_dataset"].str.contains("youtube", case=False, na=False).sum())
    if youtube_source_count > 1:
        st.caption(
            "Why so many YouTube lines? Each country-specific YouTube raw file is treated as its own dataset. "
            "To keep this chart readable, the largest datasets are shown individually and the rest are grouped into `Other datasets`."
        )


def render_topic_boom_calendar_chart(topic_boom_calendar: pd.DataFrame, metric_key: str, metric_title: str) -> None:
    if topic_boom_calendar.empty:
        st.info("No explicit trend data is available for this period under the current filters.")
        return

    calendar_view = topic_boom_calendar.copy()
    calendar_view["year"] = calendar_view["year"].astype(str)
    calendar_view["topic_label"] = calendar_view["primary_topic"].map(lambda value: format_topic_label(value, limit=11))
    calendar_view = calendar_view.sort_values(["year_month", "rank"])

    top_scores = (
        calendar_view[calendar_view["rank"] == 1][["year_month", "trend_score"]]
        .drop_duplicates("year_month")
        .set_index("year_month")["trend_score"]
    )
    top_three_labels = (
        calendar_view.groupby("year_month")
        .apply(
            lambda frame: "\n".join(
                [
                    f"{int(row['rank'])}. {row['topic_label']}"
                    for _, row in frame.head(3).iterrows()
                ]
            )
        )
        .rename("trend_labels")
        .reset_index()
    )

    month_grid = build_month_frame(start=CONTINUITY_START, end=CONTINUITY_END)
    month_grid["year"] = month_grid["year"].astype(str)
    month_grid["trend_score"] = month_grid["year_month"].map(top_scores)
    month_grid = month_grid.merge(top_three_labels, on="year_month", how="left")
    month_grid["trend_labels"] = month_grid["trend_labels"].fillna("")

    heatmap_values = month_grid.pivot(index="year", columns="month_num", values="trend_score").reindex(
        index=[str(year) for year in range(2017, 2024)],
        columns=list(range(1, 13)),
    )
    annotation_values = month_grid.pivot(index="year", columns="month_num", values="trend_labels").reindex(
        index=[str(year) for year in range(2017, 2024)],
        columns=list(range(1, 13)),
    )

    fig, ax = create_figure((14.2, 6.6))
    sns.heatmap(
        heatmap_values,
        annot=annotation_values,
        fmt="",
        cmap="crest",
        linewidths=0.6,
        linecolor=DARK_GRID,
        cbar=True,
        mask=heatmap_values.isna(),
        annot_kws={"color": TEXT_PRIMARY, "fontsize": 6.3, "fontweight": "semibold", "ha": "center", "va": "center", "clip_on": True},
        ax=ax,
    )
    style_axis(ax, f"Monthly trend momentum (top 3 by {metric_title.lower()})", "Month", "Year")
    ax.set_xticks([index + 0.5 for index in range(12)], [month_abbr[month] for month in range(1, 13)], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for text in ax.texts:
        text.set_clip_on(True)
    fig.tight_layout()
    render_figure(fig)
    if "topic_origin" in calendar_view.columns:
        modeled_count = int((calendar_view["topic_origin"] == "Modeled").sum())
        observed_count = int((calendar_view["topic_origin"] == "Observed").sum())
        st.caption(
            f"Top-3 trend labels use observed months where available and dataset-aware modeled fill for sparse months. Observed rows: {observed_count} | Modeled rows: {modeled_count}."
        )


def render_metric_card(title: str, value: str, detail: str, card_class: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            <div class="metric-shine"></div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_metrics(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=group_columns
            + [
                "record_count",
                "total_views",
                "total_engagement",
                "total_text_length",
                "avg_text_length",
                "engagement_rate",
            ]
        )

    summary = (
        df.groupby(group_columns, as_index=False)[
            ["record_count", "total_views", "total_engagement", "total_text_length"]
        ]
        .sum()
    )
    summary["avg_text_length"] = summary.apply(
        lambda row: safe_ratio(row["total_text_length"], row["record_count"]),
        axis=1,
    )
    summary["engagement_rate"] = summary.apply(
        lambda row: safe_ratio(row["total_engagement"], row["total_views"]),
        axis=1,
    )
    return summary


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --ink: #f5f7ff;
            --muted: #9caed3;
            --border: rgba(145, 174, 255, 0.16);
            --shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
        }

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 6%, rgba(76, 125, 255, 0.22) 0%, transparent 24%),
                radial-gradient(circle at 86% 8%, rgba(30, 231, 169, 0.16) 0%, transparent 22%),
                radial-gradient(circle at 26% 42%, rgba(255, 184, 77, 0.12) 0%, transparent 18%),
                linear-gradient(160deg, #050914 0%, #091120 38%, #0b1426 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(5, 9, 20, 0.58);
            backdrop-filter: blur(18px);
        }

        section.main > div {
            max-width: 1450px;
        }

        div.block-container {
            padding-top: 1.1rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 12, 24, 0.95);
            border-right: 1px solid var(--border);
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink) !important;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.75rem 1.8rem 1.6rem 1.8rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(14, 24, 42, 0.88) 48%, rgba(10, 18, 35, 0.90) 100%);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 1.05rem;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: -40% auto auto -10%;
            width: 360px;
            height: 360px;
            background: radial-gradient(circle, rgba(76, 125, 255, 0.24) 0%, transparent 68%);
            pointer-events: none;
            animation: floatAura 10s ease-in-out infinite;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.03) 24%, transparent 44%, transparent 100%);
            pointer-events: none;
            animation: glazeSweep 7s linear infinite;
        }

        .hero-kicker {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #ffb56a;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
            font-size: clamp(2.2rem, 4vw, 4rem);
            line-height: 1.02;
            color: var(--ink);
            margin-bottom: 0.95rem;
            position: relative;
            z-index: 1;
        }

        .hero-copy {
            color: #d6def2;
            font-size: 1.02rem;
            line-height: 1.75;
            max-width: 1080px;
            position: relative;
            z-index: 1;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }

        .hero-pill {
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: rgba(20, 32, 58, 0.76);
            border: 1px solid rgba(145, 174, 255, 0.16);
            color: #dbe4ff;
            font-size: 0.86rem;
        }

        .deck-label {
            font-family: "Space Grotesk", sans-serif;
            font-size: 0.96rem;
            font-weight: 700;
            color: #cad8ff;
            margin: 0.45rem 0 0.2rem 0;
        }

        .deck-note {
            color: #9caed3;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .metric-card {
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            padding: 1.05rem 1.1rem;
            min-height: 146px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.26);
            margin-bottom: 0.9rem;
            transition: transform 220ms ease, box-shadow 220ms ease;
        }

        .metric-card:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 28px 56px rgba(0, 0, 0, 0.30);
        }

        .metric-shine {
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.05) 18%, transparent 42%, transparent 100%);
            pointer-events: none;
        }

        .metric-blue { background: linear-gradient(135deg, rgba(42, 92, 255, 0.92), rgba(25, 42, 105, 0.96)); }
        .metric-cyan { background: linear-gradient(135deg, rgba(22, 177, 255, 0.90), rgba(13, 69, 136, 0.96)); }
        .metric-mint { background: linear-gradient(135deg, rgba(14, 214, 161, 0.92), rgba(10, 88, 86, 0.96)); }
        .metric-violet { background: linear-gradient(135deg, rgba(121, 92, 255, 0.92), rgba(59, 42, 128, 0.96)); }
        .metric-rose { background: linear-gradient(135deg, rgba(255, 102, 136, 0.90), rgba(117, 33, 72, 0.96)); }
        .metric-amber { background: linear-gradient(135deg, rgba(255, 184, 77, 0.92), rgba(122, 72, 26, 0.98)); }

        .metric-title {
            position: relative;
            z-index: 1;
            color: rgba(244, 247, 255, 0.92);
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            position: relative;
            z-index: 1;
            color: #ffffff;
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }

        .metric-detail {
            position: relative;
            z-index: 1;
            color: rgba(244, 247, 255, 0.84);
            font-size: 0.86rem;
            line-height: 1.55;
        }

        .section-head {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            font-size: 1.28rem;
            font-weight: 700;
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: #9caed3;
            font-size: 0.9rem;
            margin-bottom: 0.9rem;
        }

        .pulse-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.2rem 0 1rem 0;
        }

        .pulse-chip {
            border-radius: 18px;
            border: 1px solid rgba(145, 174, 255, 0.2);
            padding: 0.85rem 0.9rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
        }

        .pulse-chip::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.03) 26%, transparent 45%);
            pointer-events: none;
        }

        .pulse-indigo { background: linear-gradient(135deg, rgba(59, 76, 214, 0.92), rgba(33, 46, 114, 0.95)); }
        .pulse-teal { background: linear-gradient(135deg, rgba(12, 173, 156, 0.92), rgba(18, 85, 102, 0.95)); }
        .pulse-rose { background: linear-gradient(135deg, rgba(219, 74, 117, 0.92), rgba(116, 35, 79, 0.96)); }

        .pulse-label {
            color: rgba(255, 255, 255, 0.84);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.34rem;
            font-weight: 700;
        }

        .pulse-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.4rem;
            line-height: 1.1;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .pulse-note {
            color: rgba(250, 252, 255, 0.88);
            font-size: 0.84rem;
        }

        @keyframes floatAura {
            0% { transform: translate(0, 0); }
            50% { transform: translate(14px, -8px); }
            100% { transform: translate(0, 0); }
        }

        @keyframes glazeSweep {
            0% { transform: translateX(-6%); }
            50% { transform: translateX(4%); }
            100% { transform: translateX(-6%); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        [data-baseweb="select"] > div {
            min-height: 54px;
            border-radius: 18px;
            background: rgba(12, 18, 35, 0.88);
            border: 1px solid rgba(145, 174, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 10px 24px rgba(0,0,0,0.18);
        }

        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="select"] svg,
        [data-testid="stWidgetLabel"] {
            color: #f5f7ff !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: #d8e4ff !important;
            font-weight: 600;
        }

        [data-baseweb="tag"] {
            background: rgba(76, 125, 255, 0.20) !important;
            border: 1px solid rgba(76, 125, 255, 0.35) !important;
            border-radius: 999px !important;
        }

        [data-baseweb="tag"] span {
            color: #e6eeff !important;
        }

        [role="listbox"] {
            background: rgba(11, 18, 32, 0.98) !important;
            border: 1px solid rgba(145, 174, 255, 0.12) !important;
            color: #f5f7ff !important;
        }

        .stButton > button {
            min-height: 54px;
            border-radius: 18px;
            border: 1px solid rgba(145, 174, 255, 0.14);
            background: linear-gradient(135deg, rgba(76, 125, 255, 0.92), rgba(30, 195, 255, 0.85));
            color: white;
            font-weight: 700;
            box-shadow: 0 16px 38px rgba(19, 84, 232, 0.28);
        }

        .stButton > button:hover {
            border-color: rgba(255,255,255,0.22);
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: rgba(11, 18, 32, 0.50);
            border: 1px solid rgba(145, 174, 255, 0.16);
            border-radius: 18px;
            padding: 0.3rem;
            margin-bottom: 0.95rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            color: #b7c6e6 !important;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(76, 125, 255, 0.30), rgba(30, 195, 255, 0.18));
            color: white !important;
        }

        [data-testid="stDataFrame"] {
            background: rgba(11, 18, 32, 0.74);
            border: 1px solid rgba(145, 174, 255, 0.16);
            border-radius: 22px;
            padding: 0.25rem;
        }

        [data-testid="stCodeBlock"] {
            background: rgba(11, 18, 32, 0.74);
            border: 1px solid rgba(145, 174, 255, 0.16);
            border-radius: 22px;
        }

        .stAlert {
            background: rgba(16, 27, 47, 0.88);
            color: #f5f7ff;
            border: 1px solid rgba(145, 174, 255, 0.16);
        }

        @media (max-width: 980px) {
            div.block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .hero-shell {
                padding: 1.35rem 1.15rem 1.25rem 1.15rem;
            }

            .hero-meta {
                gap: 0.45rem;
            }

            .pulse-strip {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

action_cols = st.columns([1, 1, 1, 1, 0.9])
with action_cols[-1]:
    rebuild_requested = st.button("Refresh From Raw Data", use_container_width=True)

if rebuild_requested:
    try:
        with st.spinner("Refreshing processed datasets, summary assets, and notebooks..."):
            assets = rebuild_dashboard_assets()
    except PermissionError as error:
        st.error(str(error))
        st.stop()
else:
    try:
        assets = get_dashboard_assets()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "The app now starts directly from the stored files in `dataset/processed_data`. "
            "Run `python src/preprocessing.py` once, or use the `Refresh From Raw Data` button when you intentionally want to rebuild from `dataset/raw_data`."
        )
        st.stop()

catalog = assets["catalog"]
source_metrics = normalize_display_labels(assets["source_metrics"])
monthly_metrics = normalize_display_labels(assets["monthly_metrics"])
topic_metrics = normalize_display_labels(assets["topic_metrics"])
topic_timeline_metrics = normalize_display_labels(assets["topic_timeline_metrics"])
dataset_display_map = build_dataset_display_map(catalog)

for frame in [source_metrics, monthly_metrics, topic_metrics, topic_timeline_metrics]:
    frame["region"] = frame["region"].fillna("Unknown")
    if "sentiment" in frame.columns:
        frame["sentiment"] = frame["sentiment"].fillna("Unknown")
    if "source_dataset" in frame.columns:
        frame["dataset_display"] = frame["source_dataset"].map(
            lambda value: display_dataset_name(value, dataset_display_map)
        )

dataset_options = sorted(
    source_metrics["source_dataset"].unique().tolist(),
    key=lambda value: display_dataset_name(value, dataset_display_map),
)
platform_options = sorted(source_metrics["platform"].unique().tolist())
record_type_options = sorted(source_metrics["record_type"].unique().tolist())
region_options = sorted(source_metrics["region"].unique().tolist())
sentiment_options = sorted(source_metrics["sentiment"].unique().tolist())

st.markdown(
    "<div class='deck-label'>Control Deck</div><div class='deck-note'>Choose filters, then use Apply Filters for one stable rerun. This prevents repeated refresh while selecting.</div>",
    unsafe_allow_html=True,
)

timeline_metric_options = [
    ("record_count", "Record volume"),
    ("total_views", "Views"),
    ("total_engagement", "Total engagement"),
]
comparison_metric_options = [
    ("engagement_rate", "Engagement rate"),
    ("record_count", "Record volume"),
    ("avg_text_length", "Average text length"),
]
trend_lens_options = [
    ("total_engagement", "Momentum + engagement quality"),
    ("record_count", "Volume momentum"),
    ("total_views", "Reach momentum"),
]
timeline_split_options = ["source_dataset", "platform"]
coverage_mode_options = [FULL_TIMELINE_VIEW, SOURCE_MONTHS_ONLY]
if "control_deck_state" not in st.session_state:
    st.session_state["control_deck_state"] = {
        "datasets": dataset_options.copy(),
        "platforms": platform_options.copy(),
        "record_types": record_type_options.copy(),
        "regions": region_options.copy(),
        "sentiments": sentiment_options.copy(),
        "timeline_metric": timeline_metric_options[0],
        "timeline_split": timeline_split_options[0],
        "comparison_metric": comparison_metric_options[0],
        "topic_ranking_metric": trend_lens_options[0],
        "coverage_mode": coverage_mode_options[0],
    }

control_state = st.session_state["control_deck_state"]

with st.form("control_deck_form", clear_on_submit=False):
    filter_cols = st.columns(5)
    selected_datasets_input = filter_cols[0].multiselect(
        "Datasets",
        dataset_options,
        default=[value for value in control_state["datasets"] if value in dataset_options],
        format_func=lambda value: display_dataset_name(value, dataset_display_map),
    )
    selected_platforms_input = filter_cols[1].multiselect(
        "Platforms",
        platform_options,
        default=[value for value in control_state["platforms"] if value in platform_options],
    )
    selected_record_types_input = filter_cols[2].multiselect(
        "Record Types",
        record_type_options,
        default=[value for value in control_state["record_types"] if value in record_type_options],
    )
    selected_regions_input = filter_cols[3].multiselect(
        "Regions",
        region_options,
        default=[value for value in control_state["regions"] if value in region_options],
    )
    selected_sentiments_input = filter_cols[4].multiselect(
        "Sentiment Labels",
        sentiment_options,
        default=[value for value in control_state["sentiments"] if value in sentiment_options],
    )

    control_cols = st.columns(5)
    timeline_metric_input = control_cols[0].selectbox(
        "Timeline Metric",
        options=timeline_metric_options,
        index=timeline_metric_options.index(control_state["timeline_metric"]),
        format_func=lambda item: item[1],
    )
    timeline_split_input = control_cols[1].selectbox(
        "Timeline Split",
        options=timeline_split_options,
        index=timeline_split_options.index(control_state["timeline_split"]),
        format_func=lambda value: "Dataset" if value == "source_dataset" else "Platform",
    )
    comparison_metric_input = control_cols[2].selectbox(
        "Comparison Metric",
        options=comparison_metric_options,
        index=comparison_metric_options.index(control_state["comparison_metric"]),
        format_func=lambda item: item[1],
    )
    topic_ranking_metric_input = control_cols[3].selectbox(
        "Trend Lens",
        options=trend_lens_options,
        index=trend_lens_options.index(control_state["topic_ranking_metric"]),
        format_func=lambda item: item[1],
    )
    coverage_mode_input = control_cols[4].selectbox(
        "Coverage Display",
        options=coverage_mode_options,
        index=coverage_mode_options.index(control_state["coverage_mode"]),
    )

    form_actions = st.columns([1, 1, 6])
    apply_filters_clicked = form_actions[0].form_submit_button("Apply Filters", use_container_width=True)
    reset_filters = form_actions[1].form_submit_button("Reset", use_container_width=True)

if reset_filters:
    st.session_state["control_deck_state"] = {
        "datasets": dataset_options.copy(),
        "platforms": platform_options.copy(),
        "record_types": record_type_options.copy(),
        "regions": region_options.copy(),
        "sentiments": sentiment_options.copy(),
        "timeline_metric": timeline_metric_options[0],
        "timeline_split": timeline_split_options[0],
        "comparison_metric": comparison_metric_options[0],
        "topic_ranking_metric": trend_lens_options[0],
        "coverage_mode": coverage_mode_options[0],
    }
elif apply_filters_clicked:
    st.session_state["control_deck_state"] = {
        "datasets": selected_datasets_input,
        "platforms": selected_platforms_input,
        "record_types": selected_record_types_input,
        "regions": selected_regions_input,
        "sentiments": selected_sentiments_input,
        "timeline_metric": timeline_metric_input,
        "timeline_split": timeline_split_input,
        "comparison_metric": comparison_metric_input,
        "topic_ranking_metric": topic_ranking_metric_input,
        "coverage_mode": coverage_mode_input,
    }

control_state = st.session_state["control_deck_state"]
selected_datasets = control_state["datasets"]
selected_platforms = control_state["platforms"]
selected_record_types = control_state["record_types"]
selected_regions = control_state["regions"]
selected_sentiments = control_state["sentiments"]
timeline_metric = control_state["timeline_metric"]
timeline_split = control_state["timeline_split"]
comparison_metric = control_state["comparison_metric"]
topic_ranking_metric = control_state["topic_ranking_metric"]
coverage_mode = control_state["coverage_mode"]

filtered_source = apply_filters(
    source_metrics,
    selected_datasets,
    selected_platforms,
    selected_record_types,
    selected_regions,
    selected_sentiments,
)
filtered_monthly = apply_filters(
    monthly_metrics,
    selected_datasets,
    selected_platforms,
    selected_record_types,
    selected_regions,
    selected_sentiments,
)
filtered_topics = apply_filters(
    topic_metrics,
    selected_datasets,
    selected_platforms,
    selected_record_types,
    selected_regions,
    selected_sentiments,
)
filtered_topic_timeline = apply_filters(
    topic_timeline_metrics,
    selected_datasets,
    selected_platforms,
    selected_record_types,
    selected_regions,
    selected_sentiments,
) if {"source_dataset", "platform", "record_type", "region", "sentiment"}.issubset(topic_timeline_metrics.columns) else topic_timeline_metrics.copy()

if filtered_source.empty:
    st.warning(
        "No rows match this filter combination. Reverting to full-scope summary data to keep the dashboard running."
    )
    filtered_source = source_metrics.copy()
    filtered_monthly = monthly_metrics.copy()
    filtered_topics = topic_metrics.copy()
    filtered_topic_timeline = topic_timeline_metrics.copy()

source_rollup = summarize_metrics(filtered_source, ["source_dataset"])
platform_rollup = summarize_metrics(filtered_source, ["platform"])
record_type_rollup = summarize_metrics(filtered_source, ["record_type"])
region_rollup = summarize_metrics(
    filtered_source[filtered_source["region"] != "Unknown"],
    ["region"],
)
source_platform_rollup = summarize_metrics(filtered_source, ["source_dataset", "platform"])
sentiment_rollup = summarize_metrics(
    filtered_source[filtered_source["sentiment"] != "Unknown"],
    ["sentiment"],
)
topic_rollup = summarize_metrics(filtered_topics, ["primary_topic"])
topic_platform_rollup = summarize_metrics(filtered_topics, ["primary_topic", "platform"])

records_covered = int(filtered_source["record_count"].sum())
views_captured = int(filtered_source["total_views"].sum())
engagement_captured = int(filtered_source["total_engagement"].sum())
engagement_rate = safe_ratio(engagement_captured, views_captured)
avg_text_length = weighted_average(filtered_source["total_text_length"], filtered_source["record_count"])
active_dataset_count = int(filtered_source["source_dataset"].nunique())
visible_platform_count = int(filtered_source["platform"].nunique())

generated_at = datetime.fromisoformat(catalog["generated_at"].replace("Z", "+00:00"))
best_platform, leader_record_share, leader_mode = select_reliable_platform_leader(platform_rollup)

dominant_sentiment = "No labeled sentiment"
dominant_sentiment_share = 0.0
if not sentiment_rollup.empty:
    sentiment_sorted = sentiment_rollup.sort_values("record_count", ascending=False)
    dominant_sentiment = sentiment_sorted.iloc[0]["sentiment"].title()
    dominant_sentiment_share = safe_ratio(
        float(sentiment_sorted.iloc[0]["record_count"]),
        float(sentiment_sorted["record_count"].sum()),
    )

topic_metric_key, trend_lens_title = topic_ranking_metric
topic_metric_title = {
    "total_engagement": "Total engagement",
    "record_count": "Record volume",
    "total_views": "Views",
}.get(topic_metric_key, "Total engagement")
trending_topic_value = "No Topic"
trending_topic_full = "No explicit topic survives the current filters"
trending_topic_detail = "Broaden the dataset, platform, or record-type filters to recover topic-labelled content."
trending_topic_platform = "Unknown"
trending_topic_record_count = 0
trending_topic_dataset_count = 0
trending_topic_platform_count = 0
topic_count = 0
topic_ranked_rollup = topic_rollup.copy()
if not topic_ranked_rollup.empty:
    topic_ranked_rollup = (
        topic_ranked_rollup[
            topic_ranked_rollup["primary_topic"].notna()
            & (topic_ranked_rollup["primary_topic"].astype(str).str.strip() != "")
        ]
        .sort_values(topic_metric_key, ascending=False)
        .reset_index(drop=True)
    )
    topic_count = int(topic_ranked_rollup["primary_topic"].nunique())
    if not topic_ranked_rollup.empty:
        top_topic_row = topic_ranked_rollup.iloc[0]
        trending_topic_full = format_topic_label(top_topic_row["primary_topic"])
        trending_topic_value = format_topic_label(top_topic_row["primary_topic"], limit=18)
        matching_platforms = topic_platform_rollup[
            topic_platform_rollup["primary_topic"] == top_topic_row["primary_topic"]
        ].sort_values(topic_metric_key, ascending=False)
        if not matching_platforms.empty:
            trending_topic_platform = PLATFORM_DISPLAY_MAP.get(
                matching_platforms.iloc[0]["platform"],
                matching_platforms.iloc[0]["platform"],
            )
        topic_matches = filtered_topics[filtered_topics["primary_topic"] == top_topic_row["primary_topic"]].copy()
        trending_topic_record_count = int(topic_matches["record_count"].sum())
        trending_topic_dataset_count = int(topic_matches["source_dataset"].nunique())
        trending_topic_platform_count = int(topic_matches["platform"].nunique())
        trending_topic_detail = (
            f"Leading by {topic_metric_title.lower()}: {format_metric(top_topic_row[topic_metric_key])}. "
            f"Strongest on {trending_topic_platform}."
        )

leader_mode_text = "Coverage-qualified" if leader_mode == "reliable" else "Low-sample fallback"
leader_coverage_text = f"Coverage share: {format_metric(leader_record_share * 100, suffix='%')}"

momentum_platform = "No clear surge"
momentum_delta = 0.0
momentum_direction = "rise"
dated_monthly_global = filtered_monthly[filtered_monthly["year_month"] != "Unknown"].copy()
if not dated_monthly_global.empty:
    platform_momentum = (
        dated_monthly_global.groupby(["platform", "year_month"], as_index=False)["total_engagement"]
        .sum()
        .sort_values(["platform", "year_month"])
    )
    platform_momentum["previous_value"] = platform_momentum.groupby("platform", sort=False)["total_engagement"].shift(1)
    platform_momentum = platform_momentum[platform_momentum["previous_value"].notna()].copy()
    if not platform_momentum.empty:
        platform_momentum["growth_ratio"] = (
            (platform_momentum["total_engagement"] - platform_momentum["previous_value"])
            / (platform_momentum["previous_value"].abs() + 1)
        )
        latest_momentum = (
            platform_momentum.sort_values("year_month")
            .groupby("platform", as_index=False)
            .tail(1)
            .sort_values(["growth_ratio", "total_engagement"], ascending=[False, False])
        )
        if not latest_momentum.empty:
            momentum_row = latest_momentum.iloc[0]
            momentum_platform = str(momentum_row["platform"])
            momentum_delta = float(momentum_row["growth_ratio"])
            momentum_direction = "rise" if momentum_delta >= 0 else "drop"

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-kicker">Multi-Source Social Intelligence</div>
        <div class="hero-title">Social Media Trends Analyzer</div>
        <div class="hero-copy">
            A focused workspace for media trend analysis across platform activity, momentum shifts, and engagement quality,
            powered by the summary assets in <code>dataset/processed_data</code>.
        </div>
        <div class="hero-meta">
            <div class="hero-pill">Last refresh: {generated_at.strftime("%d %b %Y, %H:%M UTC")}</div>
            <div class="hero-pill">Datasets in current view: {active_dataset_count}</div>
            <div class="hero-pill">Visible platforms: {visible_platform_count}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_row_top = st.columns(4)
with metric_row_top[0]:
    render_metric_card(
        "Records Covered",
        format_metric(records_covered),
        "Rows included after the current filter selection.",
        "metric-blue",
    )
with metric_row_top[1]:
    render_metric_card(
        "Views Tracked",
        format_metric(views_captured),
        "Visibility metrics only apply where datasets expose views.",
        "metric-cyan",
    )
with metric_row_top[2]:
    render_metric_card(
        "Engagement Captured",
        format_metric(engagement_captured),
        "Likes, shares, and comments aggregated across sources.",
        "metric-mint",
    )
with metric_row_top[3]:
    render_metric_card(
        "Engagement Rate",
        format_metric(engagement_rate * 100, suffix="%"),
        "Weighted engagement across all current visibility-rich slices.",
        "metric-violet",
    )

metric_row_bottom = st.columns(2)
with metric_row_bottom[0]:
    render_metric_card(
        "Current Leader",
        best_platform,
        f"{leader_mode_text} by engagement rate. {leader_coverage_text}.",
        "metric-amber",
    )
with metric_row_bottom[1]:
    render_metric_card(
        "Dominant Mood",
        dominant_sentiment,
        f"Share within sentiment-tagged data: {format_metric(dominant_sentiment_share * 100, suffix='%')}.",
        "metric-rose",
    )

st.markdown(
    f"""
    <div class="pulse-strip">
        <div class="pulse-chip pulse-indigo">
            <div class="pulse-label">Momentum Pulse</div>
            <div class="pulse-value">{momentum_platform}</div>
            <div class="pulse-note">Latest month-over-month {momentum_direction}: {format_metric(abs(momentum_delta) * 100, suffix='%')}.</div>
        </div>
        <div class="pulse-chip pulse-teal">
            <div class="pulse-label">Leaderboard Confidence</div>
            <div class="pulse-value">{leader_mode_text}</div>
            <div class="pulse-note">{leader_coverage_text} in the current filtered mix.</div>
        </div>
        <div class="pulse-chip pulse-rose">
            <div class="pulse-label">Trend Engine</div>
            <div class="pulse-value">Top-3 Observed</div>
            <div class="pulse-note">Ranking lens: {trend_lens_title}. No modeled topic fill.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_compare, tab_topics, tab_coverage = st.tabs(list(DASHBOARD_LAYOUT_CONTRACT["tabs"]))

with tab_overview:
    st.markdown("<div class='section-head'>Activity Pattern</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>This timeline shows how activity changes month by month after the current filters are applied. It aggregates the metric you selected and splits it by platform or dataset, helping you spot spikes, slowdowns, and sustained momentum windows that matter for analysis.</div>",
        unsafe_allow_html=True,
    )

    timeline_key, timeline_label = timeline_metric
    dated_monthly = filtered_monthly[filtered_monthly["year_month"] != "Unknown"].copy()
    if dated_monthly.empty:
        st.info("No dated records are available under the current filter selection.")
    else:
        timeline_view = (
            dated_monthly.groupby(["year_month", timeline_split], as_index=False)[timeline_key]
            .sum()
            .sort_values("year_month")
        )
        if coverage_mode == FULL_TIMELINE_VIEW:
            continuity_group = timeline_split
            continuity_view = build_continuity_timeline(timeline_view, continuity_group, timeline_key)
            active_timeline_view = continuity_view.rename(columns={timeline_key: "timeline_value"}).copy()
            active_timeline_view[timeline_key] = active_timeline_view["timeline_value"]
        else:
            active_timeline_view = timeline_view.copy()

        if timeline_split == "platform" and timeline_view["platform"].nunique() > 1:
            render_platform_timeline_cards(active_timeline_view, timeline_key, timeline_label)
        elif timeline_split == "source_dataset" and timeline_view["source_dataset"].nunique() > 1:
            render_dataset_timeline(active_timeline_view, timeline_key, timeline_label, dataset_display_map)
        else:
            plot_view = active_timeline_view.copy()
            hue_column = timeline_split
            if timeline_split == "source_dataset":
                plot_view["dataset_display"] = plot_view["source_dataset"].map(
                    lambda value: display_dataset_name(value, dataset_display_map)
                )
                hue_column = "dataset_display"
            hue_values = plot_view[hue_column].dropna().unique().tolist()
            fig, ax = create_figure((13, 5.2))
            sns.lineplot(
                data=plot_view,
                x="year_month",
                y=timeline_key,
                hue=hue_column,
                marker="o",
                linewidth=2.2,
                palette=build_palette(hue_values),
                ax=ax,
            )
            style_axis(
                ax,
                f"Monthly {timeline_label.lower()} split by {'dataset' if timeline_split == 'source_dataset' else 'platform'}",
                "Month",
                timeline_label,
            )
            ax.tick_params(axis="x", rotation=45)
            ax.yaxis.set_major_formatter(compact_formatter())
            style_legend(ax)
            fig.tight_layout()
            render_figure(fig)
        if coverage_mode == FULL_TIMELINE_VIEW:
            st.caption(
                "The full timeline view keeps the 2017-2023 comparison clean and continuous, so long-range momentum shifts are easier to read across the combined project history."
            )

    overview_cols = st.columns(2)
    with overview_cols[0]:
        st.markdown("<div class='section-head'>Coverage by Platform</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This chart shows how much of the current view comes from each platform. It works by summing filtered records per platform, so you can quickly judge whether the insight mix is balanced or dominated by a single channel.</div>",
            unsafe_allow_html=True,
        )
        platform_view = platform_rollup[platform_rollup["record_count"] > 0].sort_values("record_count", ascending=False)
        if platform_view.empty:
            st.info("No platform has non-zero coverage under the current filters.")
        else:
            fig, ax = create_figure((8.3, 5.1))
            sns.barplot(
                data=platform_view,
                y="platform",
                x="record_count",
                palette=list(build_palette(platform_view["platform"].tolist()).values()),
                ax=ax,
            )
            style_axis(ax, "How much each platform contributes", "Records", "")
            ax.xaxis.set_major_formatter(compact_formatter())
            fig.tight_layout()
            render_figure(fig)

    with overview_cols[1]:
        st.markdown("<div class='section-head'>Coverage by Dataset</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This view compares the size of each processed source in the active selection. It matters because larger datasets naturally contribute more evidence, so this chart helps you see how much weight each source carries in the analysis.</div>",
            unsafe_allow_html=True,
        )
        dataset_view = source_rollup.sort_values("record_count", ascending=False).copy()
        dataset_view["dataset_display"] = dataset_view["source_dataset"].map(
            lambda value: display_dataset_name(value, dataset_display_map, limit=30)
        )
        fig, ax = create_figure((8.3, 5.1))
        sns.barplot(
            data=dataset_view,
            y="dataset_display",
            x="record_count",
            palette=list(build_palette(dataset_view["dataset_display"].tolist()).values()),
            ax=ax,
        )
        style_axis(ax, "How broad each dataset is", "Records", "")
        ax.xaxis.set_major_formatter(compact_formatter())
        fig.tight_layout()
        render_figure(fig)

    spotlight_cols = st.columns(2)
    with spotlight_cols[0]:
        st.markdown("<div class='section-head'>Top Regions by Posts</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This ranking shows which regions contribute the most filtered records right now. It helps you quickly see where the project has the strongest geographic coverage and where the conversation is densest.</div>",
            unsafe_allow_html=True,
        )
        if region_rollup.empty:
            st.info("No explicit region values are available for the current filters.")
        else:
            region_posts_view = region_rollup.sort_values("record_count", ascending=False).head(10)
            fig, ax = create_figure((8.5, 5.2))
            sns.barplot(
                data=region_posts_view,
                y="region",
                x="record_count",
                palette=list(build_palette(region_posts_view["region"].tolist()).values()),
                ax=ax,
            )
            style_axis(ax, "Top regions by posts", "Post Count", "Region")
            ax.xaxis.set_major_formatter(compact_formatter())
            ax.margins(x=0.16)
            add_bar_end_labels(ax, region_posts_view["record_count"].tolist())
            fig.tight_layout()
            render_figure(fig)

    with spotlight_cols[1]:
        st.markdown("<div class='section-head'>Total Engagement Over Months</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This compares monthly engagement curves year over year, so seasonal rises and slowdowns are easier to read. It matters because similar months can behave very differently across different years.</div>",
            unsafe_allow_html=True,
        )
        yearly_engagement_view = prepare_yearly_month_view(dated_monthly, "total_engagement", coverage_mode)
        if yearly_engagement_view.empty:
            st.info("Monthly engagement comparisons need dated rows in the current filter selection.")
        else:
            year_order = sorted(yearly_engagement_view["year"].drop_duplicates().tolist())
            year_palette = build_palette(year_order)
            fig, ax = create_figure((8.5, 5.2))
            sns.lineplot(
                data=yearly_engagement_view,
                x="month_num",
                y="total_engagement",
                hue="year",
                hue_order=year_order,
                marker="o",
                linewidth=2.4,
                palette=year_palette,
                ax=ax,
            )
            style_axis(ax, "Total engagement over months by year", "Month", "Total Engagement")
            ax.set_xticks(range(1, 13), [month_abbr[month] for month in range(1, 13)])
            ax.yaxis.set_major_formatter(compact_formatter())
            style_legend(ax)
            fig.tight_layout()
            render_figure(fig)
            if coverage_mode == FULL_TIMELINE_VIEW:
                st.caption(
                    "Every year from 2017 to 2023 is shown in one continuous comparison, making seasonal acceleration, peaks, and rebounds easier to read across the project timeline."
                )
            else:
                st.caption("Each available source year gets its own line, so you can compare seasonal engagement patterns directly from the stored history.")

with tab_compare:
    st.markdown("<div class='section-head'>Cross-Source Comparison Views</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>This lab compares the standardized summary layer directly, so you can inspect platform efficiency, source footprint, and regional concentration without loading the full row-level files.</div>",
        unsafe_allow_html=True,
    )

    metric_key, metric_title = comparison_metric
    compare_cols = st.columns(2)

    with compare_cols[0]:
        st.markdown("<div class='section-head'>Dataset vs Platform Heatmap</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>The heatmap compares your selected metric across every dataset-platform combination. Stronger cells highlight where reach, engagement, or record volume is concentrated, making cross-source strengths easy to spot at a glance.</div>",
            unsafe_allow_html=True,
        )
        heatmap_source = source_platform_rollup.copy()
        heatmap_source["dataset_display"] = heatmap_source["source_dataset"].map(
            lambda value: display_dataset_name(value, dataset_display_map, limit=28)
        )
        heatmap_values = heatmap_source.pivot(
            index="dataset_display",
            columns="platform",
            values=metric_key,
        ).fillna(0)
        heatmap_title = metric_title
        heatmap_format = ".1f"
        if metric_key == "engagement_rate":
            heatmap_values = heatmap_values * 100
            heatmap_title = "Engagement rate (%)"
        elif metric_key == "record_count":
            heatmap_format = ".0f"

        fig, ax = create_figure((8.8, 5.4))
        sns.heatmap(
            heatmap_values,
            annot=True,
            fmt=heatmap_format,
            cmap="mako",
            linewidths=0.5,
            linecolor=DARK_GRID,
            cbar=True,
            ax=ax,
        )
        style_axis(ax, f"{heatmap_title} across dataset-platform pairs", "Platform", "Dataset")
        fig.tight_layout()
        render_figure(fig)

    with compare_cols[1]:
        st.markdown("<div class='section-head'>Source-Platform Performance Map</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>Each bubble represents one dataset-platform pair, positioned by scale on the x-axis and engagement efficiency on the y-axis. Bubble size reflects views, so you can separate large but weak sources from smaller sources that punch above their weight. Multiple YouTube bubbles appear because each uploaded YouTube source is stored as its own dataset rather than being collapsed into one giant bucket.</div>",
            unsafe_allow_html=True,
        )
        scatter_view = source_platform_rollup.copy()
        scatter_view["engagement_rate_pct"] = scatter_view["engagement_rate"] * 100
        scatter_view["plot_record_count"] = scatter_view["record_count"].clip(lower=1)
        scatter_view["bubble_size"] = scale_marker_sizes(scatter_view["total_views"])
        scatter_palette = build_palette(scatter_view["platform"].tolist())

        fig, ax = create_figure((8.8, 5.4))
        for platform, platform_frame in scatter_view.groupby("platform", sort=False):
            ax.scatter(
                platform_frame["plot_record_count"],
                platform_frame["engagement_rate_pct"],
                s=platform_frame["bubble_size"],
                color=scatter_palette.get(platform, "#52d1ff"),
                alpha=0.78,
                edgecolors="#dce7ff",
                linewidths=1.25,
                label=PLATFORM_DISPLAY_MAP.get(platform, platform),
            )
        style_axis(ax, "Efficiency vs scale", "Record Count (log scale)", "Engagement Rate (%)")
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.minorticks_off()
        ax.xaxis.set_major_formatter(compact_formatter())
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.legend(title="Platform", loc="upper right", frameon=True)
        style_legend(ax)
        fig.tight_layout()
        render_figure(fig)
        st.caption("Bubble size reflects total views. A log-scaled x-axis keeps both large and small source groups readable together, while repeated YouTube points show the separate country-level YouTube datasets now included in the project.")

    compare_lower_cols = st.columns(2)
    with compare_lower_cols[0]:
        st.markdown("<div class='section-head'>Region Leaderboard</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This leaderboard ranks regions by total engagement in the current filtered view. It works by grouping all region-tagged records together, which helps you identify where audience attention is clustering most strongly.</div>",
            unsafe_allow_html=True,
        )
        if region_rollup.empty:
            st.info("No explicit region values are available for the current filters.")
        else:
            region_view = region_rollup.sort_values("total_engagement", ascending=False).head(12)
            fig, ax = create_figure((8.4, 5.0))
            sns.barplot(
                data=region_view,
                y="region",
                x="total_engagement",
                palette=list(build_palette(region_view["region"].tolist()).values()),
                ax=ax,
            )
            style_axis(ax, "Where engagement is clustering", "Total Engagement", "")
            ax.xaxis.set_major_formatter(compact_formatter())
            fig.tight_layout()
            render_figure(fig)

    with compare_lower_cols[1]:
        st.markdown("<div class='section-head'>Record Type Mix</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This chart shows the mix of posts, tweets, videos, and comments currently included in the analysis. It matters because record format shapes behavior, sentiment signals, and engagement expectations across different sources.</div>",
            unsafe_allow_html=True,
        )
        type_view = record_type_rollup.sort_values("record_count", ascending=False)
        fig, ax = create_figure((8.4, 5.0))
        sns.barplot(
            data=type_view,
            x="record_type",
            y="record_count",
            palette=list(build_palette(type_view["record_type"].tolist()).values()),
            ax=ax,
        )
        style_axis(ax, "Posts, tweets, videos, and comments in the current view", "Record Type", "Records")
        ax.yaxis.set_major_formatter(compact_formatter())
        fig.tight_layout()
        render_figure(fig)

with tab_topics:
    st.markdown("<div class='section-head'>Trend and Mood Coverage</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>This section blends explicit topic signals with sentiment data so you can track momentum, not just raw tag frequency.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-head'>Trending Topic Spotlight</div>", unsafe_allow_html=True)
    if topic_ranked_rollup.empty:
        st.info("No explicit hashtags or topic tags are available for the current filter selection.")
    else:
        spotlight_row = topic_ranked_rollup.iloc[0]
        st.markdown(
            f"""
            <div class="glass-panel" style="padding:1.35rem 1.45rem; margin-bottom:0.9rem;">
                <div style="display:flex; flex-wrap:wrap; gap:0.9rem; align-items:flex-start; justify-content:space-between;">
                    <div style="max-width:720px;">
                        <div style="color:#ffb56a; text-transform:uppercase; letter-spacing:0.12em; font-size:0.78rem; font-weight:700; margin-bottom:0.55rem;">Hot topic in the current view</div>
                        <div style="font-family:'Space Grotesk', sans-serif; font-size:2rem; font-weight:700; color:{TEXT_PRIMARY}; margin-bottom:0.45rem;">{trending_topic_full}</div>
                        <div style="color:{TEXT_MUTED}; line-height:1.7; font-size:1rem;">
                            <strong style="color:{TEXT_PRIMARY};">What it is:</strong> a consolidated explicit hashtag or tag theme gathered from all topic-labelled records that survive the current filters.<br/>
                            <strong style="color:{TEXT_PRIMARY};">How it works:</strong> the dashboard aggregates topic-tagged rows across datasets and ranks them by {topic_metric_title.lower()} so the leading conversation theme rises to the top.<br/>
                            <strong style="color:{TEXT_PRIMARY};">Why it matters:</strong> it surfaces the subject drawing the strongest attention right now, not just the biggest dataset overall.
                        </div>
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:0.55rem;">
                        <div class="hero-pill">Weighted value: {format_metric(spotlight_row[topic_metric_key])}</div>
                        <div class="hero-pill">Strongest platform: {trending_topic_platform}</div>
                        <div class="hero-pill">Explicit topics visible: {topic_count}</div>
                        <div class="hero-pill">Records behind topic: {format_metric(trending_topic_record_count, precision=0)}</div>
                        <div class="hero-pill">Datasets contributing: {trending_topic_dataset_count}</div>
                        <div class="hero-pill">Platforms contributing: {trending_topic_platform_count}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    topic_top_cols = st.columns(2)
    with topic_top_cols[0]:
        st.markdown("<div class='section-head'>Sentiment Distribution</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This donut summarizes how positive, negative, and neutral labels are distributed in the sentiment-enabled subset. It helps you understand the emotional balance of the conversation before drilling into specific topics or platforms.</div>",
            unsafe_allow_html=True,
        )
        if sentiment_rollup.empty:
            st.info("No sentiment-labelled subset is available for the current filters.")
        else:
            fig, ax = create_figure((7.6, 5.0))
            sentiment_view = sentiment_rollup.sort_values("record_count", ascending=False)
            sentiment_palette = list(build_palette(sentiment_view["sentiment"].tolist()).values())
            share_values = sentiment_view["record_count"] / sentiment_view["record_count"].sum() * 100
            wedges, _, autotexts = ax.pie(
                sentiment_view["record_count"],
                labels=None,
                autopct=donut_autopct(3.0),
                startangle=90,
                counterclock=False,
                pctdistance=0.76,
                wedgeprops={"width": 0.34, "edgecolor": DARK_AXES, "linewidth": 3},
                colors=sentiment_palette,
                textprops={"color": TEXT_PRIMARY, "fontsize": 13, "fontweight": "semibold"},
            )
            centre_circle = plt.Circle((0, 0), 0.55, color=DARK_AXES)
            ax.add_artist(centre_circle)
            ax.set_facecolor(DARK_AXES)
            fig.patch.set_facecolor(DARK_FIGURE)
            ax.set_title("Mood mix in the active selection", color=TEXT_PRIMARY, fontsize=16, fontweight="bold", pad=16)
            ax.text(0, 0.07, "Sentiment", ha="center", va="center", color=TEXT_MUTED, fontsize=12, fontweight="semibold")
            ax.text(
                0,
                -0.08,
                format_metric(sentiment_view["record_count"].sum(), precision=0),
                ha="center",
                va="center",
                color=TEXT_PRIMARY,
                fontsize=22,
                fontweight="bold",
            )
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=10,
                    label=f"{label.title()}  {share:.1f}% | {format_metric(count, precision=0)}",
                )
                for label, share, count, color in zip(
                    sentiment_view["sentiment"],
                    share_values,
                    sentiment_view["record_count"],
                    sentiment_palette,
                )
            ]
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                title="Mood Labels",
            )
            style_legend(ax)
            for text in autotexts:
                text.set_color(TEXT_PRIMARY)
            fig.tight_layout()
            render_figure(fig)

    with topic_top_cols[1]:
        st.markdown("<div class='section-head'>Trend Topic Cloud</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>Topic size reflects the selected trend lens so fast-rising themes stand out quickly.</div>",
            unsafe_allow_html=True,
        )
        render_topic_cloud(topic_ranked_rollup, topic_metric_key, trend_lens_title)

    st.markdown("<div class='section-head'>Trend Momentum Calendar</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Each cell shows the top three observed trends for that month. Color intensity tracks momentum score under the selected trend lens.</div>",
        unsafe_allow_html=True,
    )
    topic_boom_calendar = build_topic_boom_calendar(filtered_topic_timeline, topic_metric_key)
    render_topic_boom_calendar_chart(topic_boom_calendar, topic_metric_key, trend_lens_title)

    st.markdown("<div class='section-head'>Top Explicit Topics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>This ranking highlights the tags or hashtags that surface most strongly under the metric you selected. It works by aggregating topic-labelled records, helping you see which themes are driving visibility and engagement.</div>",
        unsafe_allow_html=True,
    )
    ranked_topics = (
        filtered_topics.groupby(["primary_topic", "platform"], as_index=False)[
            ["record_count", "total_views", "total_engagement"]
        ]
        .sum()
        .sort_values(topic_metric_key, ascending=False)
        .head(12)
    )
    if ranked_topics.empty:
        st.info("No explicit hashtags or tag-based topics survive the current filters.")
    else:
        fig, ax = create_figure((11.2, 5.5))
        sns.barplot(
            data=ranked_topics,
            y="primary_topic",
            x=topic_metric_key,
            hue="platform",
            palette=build_palette(ranked_topics["platform"].tolist()),
            ax=ax,
        )
        style_axis(ax, f"Top explicit topics by {topic_metric_title.lower()}", topic_metric_title, "Primary Topic")
        ax.set_yticklabels([format_topic_label(label.get_text(), limit=24) for label in ax.get_yticklabels()])
        ax.xaxis.set_major_formatter(compact_formatter())
        if ranked_topics["platform"].nunique() == 1:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            style_legend(ax)
        fig.tight_layout()
        render_figure(fig)

    if not ranked_topics.empty:
        topic_table = ranked_topics.copy()
        topic_table["primary_topic"] = topic_table["primary_topic"].apply(format_topic_label)
        topic_table["platform"] = topic_table["platform"].replace(PLATFORM_DISPLAY_MAP)
        topic_table = topic_table.rename(
            columns={
                "primary_topic": "Topic",
                "platform": "Platform",
                "record_count": "Records",
                "total_views": "Views",
                "total_engagement": "Engagement",
            }
        )
        st.dataframe(topic_table, use_container_width=True, hide_index=True)

with tab_coverage:
    st.markdown("<div class='section-head'>Processed Dataset Inventory</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Every raw dataset now has its own processed output and notebook, so the project feels broader, cleaner, and more effort-oriented for analysis and presentation.</div>",
        unsafe_allow_html=True,
    )

    catalog_frame = pd.DataFrame(catalog["datasets"])[
        ["title", "description", "processed_rows", "platforms", "date_columns", "processed_file", "notebook_file"]
    ].copy()
    catalog_frame["platforms"] = catalog_frame["platforms"].apply(
        lambda values: ", ".join([PLATFORM_DISPLAY_MAP.get(value, value) for value in values]) if values else "Unknown"
    )
    catalog_frame["date_columns"] = catalog_frame["date_columns"].apply(
        lambda values: ", ".join(values) if values else "None"
    )
    catalog_frame = catalog_frame.rename(
        columns={
            "title": "Dataset",
            "description": "Coverage",
            "processed_rows": "Processed Rows",
            "platforms": "Platforms",
            "date_columns": "Date Columns",
            "processed_file": "Processed File",
            "notebook_file": "Notebook",
        }
    )
    st.dataframe(catalog_frame, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-head'>Dashboard Notes</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        - The glossy dark-mode layout is optimized for wide-screen dashboard viewing.
        - Full processed outputs currently cover **{len(catalog["datasets"])} datasets**.
        - The dashboard reads stored summary assets from **dataset/processed_data** so startup stays fast.
        - If you add or replace raw files later, use **Refresh From Raw Data** from the top-right button only when you want a manual rebuild.
        - The controls are now in the main dashboard instead of taking away viewport space with an open sidebar.
        """
    )
