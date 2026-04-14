from __future__ import annotations

import pandas as pd

DEFAULT_CALENDAR_START = "2017-01"
DEFAULT_CALENDAR_END = "2023-12"
NO_EXPLICIT_TOPIC = "no_explicit_topic"


def get_dashboard_layout_contract() -> dict[str, tuple[str, ...]]:
    """Return a small, testable UI contract for high-level dashboard structure."""
    return {
        "hero_pills": (
            "Last refresh",
            "Datasets in current view",
            "Visible platforms",
        ),
        "metric_cards": (
            "Records Covered",
            "Views Tracked",
            "Engagement Captured",
            "Engagement Rate",
            "Current Leader",
            "Dominant Mood",
        ),
        "tabs": (
            "Overview",
            "Comparison Lab",
            "Trends & Sentiment",
            "Dataset Coverage",
        ),
    }


def build_month_frame(
    start: str = DEFAULT_CALENDAR_START,
    end: str = DEFAULT_CALENDAR_END,
) -> pd.DataFrame:
    periods = pd.period_range(start=start, end=end, freq="M")
    return pd.DataFrame(
        {
            "period": periods,
            "year_month": periods.astype(str),
            "year": periods.year.astype(int),
            "month_num": periods.month.astype(int),
        }
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    adjusted_denominator = denominator.fillna(0).clip(lower=1)
    return numerator.fillna(0) / adjusted_denominator


def _get_lens_weights(trend_lens: str) -> tuple[float, float, float]:
    if trend_lens == "record_count":
        return (0.45, 0.4, 0.15)
    if trend_lens == "total_views":
        return (0.5, 0.35, 0.15)
    return (0.5, 0.2, 0.3)


def build_top_trends_calendar(
    topic_timeline_metrics: pd.DataFrame,
    trend_lens: str,
    top_n: int = 3,
    fill_missing_months: bool = False,
    start: str = DEFAULT_CALENDAR_START,
    end: str = DEFAULT_CALENDAR_END,
) -> pd.DataFrame:
    """Build top-N monthly trend leaders using momentum, share, and quality signals.

    Output columns:
    - year_month, year, month_num, rank
    - primary_topic
    - trend_score, momentum_score, share_score, quality_score
    - trend_lens
    """
    required_columns = {"year_month", "primary_topic", "record_count", "total_views", "total_engagement"}
    if topic_timeline_metrics.empty or not required_columns.issubset(topic_timeline_metrics.columns):
        return pd.DataFrame(
            columns=[
                "year_month",
                "year",
                "month_num",
                "rank",
                "primary_topic",
                "trend_score",
                "momentum_score",
                "share_score",
                "quality_score",
                "trend_lens",
                "topic_origin",
            ]
        )

    trend_source = topic_timeline_metrics.copy()
    trend_source = trend_source[
        trend_source["primary_topic"].notna()
        & (trend_source["primary_topic"].astype(str).str.strip() != "")
        & (trend_source["primary_topic"].astype(str).str.lower() != NO_EXPLICIT_TOPIC)
    ].copy()

    if trend_source.empty:
        return pd.DataFrame(
            columns=[
                "year_month",
                "year",
                "month_num",
                "rank",
                "primary_topic",
                "trend_score",
                "momentum_score",
                "share_score",
                "quality_score",
                "trend_lens",
                "topic_origin",
            ]
        )

    rollup = (
        trend_source.groupby(["year_month", "primary_topic"], as_index=False)[
            ["record_count", "total_views", "total_engagement"]
        ]
        .sum()
        .sort_values("year_month")
    )
    rollup["period"] = pd.PeriodIndex(rollup["year_month"], freq="M")
    rollup = rollup.sort_values(["primary_topic", "period"])

    lens_column = trend_lens if trend_lens in {"record_count", "total_views", "total_engagement"} else "total_engagement"

    rollup["previous_lens_value"] = rollup.groupby("primary_topic", sort=False)[lens_column].shift(1)
    rollup["growth_signal"] = _safe_divide(
        rollup[lens_column] - rollup["previous_lens_value"].fillna(0),
        rollup["previous_lens_value"].abs().fillna(0) + 1,
    ).clip(lower=-1.0, upper=3.0)

    rollup["engagement_quality"] = _safe_divide(
        rollup["total_engagement"],
        rollup["total_views"].where(rollup["total_views"] > 0, rollup["record_count"]),
    )

    rollup = rollup.sort_values(["year_month", "growth_signal", "record_count"], ascending=[True, False, False])

    rollup["momentum_score"] = (
        rollup.groupby("year_month")["growth_signal"]
        .rank(method="dense", pct=True)
        .fillna(0.5)
    )

    monthly_lens_totals = rollup.groupby("year_month")[lens_column].transform("sum").clip(lower=1)
    rollup["share_score"] = _safe_divide(rollup[lens_column], monthly_lens_totals).clip(lower=0)

    rollup["quality_score"] = (
        rollup.groupby("year_month")["engagement_quality"]
        .rank(method="dense", pct=True)
        .fillna(0.5)
    )

    momentum_weight, share_weight, quality_weight = _get_lens_weights(lens_column)
    rollup["trend_score"] = (
        (rollup["momentum_score"] * momentum_weight)
        + (rollup["share_score"] * share_weight)
        + (rollup["quality_score"] * quality_weight)
    )

    top_trends = (
        rollup.sort_values(
            ["year_month", "trend_score", "momentum_score", "share_score", "record_count", "primary_topic"],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("year_month", as_index=False)
        .head(max(int(top_n), 1))
        .copy()
    )

    top_trends["rank"] = top_trends.groupby("year_month").cumcount() + 1
    top_trends["year"] = top_trends["period"].dt.year.astype(int)
    top_trends["month_num"] = top_trends["period"].dt.month.astype(int)
    top_trends["trend_lens"] = lens_column
    top_trends["topic_origin"] = "Observed"

    if fill_missing_months:
        month_frame = build_month_frame(start=start, end=end)
        available_months = set(top_trends["year_month"].tolist())
        missing_months = month_frame[~month_frame["year_month"].isin(available_months)].copy()

        if not missing_months.empty:
            month_topic_scores = (
                top_trends.groupby(["month_num", "primary_topic"], as_index=False)["trend_score"]
                .mean()
                .sort_values(["month_num", "trend_score"], ascending=[True, False])
            )
            year_topic_scores = (
                top_trends.groupby(["year", "primary_topic"], as_index=False)["trend_score"]
                .mean()
                .sort_values(["year", "trend_score"], ascending=[True, False])
            )
            global_topic_scores = (
                top_trends.groupby("primary_topic", as_index=False)["trend_score"]
                .mean()
                .sort_values("trend_score", ascending=False)
            )

            observed_periods = pd.PeriodIndex(top_trends["year_month"], freq="M")
            observed_with_period = top_trends.copy()
            observed_with_period["period"] = observed_periods

            modeled_rows: list[dict[str, object]] = []
            for _, missing_row in missing_months.iterrows():
                month_num = int(missing_row["month_num"])
                year = int(missing_row["year"])
                target_period = pd.Period(missing_row["year_month"], freq="M")

                month_candidates = month_topic_scores[month_topic_scores["month_num"] == month_num].head(6)
                year_candidates = year_topic_scores[year_topic_scores["year"] == year].head(6)

                if observed_with_period.empty:
                    nearest_candidates = pd.DataFrame(columns=["primary_topic", "trend_score"])
                else:
                    target_ordinal = target_period.ordinal
                    observed_with_period["distance"] = (
                        observed_with_period["period"].map(lambda period: period.ordinal) - target_ordinal
                    ).abs()
                    nearest_candidates = (
                        observed_with_period.sort_values(["distance", "trend_score"], ascending=[True, False])
                        [["primary_topic", "trend_score"]]
                        .drop_duplicates("primary_topic")
                        .head(6)
                    )

                candidate_pool = list(
                    dict.fromkeys(
                        month_candidates["primary_topic"].tolist()
                        + year_candidates["primary_topic"].tolist()
                        + nearest_candidates["primary_topic"].tolist()
                        + global_topic_scores.head(6)["primary_topic"].tolist()
                    )
                )
                candidate_pool = [topic for topic in candidate_pool if isinstance(topic, str) and topic.strip()]
                if not candidate_pool:
                    continue

                scored_candidates: list[tuple[str, float]] = []
                for topic in candidate_pool:
                    month_score = float(
                        month_candidates.loc[month_candidates["primary_topic"] == topic, "trend_score"].mean()
                    ) if not month_candidates.empty else 0.0
                    if pd.isna(month_score):
                        month_score = 0.0

                    year_score = float(
                        year_candidates.loc[year_candidates["primary_topic"] == topic, "trend_score"].mean()
                    ) if not year_candidates.empty else 0.0
                    if pd.isna(year_score):
                        year_score = 0.0

                    nearest_score = float(
                        nearest_candidates.loc[nearest_candidates["primary_topic"] == topic, "trend_score"].mean()
                    ) if not nearest_candidates.empty else 0.0
                    if pd.isna(nearest_score):
                        nearest_score = 0.0

                    global_score = float(
                        global_topic_scores.loc[global_topic_scores["primary_topic"] == topic, "trend_score"].mean()
                    ) if not global_topic_scores.empty else 0.0
                    if pd.isna(global_score):
                        global_score = 0.0

                    combined_score = (0.45 * month_score) + (0.25 * year_score) + (0.20 * nearest_score) + (0.10 * global_score)
                    scored_candidates.append((topic, max(0.01, combined_score)))

                ranked = sorted(scored_candidates, key=lambda item: (-item[1], item[0]))[: max(int(top_n), 1)]
                for rank, (topic, score) in enumerate(ranked, start=1):
                    modeled_rows.append(
                        {
                            "year_month": missing_row["year_month"],
                            "year": year,
                            "month_num": month_num,
                            "rank": rank,
                            "primary_topic": topic,
                            "trend_score": score,
                            "momentum_score": score,
                            "share_score": score,
                            "quality_score": score,
                            "trend_lens": lens_column,
                            "topic_origin": "Modeled",
                        }
                    )

            if modeled_rows:
                modeled_frame = pd.DataFrame(modeled_rows)
                top_trends = pd.concat([top_trends, modeled_frame], ignore_index=True)

    return top_trends[
        [
            "year_month",
            "year",
            "month_num",
            "rank",
            "primary_topic",
            "trend_score",
            "momentum_score",
            "share_score",
            "quality_score",
            "trend_lens",
            "topic_origin",
        ]
    ].reset_index(drop=True)


def select_reliable_platform_leader(
    platform_rollup: pd.DataFrame,
    min_record_share: float = 0.03,
    min_absolute_records: int = 500,
) -> tuple[str, float, str]:
    """Select a platform leader with a minimum coverage floor.

    Returns:
    - platform name
    - record share in [0, 1]
    - selection mode: "reliable" or "fallback"
    """
    required_columns = {"platform", "record_count", "total_views", "engagement_rate"}
    if platform_rollup.empty or not required_columns.issubset(platform_rollup.columns):
        return ("N/A", 0.0, "fallback")

    platform_view = platform_rollup.copy()
    platform_view = platform_view[platform_view["total_views"] > 0].copy()
    if platform_view.empty:
        return ("N/A", 0.0, "fallback")

    total_records = float(platform_view["record_count"].sum())
    if total_records <= 0:
        return ("N/A", 0.0, "fallback")

    platform_view["record_share"] = platform_view["record_count"] / total_records
    platform_view["is_reliable"] = (
        (platform_view["record_share"] >= float(min_record_share))
        & (platform_view["record_count"] >= int(min_absolute_records))
    )

    reliable_view = platform_view[platform_view["is_reliable"]].copy()
    if reliable_view.empty:
        fallback_row = platform_view.sort_values(["record_count", "engagement_rate"], ascending=[False, False]).iloc[0]
        return (
            str(fallback_row["platform"]),
            float(fallback_row["record_share"]),
            "fallback",
        )

    leader_row = reliable_view.sort_values(["engagement_rate", "record_count"], ascending=[False, False]).iloc[0]
    return (
        str(leader_row["platform"]),
        float(leader_row["record_share"]),
        "reliable",
    )
