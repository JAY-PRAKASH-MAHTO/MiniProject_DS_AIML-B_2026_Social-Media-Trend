from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import (  # noqa: E402
    DATASET_NOTEBOOK_DIR,
    NOTEBOOKS_DIR,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_project_directories,
)

ensure_project_directories()
PREPROCESS_DIR = PROCESSED_DATA_DIR

UNKNOWN_VALUE = "Unknown"
NO_TOPIC_VALUE = "no_explicit_topic"
PLATFORM_NAME_MAP = {
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "twitter": "Twitter",
    "facebook": "Facebook",
}
COUNTRY_CODE_REGION_MAP = {
    "CA": "Canada",
    "DE": "Germany",
    "FR": "France",
    "GB": "United Kingdom",
    "IN": "India",
    "JP": "Japan",
    "KR": "South Korea",
    "MX": "Mexico",
    "RU": "Russia",
    "US": "United States",
}

CANONICAL_COLUMNS = [
    "record_id",
    "source_dataset",
    "source_file",
    "platform",
    "record_type",
    "content_type",
    "publisher",
    "published_at",
    "trend_date",
    "year",
    "month",
    "year_month",
    "year_week",
    "region",
    "primary_topic",
    "sentiment",
    "content_text",
    "views",
    "likes",
    "shares",
    "comments",
    "dislikes",
    "total_engagement",
    "engagement_rate",
    "trend_lag_days",
    "text_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "link_count",
    "question_count",
    "exclamation_count",
]

SUMMARY_METRICS = [
    "record_count",
    "total_views",
    "total_likes",
    "total_shares",
    "total_comments",
    "total_dislikes",
    "total_engagement",
    "total_text_length",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    source_file: str
    processed_file: str
    notebook_file: str
    description: str
    processor: Callable[[pd.DataFrame, int], tuple[pd.DataFrame, pd.DataFrame]]
    chunk_size: int | None = None
    read_csv_kwargs: dict[str, Any] = field(default_factory=dict)


def _normalize_whitespace(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str)
    return cleaned.str.replace(r"\s+", " ", regex=True).str.strip()


def _normalize_title_series(series: pd.Series, default: str = UNKNOWN_VALUE) -> pd.Series:
    cleaned = _normalize_whitespace(series).str.title()
    return cleaned.where(cleaned.ne(""), default)


def _normalize_platform_series(series: pd.Series, default: str = UNKNOWN_VALUE) -> pd.Series:
    cleaned = _normalize_whitespace(series).str.lower()
    normalized = cleaned.map(PLATFORM_NAME_MAP)
    fallback = cleaned.str.title().where(cleaned.ne(""), default)
    return normalized.fillna(fallback)


def _normalize_lower_series(series: pd.Series, default: str = UNKNOWN_VALUE) -> pd.Series:
    cleaned = _normalize_whitespace(series).str.lower()
    return cleaned.where(cleaned.ne(""), default.lower())


def _normalize_hashtag_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_whitespace(series).str.lower()
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    cleaned = cleaned.str.replace(r"^([^#].*)$", r"#\1", regex=True)
    return cleaned.where(cleaned.ne(""), NO_TOPIC_VALUE)


def _to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _parse_classic_youtube_trending_date(series: pd.Series) -> pd.Series:
    text = _normalize_whitespace(series)
    parsed = pd.to_datetime(text, format="%y.%d.%m", errors="coerce")
    if parsed.notna().all():
        return parsed
    fallback = pd.to_datetime(text.loc[parsed.isna()], errors="coerce")
    parsed.loc[parsed.isna()] = fallback
    return parsed


def _parse_slash_date(series: pd.Series) -> pd.Series:
    text = _normalize_whitespace(series)
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return parsed.where(parsed.notna(), fallback)


def _to_int_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    numeric = numeric.clip(lower=0)
    return numeric.round().astype("int64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).round(6)


def _year_month_from_dates(series: pd.Series) -> pd.Series:
    year_month = series.dt.to_period("M").astype(str)
    return year_month.where(series.notna(), UNKNOWN_VALUE)


def _year_week_from_dates(series: pd.Series) -> pd.Series:
    iso = series.dt.isocalendar()
    year = iso["year"].astype("Int64").astype(str)
    week = iso["week"].astype("Int64").astype(str).str.zfill(2)
    result = year + "-W" + week
    return result.where(series.notna(), UNKNOWN_VALUE)


def _extract_primary_topic_from_tags(series: pd.Series) -> pd.Series:
    def pick_first(value: Any) -> str:
        text = "" if pd.isna(value) else str(value).strip()
        if not text or text == "[None]":
            return NO_TOPIC_VALUE
        for token in text.split("|"):
            token = token.strip().lower()
            if token and token != "[none]":
                return token
        return NO_TOPIC_VALUE

    return series.map(pick_first)


def _extract_primary_topic_from_text(series: pd.Series) -> pd.Series:
    hashtag_pattern = re.compile(r"#\w+")

    def pick_first(value: Any) -> str:
        text = "" if pd.isna(value) else str(value)
        match = hashtag_pattern.search(text.lower())
        return match.group(0) if match else NO_TOPIC_VALUE

    return series.map(pick_first)


def _add_date_parts(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    dates = df[date_column]
    df["year"] = dates.dt.year.fillna(0).astype("int64")
    df["month"] = dates.dt.month.fillna(0).astype("int64")
    df["year_month"] = _year_month_from_dates(dates)
    df["year_week"] = _year_week_from_dates(dates)
    return df


def _add_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    text = _normalize_whitespace(df[text_column])
    df[text_column] = text
    df["text_length"] = text.str.len().astype("int64")
    df["word_count"] = text.str.split().str.len().fillna(0).astype("int64")
    df["hashtag_count"] = text.str.count(r"#\w+").astype("int64")
    df["mention_count"] = text.str.count(r"@\w+").astype("int64")
    df["link_count"] = text.str.count(r"http\S+|www\.\S+").astype("int64")
    df["question_count"] = text.str.count(r"\?").astype("int64")
    df["exclamation_count"] = text.str.count(r"!").astype("int64")
    return df


def _build_canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.loc[:, CANONICAL_COLUMNS].copy()
    for column in ["platform", "record_type", "content_type", "publisher", "region", "primary_topic", "sentiment"]:
        canonical[column] = canonical[column].fillna(UNKNOWN_VALUE).replace("", UNKNOWN_VALUE)
    canonical["primary_topic"] = canonical["primary_topic"].replace(UNKNOWN_VALUE, NO_TOPIC_VALUE)
    canonical["content_text"] = canonical["content_text"].fillna("")
    return canonical


def transform_viral_social_media_trends(df: pd.DataFrame, row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.rename(
        columns={
            "Post_ID": "post_id",
            "Post_Date": "post_date",
            "Platform": "platform",
            "Hashtag": "hashtag",
            "Content_Type": "content_type",
            "Region": "region",
            "Views": "views",
            "Likes": "likes",
            "Shares": "shares",
            "Comments": "comments",
            "Engagement_Level": "engagement_level",
        }
    ).copy()

    working["post_id"] = _normalize_whitespace(working["post_id"])
    working["post_date"] = _to_datetime(working["post_date"])
    working["platform"] = _normalize_platform_series(working["platform"])
    working["hashtag"] = _normalize_hashtag_series(working["hashtag"])
    working["content_type"] = _normalize_title_series(working["content_type"])
    working["region"] = _normalize_title_series(working["region"])
    working["engagement_level"] = _normalize_title_series(working["engagement_level"])

    for metric in ["views", "likes", "shares", "comments"]:
        working[metric] = _to_int_series(working[metric])

    working["total_engagement"] = working["likes"] + working["shares"] + working["comments"]
    working["engagement_rate"] = _safe_ratio(working["total_engagement"], working["views"])
    working["source_dataset"] = "viral_social_media_trends"
    working["source_file"] = "Cleaned_Viral_Social_Media_Trends.csv"
    working["record_type"] = "post"
    working["publisher"] = UNKNOWN_VALUE
    working["published_at"] = working["post_date"]
    working["trend_date"] = working["post_date"]
    working["primary_topic"] = working["hashtag"]
    working["sentiment"] = UNKNOWN_VALUE
    working["content_text"] = working["hashtag"]
    working["dislikes"] = 0
    working["trend_lag_days"] = 0
    working = _add_date_parts(working, "post_date")
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "post_id",
            "post_date",
            "platform",
            "hashtag",
            "content_type",
            "region",
            "views",
            "likes",
            "shares",
            "comments",
            "engagement_level",
            "total_engagement",
            "engagement_rate",
            "year",
            "month",
            "year_month",
            "year_week",
            "source_dataset",
            "record_type",
            "primary_topic",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["post_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["published_at"],
            "trend_date": working["trend_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_indian_youtube_trending(df: pd.DataFrame, row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.rename(
        columns={
            "publishedAt": "published_at",
            "channelId": "channel_id",
            "channelTitle": "channel_title",
            "categoryId": "category_id",
            "trending_date": "trending_date",
            "view_count": "views",
            "comment_count": "comments",
            "thumbnail_link": "thumbnail_link",
        }
    ).copy()

    working["video_id"] = _normalize_whitespace(working["video_id"])
    working["title"] = _normalize_whitespace(working["title"])
    working["published_at"] = _to_datetime(working["published_at"])
    working["trending_date"] = _to_datetime(working["trending_date"])
    working["channel_id"] = _normalize_whitespace(working["channel_id"])
    working["channel_title"] = _normalize_whitespace(working["channel_title"]).replace("", UNKNOWN_VALUE)
    working["category_id"] = _to_int_series(working["category_id"])
    working["tags"] = _normalize_whitespace(working["tags"])
    working["description"] = _normalize_whitespace(working["description"])

    for metric in ["views", "likes", "dislikes", "comments"]:
        working[metric] = _to_int_series(working[metric])

    working["comments_disabled"] = working["comments_disabled"].fillna(False).astype(bool)
    working["ratings_disabled"] = working["ratings_disabled"].fillna(False).astype(bool)
    working["description_preview"] = working["description"].str.slice(0, 280)
    working["description_length"] = working["description"].str.len().astype("int64")
    working["title_word_count"] = working["title"].str.split().str.len().fillna(0).astype("int64")
    working["tag_count"] = working["tags"].replace("[None]", "").str.count(r"\|").add(1)
    working.loc[working["tags"].isin(["", "[None]"]), "tag_count"] = 0
    working["tag_count"] = working["tag_count"].astype("int64")
    working["primary_topic"] = _extract_primary_topic_from_tags(working["tags"])
    working["platform"] = "YouTube"
    working["record_type"] = "video"
    working["content_type"] = "Trending Video"
    working["region"] = "India"
    working["source_dataset"] = "indian_youtube_trending"
    working["source_file"] = "IN_youtube_trending_data.csv"
    working["publisher"] = working["channel_title"]
    working["sentiment"] = UNKNOWN_VALUE
    working["shares"] = 0
    working["total_engagement"] = working["likes"] + working["comments"]
    working["engagement_rate"] = _safe_ratio(working["total_engagement"], working["views"])
    working["trend_lag_days"] = (working["trending_date"] - working["published_at"]).dt.days.fillna(0).astype("int64")
    working["content_text"] = (working["title"] + " " + working["tags"].replace("[None]", "")).str.strip()
    working = _add_date_parts(working, "trending_date")
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "video_id",
            "title",
            "published_at",
            "trending_date",
            "channel_id",
            "channel_title",
            "category_id",
            "tags",
            "primary_topic",
            "tag_count",
            "views",
            "likes",
            "dislikes",
            "comments",
            "total_engagement",
            "engagement_rate",
            "trend_lag_days",
            "comments_disabled",
            "ratings_disabled",
            "title_word_count",
            "description_length",
            "description_preview",
            "platform",
            "region",
            "source_dataset",
            "record_type",
            "year",
            "month",
            "year_month",
            "year_week",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["video_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["published_at"],
            "trend_date": working["trending_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_regional_youtube_trending(
    df: pd.DataFrame,
    row_offset: int = 0,
    *,
    dataset_name: str,
    region: str,
    source_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()

    working["video_id"] = _normalize_whitespace(working["video_id"])
    working["title"] = _normalize_whitespace(working["title"])
    working["publish_time"] = _to_datetime(working["publish_time"])
    working["trending_date"] = _parse_classic_youtube_trending_date(working["trending_date"])
    working["channel_title"] = _normalize_whitespace(working["channel_title"]).replace("", UNKNOWN_VALUE)
    working["category_id"] = _to_int_series(working["category_id"])
    working["tags"] = _normalize_whitespace(working["tags"])
    working["description"] = _normalize_whitespace(working["description"])

    for metric in ["views", "likes", "dislikes", "comment_count"]:
        working[metric] = _to_int_series(working[metric])

    working["comments_disabled"] = working["comments_disabled"].fillna(False).astype(bool)
    working["ratings_disabled"] = working["ratings_disabled"].fillna(False).astype(bool)
    working["description_preview"] = working["description"].str.slice(0, 280)
    working["description_length"] = working["description"].str.len().astype("int64")
    working["title_word_count"] = working["title"].str.split().str.len().fillna(0).astype("int64")
    working["tag_count"] = working["tags"].replace("[none]", "").replace("[None]", "").str.count(r"\|").add(1)
    working.loc[working["tags"].isin(["", "[None]", "[none]"]), "tag_count"] = 0
    working["tag_count"] = working["tag_count"].astype("int64")
    working["primary_topic"] = _extract_primary_topic_from_tags(working["tags"])
    working["platform"] = "YouTube"
    working["record_type"] = "video"
    working["content_type"] = "Trending Video"
    working["region"] = region
    working["source_dataset"] = dataset_name
    working["source_file"] = source_file
    working["publisher"] = working["channel_title"]
    working["sentiment"] = UNKNOWN_VALUE
    working["shares"] = 0
    working["comments"] = working["comment_count"]
    working["total_engagement"] = working["likes"] + working["comments"]
    working["engagement_rate"] = _safe_ratio(working["total_engagement"], working["views"])
    working["trend_lag_days"] = (working["trending_date"] - working["publish_time"]).dt.days.fillna(0).astype("int64")
    working["content_text"] = (working["title"] + " " + working["tags"].replace("[None]", "").replace("[none]", "")).str.strip()
    working = _add_date_parts(working, "trending_date")
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "video_id",
            "title",
            "publish_time",
            "trending_date",
            "channel_title",
            "category_id",
            "tags",
            "primary_topic",
            "tag_count",
            "views",
            "likes",
            "dislikes",
            "comments",
            "total_engagement",
            "engagement_rate",
            "trend_lag_days",
            "comments_disabled",
            "ratings_disabled",
            "title_word_count",
            "description_length",
            "description_preview",
            "platform",
            "region",
            "source_dataset",
            "record_type",
            "year",
            "month",
            "year_month",
            "year_week",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["video_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["publish_time"],
            "trend_date": working["trending_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_youtube_publish_country_snapshot(
    df: pd.DataFrame,
    row_offset: int = 0,
    *,
    dataset_name: str,
    source_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()

    working["video_id"] = _normalize_whitespace(working["video_id"])
    working["title"] = _normalize_whitespace(working["title"])
    working["publish_date"] = _parse_slash_date(working["publish_date"])
    working["trending_date"] = _parse_classic_youtube_trending_date(working["trending_date"])
    working["channel_title"] = _normalize_whitespace(working["channel_title"]).replace("", UNKNOWN_VALUE)
    working["category_id"] = _to_int_series(working["category_id"])
    working["publish_country"] = _normalize_whitespace(working["publish_country"]).str.upper()
    working["region"] = working["publish_country"].map(COUNTRY_CODE_REGION_MAP).fillna(
        _normalize_title_series(working["publish_country"])
    )
    working["published_day_of_week"] = _normalize_title_series(working["published_day_of_week"])
    working["time_frame"] = _normalize_whitespace(working["time_frame"])
    working["tags"] = _normalize_whitespace(working["tags"])
    working["description_preview"] = ""
    working["description_length"] = 0

    for metric in ["views", "likes", "dislikes", "comment_count"]:
        working[metric] = _to_int_series(working[metric])

    working["comments_disabled"] = working["comments_disabled"].fillna(False).astype(bool)
    working["ratings_disabled"] = working["ratings_disabled"].fillna(False).astype(bool)
    working["title_word_count"] = working["title"].str.split().str.len().fillna(0).astype("int64")
    working["tag_count"] = working["tags"].replace("[none]", "").replace("[None]", "").str.count(r"\|").add(1)
    working.loc[working["tags"].isin(["", "[None]", "[none]"]), "tag_count"] = 0
    working["tag_count"] = working["tag_count"].astype("int64")
    working["primary_topic"] = _extract_primary_topic_from_tags(working["tags"])
    working["platform"] = "YouTube"
    working["record_type"] = "video"
    working["content_type"] = "Trending Video"
    working["source_dataset"] = dataset_name
    working["source_file"] = source_file
    working["publisher"] = working["channel_title"]
    working["sentiment"] = UNKNOWN_VALUE
    working["shares"] = 0
    working["comments"] = working["comment_count"]
    working["total_engagement"] = working["likes"] + working["comments"]
    working["engagement_rate"] = _safe_ratio(working["total_engagement"], working["views"])
    working["trend_lag_days"] = (working["trending_date"] - working["publish_date"]).dt.days.fillna(0).astype("int64")
    working["content_text"] = (working["title"] + " " + working["tags"].replace("[None]", "").replace("[none]", "")).str.strip()
    working = _add_date_parts(working, "trending_date")
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "video_id",
            "title",
            "publish_date",
            "trending_date",
            "channel_title",
            "category_id",
            "publish_country",
            "published_day_of_week",
            "time_frame",
            "tags",
            "primary_topic",
            "tag_count",
            "views",
            "likes",
            "dislikes",
            "comments",
            "total_engagement",
            "engagement_rate",
            "trend_lag_days",
            "comments_disabled",
            "ratings_disabled",
            "title_word_count",
            "platform",
            "region",
            "source_dataset",
            "record_type",
            "year",
            "month",
            "year_month",
            "year_week",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["video_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["publish_date"],
            "trend_date": working["trending_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_selected_tweets(df: pd.DataFrame, row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.rename(columns={"textID": "text_id", "selected_text": "selected_text"}).copy()

    working["text_id"] = _normalize_whitespace(working["text_id"])
    working["text"] = _normalize_whitespace(working["text"])
    working["selected_text"] = _normalize_whitespace(working["selected_text"])
    working["sentiment"] = _normalize_lower_series(working["sentiment"])
    working["primary_topic"] = _extract_primary_topic_from_text(working["text"])
    working["platform"] = "Twitter"
    working["record_type"] = "tweet"
    working["content_type"] = "Tweet"
    working["region"] = UNKNOWN_VALUE
    working["publisher"] = UNKNOWN_VALUE
    working["source_dataset"] = "tweet_sentiment_selection"
    working["source_file"] = "Tweets.csv"
    working["published_at"] = pd.NaT
    working["trend_date"] = pd.NaT
    working["year"] = 0
    working["month"] = 0
    working["year_month"] = UNKNOWN_VALUE
    working["year_week"] = UNKNOWN_VALUE
    working["views"] = 0
    working["likes"] = 0
    working["shares"] = 0
    working["comments"] = 0
    working["dislikes"] = 0
    working["total_engagement"] = 0
    working["engagement_rate"] = 0.0
    working["trend_lag_days"] = 0
    working["content_text"] = working["text"]
    working = _add_text_features(working, "content_text")
    working["selected_text_length"] = working["selected_text"].str.len().astype("int64")
    denominator = working["text_length"].replace(0, np.nan)
    working["selected_text_ratio"] = _safe_ratio(working["selected_text_length"], denominator)

    processed = working[
        [
            "text_id",
            "text",
            "selected_text",
            "sentiment",
            "primary_topic",
            "text_length",
            "word_count",
            "hashtag_count",
            "mention_count",
            "link_count",
            "question_count",
            "exclamation_count",
            "selected_text_length",
            "selected_text_ratio",
            "platform",
            "record_type",
            "source_dataset",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["text_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["published_at"],
            "trend_date": working["trend_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_twitter_train(df: pd.DataFrame, row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.rename(columns={"sentence": "text"}).copy()

    working["record_id"] = [f"tw_train_{row_offset + index + 1:08d}" for index in range(len(working))]
    working["text"] = _normalize_whitespace(working["text"])
    sentiment_map = {0: "negative", 1: "positive"}
    working["sentiment"] = pd.to_numeric(working["sentiment"], errors="coerce").map(sentiment_map).fillna(UNKNOWN_VALUE)
    working["primary_topic"] = _extract_primary_topic_from_text(working["text"])
    working["platform"] = "Twitter"
    working["record_type"] = "tweet"
    working["content_type"] = "Tweet"
    working["region"] = UNKNOWN_VALUE
    working["publisher"] = UNKNOWN_VALUE
    working["source_dataset"] = "twitter_train_binary_sentiment"
    working["source_file"] = "twitter_train.csv"
    working["published_at"] = pd.NaT
    working["trend_date"] = pd.NaT
    working["year"] = 0
    working["month"] = 0
    working["year_month"] = UNKNOWN_VALUE
    working["year_week"] = UNKNOWN_VALUE
    working["views"] = 0
    working["likes"] = 0
    working["shares"] = 0
    working["comments"] = 0
    working["dislikes"] = 0
    working["total_engagement"] = 0
    working["engagement_rate"] = 0.0
    working["trend_lag_days"] = 0
    working["content_text"] = working["text"]
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "record_id",
            "text",
            "sentiment",
            "primary_topic",
            "text_length",
            "word_count",
            "hashtag_count",
            "mention_count",
            "link_count",
            "question_count",
            "exclamation_count",
            "platform",
            "record_type",
            "source_dataset",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["record_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["published_at"],
            "trend_date": working["trend_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


def transform_youtube_comments(df: pd.DataFrame, row_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.rename(columns={"Comment": "comment", "Sentiment": "sentiment"}).copy()

    working["record_id"] = [f"yt_comment_{row_offset + index + 1:08d}" for index in range(len(working))]
    working["comment"] = _normalize_whitespace(working["comment"])
    working["sentiment"] = _normalize_lower_series(working["sentiment"])
    working["primary_topic"] = _extract_primary_topic_from_text(working["comment"])
    working["platform"] = "YouTube"
    working["record_type"] = "comment"
    working["content_type"] = "Comment"
    working["region"] = UNKNOWN_VALUE
    working["publisher"] = UNKNOWN_VALUE
    working["source_dataset"] = "youtube_comments_sentiment"
    working["source_file"] = "YoutubeCommentsDataSet.csv"
    working["published_at"] = pd.NaT
    working["trend_date"] = pd.NaT
    working["year"] = 0
    working["month"] = 0
    working["year_month"] = UNKNOWN_VALUE
    working["year_week"] = UNKNOWN_VALUE
    working["views"] = 0
    working["likes"] = 0
    working["shares"] = 0
    working["comments"] = 0
    working["dislikes"] = 0
    working["total_engagement"] = 0
    working["engagement_rate"] = 0.0
    working["trend_lag_days"] = 0
    working["content_text"] = working["comment"]
    working = _add_text_features(working, "content_text")

    processed = working[
        [
            "record_id",
            "comment",
            "sentiment",
            "primary_topic",
            "text_length",
            "word_count",
            "hashtag_count",
            "mention_count",
            "link_count",
            "question_count",
            "exclamation_count",
            "platform",
            "record_type",
            "source_dataset",
        ]
    ].copy()

    canonical_source = pd.DataFrame(
        {
            "record_id": working["record_id"],
            "source_dataset": working["source_dataset"],
            "source_file": working["source_file"],
            "platform": working["platform"],
            "record_type": working["record_type"],
            "content_type": working["content_type"],
            "publisher": working["publisher"],
            "published_at": working["published_at"],
            "trend_date": working["trend_date"],
            "year": working["year"],
            "month": working["month"],
            "year_month": working["year_month"],
            "year_week": working["year_week"],
            "region": working["region"],
            "primary_topic": working["primary_topic"],
            "sentiment": working["sentiment"],
            "content_text": working["content_text"],
            "views": working["views"],
            "likes": working["likes"],
            "shares": working["shares"],
            "comments": working["comments"],
            "dislikes": working["dislikes"],
            "total_engagement": working["total_engagement"],
            "engagement_rate": working["engagement_rate"],
            "trend_lag_days": working["trend_lag_days"],
            "text_length": working["text_length"],
            "word_count": working["word_count"],
            "hashtag_count": working["hashtag_count"],
            "mention_count": working["mention_count"],
            "link_count": working["link_count"],
            "question_count": working["question_count"],
            "exclamation_count": working["exclamation_count"],
        }
    )
    return processed, _build_canonical_frame(canonical_source)


DATASET_SPECS = [
    DatasetSpec(
        name="viral_social_media_trends",
        title="Viral Social Media Trends",
        source_file="Cleaned_Viral_Social_Media_Trends.csv",
        processed_file="viral_social_media_trends_processed.csv",
        notebook_file="01_viral_social_media_trends.ipynb",
        description="Platform-level post performance with dates, regions, and engagement metrics.",
        processor=transform_viral_social_media_trends,
        read_csv_kwargs={
            "usecols": [
                "Post_ID",
                "Post_Date",
                "Platform",
                "Hashtag",
                "Content_Type",
                "Region",
                "Views",
                "Likes",
                "Shares",
                "Comments",
                "Engagement_Level",
            ],
            "dtype": {
                "Post_ID": "string",
                "Post_Date": "string",
                "Platform": "string",
                "Hashtag": "string",
                "Content_Type": "string",
                "Region": "string",
                "Views": "string",
                "Likes": "string",
                "Shares": "string",
                "Comments": "string",
                "Engagement_Level": "string",
            },
            "na_filter": False,
        },
    ),
    DatasetSpec(
        name="indian_youtube_trending",
        title="Indian YouTube Trending Videos",
        source_file="IN_youtube_trending_data.csv",
        processed_file="indian_youtube_trending_processed.csv",
        notebook_file="02_indian_youtube_trending.ipynb",
        description="Trending-video behavior with views, likes, comments, tags, and trend lag.",
        processor=transform_indian_youtube_trending,
        chunk_size=100_000,
        read_csv_kwargs={
            "usecols": [
                "video_id",
                "title",
                "publishedAt",
                "channelId",
                "channelTitle",
                "categoryId",
                "trending_date",
                "tags",
                "view_count",
                "likes",
                "dislikes",
                "comment_count",
                "comments_disabled",
                "ratings_disabled",
                "description",
            ],
            "dtype": {
                "video_id": "string",
                "title": "string",
                "publishedAt": "string",
                "channelId": "string",
                "channelTitle": "string",
                "categoryId": "string",
                "trending_date": "string",
                "tags": "string",
                "view_count": "string",
                "likes": "string",
                "dislikes": "string",
                "comment_count": "string",
                "comments_disabled": "boolean",
                "ratings_disabled": "boolean",
                "description": "string",
            },
        },
    ),
    DatasetSpec(
        name="tweet_sentiment_selection",
        title="Tweet Sentiment Selection",
        source_file="Tweets.csv",
        processed_file="tweet_sentiment_selection_processed.csv",
        notebook_file="03_tweet_sentiment_selection.ipynb",
        description="Short-form tweet text with curated sentiment and selected text spans.",
        processor=transform_selected_tweets,
        read_csv_kwargs={
            "usecols": ["textID", "text", "selected_text", "sentiment"],
            "dtype": {
                "textID": "string",
                "text": "string",
                "selected_text": "string",
                "sentiment": "string",
            },
            "na_filter": False,
        },
    ),
    DatasetSpec(
        name="twitter_train_binary_sentiment",
        title="Twitter Binary Sentiment Training Data",
        source_file="twitter_train.csv",
        processed_file="twitter_train_binary_sentiment_processed.csv",
        notebook_file="04_twitter_binary_sentiment.ipynb",
        description="Large-scale binary sentiment dataset for broader text and mood coverage.",
        processor=transform_twitter_train,
        chunk_size=200_000,
        read_csv_kwargs={
            "usecols": ["sentence", "sentiment"],
            "dtype": {
                "sentence": "string",
                "sentiment": "Int8",
            },
        },
    ),
    DatasetSpec(
        name="youtube_comments_sentiment",
        title="YouTube Comments Sentiment",
        source_file="YoutubeCommentsDataSet.csv",
        processed_file="youtube_comments_sentiment_processed.csv",
        notebook_file="05_youtube_comments_sentiment.ipynb",
        description="Comment-level audience reactions and sentiment from YouTube discussions.",
        processor=transform_youtube_comments,
        read_csv_kwargs={
            "usecols": ["Comment", "Sentiment"],
            "dtype": {
                "Comment": "string",
                "Sentiment": "string",
            },
            "na_filter": False,
        },
    ),
]


def _build_classic_youtube_spec(country_code: str, notebook_index: int) -> DatasetSpec:
    region = COUNTRY_CODE_REGION_MAP[country_code]
    slug = region.lower().replace(" ", "_")
    source_file = f"{country_code}videos.csv"
    dataset_name = f"youtube_trending_{slug}"
    return DatasetSpec(
        name=dataset_name,
        title=f"{region} YouTube Trending Videos",
        source_file=source_file,
        processed_file=f"{dataset_name}_processed.csv",
        notebook_file=f"{notebook_index:02d}_{dataset_name}.ipynb",
        description=f"Regional YouTube trending-video behavior for {region} with views, likes, comments, and trend lag.",
        processor=partial(
            transform_regional_youtube_trending,
            dataset_name=dataset_name,
            region=region,
            source_file=source_file,
        ),
        chunk_size=100_000,
        read_csv_kwargs={
            "usecols": [
                "video_id",
                "trending_date",
                "title",
                "channel_title",
                "category_id",
                "publish_time",
                "tags",
                "views",
                "likes",
                "dislikes",
                "comment_count",
                "comments_disabled",
                "ratings_disabled",
                "description",
            ],
            "dtype": {
                "video_id": "string",
                "trending_date": "string",
                "title": "string",
                "channel_title": "string",
                "category_id": "string",
                "publish_time": "string",
                "tags": "string",
                "views": "string",
                "likes": "string",
                "dislikes": "string",
                "comment_count": "string",
                "comments_disabled": "boolean",
                "ratings_disabled": "boolean",
                "description": "string",
            },
        },
    )


EXTRA_DATASET_SPECS = [
    _build_classic_youtube_spec("US", 6),
    _build_classic_youtube_spec("GB", 7),
    _build_classic_youtube_spec("CA", 8),
    _build_classic_youtube_spec("DE", 9),
    _build_classic_youtube_spec("FR", 10),
    _build_classic_youtube_spec("JP", 11),
    _build_classic_youtube_spec("KR", 12),
    _build_classic_youtube_spec("MX", 13),
    _build_classic_youtube_spec("RU", 14),
    _build_classic_youtube_spec("IN", 15),
    DatasetSpec(
        name="youtube_publish_country_snapshot",
        title="Cross-Country YouTube Trending Snapshot",
        source_file="youtube.csv",
        processed_file="youtube_publish_country_snapshot_processed.csv",
        notebook_file="16_youtube_publish_country_snapshot.ipynb",
        description="Compact YouTube trending snapshot with publish-country coverage, day-of-week, and time-frame metadata.",
        processor=partial(
            transform_youtube_publish_country_snapshot,
            dataset_name="youtube_publish_country_snapshot",
            source_file="youtube.csv",
        ),
        chunk_size=100_000,
        read_csv_kwargs={
            "usecols": [
                "video_id",
                "trending_date",
                "title",
                "channel_title",
                "category_id",
                "publish_date",
                "time_frame",
                "published_day_of_week",
                "publish_country",
                "tags",
                "views",
                "likes",
                "dislikes",
                "comment_count",
                "comments_disabled",
                "ratings_disabled",
            ],
            "dtype": {
                "video_id": "string",
                "trending_date": "string",
                "title": "string",
                "channel_title": "string",
                "category_id": "string",
                "publish_date": "string",
                "time_frame": "string",
                "published_day_of_week": "string",
                "publish_country": "string",
                "tags": "string",
                "views": "string",
                "likes": "string",
                "dislikes": "string",
                "comment_count": "string",
                "comments_disabled": "boolean",
                "ratings_disabled": "boolean",
            },
        },
    ),
]

DATASET_SPECS = DATASET_SPECS + EXTRA_DATASET_SPECS


def available_dataset_names() -> list[str]:
    return [spec.name for spec in DATASET_SPECS]


def _normalize_requested_dataset_names(dataset_names: Iterable[str] | None) -> list[str]:
    if dataset_names is None:
        return []

    normalized: list[str] = []
    for name in dataset_names:
        parts = [part.strip() for part in str(name).split(",")]
        normalized.extend(part for part in parts if part)
    return normalized


def select_dataset_specs(dataset_names: Iterable[str] | None = None) -> list[DatasetSpec]:
    requested_names = _normalize_requested_dataset_names(dataset_names)
    if not requested_names:
        return list(DATASET_SPECS)

    lookup = {spec.name: spec for spec in DATASET_SPECS}
    unknown = [name for name in requested_names if name not in lookup]
    if unknown:
        available = ", ".join(available_dataset_names())
        unknown_list = ", ".join(unknown)
        raise ValueError(f"Unknown dataset name(s): {unknown_list}. Available datasets: {available}")

    return [lookup[name] for name in requested_names]


def _create_read_iterator(source_path: Path, spec: DatasetSpec) -> Iterable[pd.DataFrame]:
    read_csv_kwargs = dict(spec.read_csv_kwargs)
    read_csv_kwargs.setdefault("memory_map", True)
    read_csv_kwargs.setdefault("encoding_errors", "replace")

    if spec.chunk_size:
        return pd.read_csv(source_path, chunksize=spec.chunk_size, **read_csv_kwargs)
    return [pd.read_csv(source_path, **read_csv_kwargs)]


def _print_progress(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _append_csv(df: pd.DataFrame, output_path: Path, include_header: bool) -> None:
    df.to_csv(output_path, mode="a", index=False, header=include_header)


def _aggregate_chunk(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_columns + SUMMARY_METRICS)

    grouped = (
        df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            record_count=("record_id", "count"),
            total_views=("views", "sum"),
            total_likes=("likes", "sum"),
            total_shares=("shares", "sum"),
            total_comments=("comments", "sum"),
            total_dislikes=("dislikes", "sum"),
            total_engagement=("total_engagement", "sum"),
            total_text_length=("text_length", "sum"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return grouped


def _merge_aggregate_frames(current: pd.DataFrame | None, incoming: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if current is None or current.empty:
        return incoming
    if incoming.empty:
        return current
    combined = pd.concat([current, incoming], ignore_index=True)
    return combined.groupby(group_columns, dropna=False, as_index=False)[SUMMARY_METRICS].sum()


def _finalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    finalized = df.copy()
    finalized["avg_text_length"] = _safe_ratio(finalized["total_text_length"], finalized["record_count"])
    finalized["engagement_rate"] = _safe_ratio(finalized["total_engagement"], finalized["total_views"])
    sort_columns = list(finalized.columns[: len(finalized.columns) - len(SUMMARY_METRICS) - 2])
    return finalized.sort_values(sort_columns).reset_index(drop=True)


def _metadata_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.head(3).fillna("").astype(str).to_dict(orient="records")


def _cleanup_previous_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "social_media_analysis_ready.csv",
        "dashboard_monthly_metrics.csv",
        "dashboard_source_metrics.csv",
        "dashboard_topic_metrics.csv",
        "dashboard_topic_timeline_metrics.csv",
        "dataset_catalog.json",
    ]:
        target = output_dir / file_name
        if target.exists():
            try:
                target.unlink()
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot refresh '{target}'. Close any app that is using this file and run the preprocessing step again."
                ) from exc
    for spec in DATASET_SPECS:
        target = output_dir / spec.processed_file
        if target.exists():
            try:
                target.unlink()
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot refresh '{target}'. Close any app that is using this file and run the preprocessing step again."
                ) from exc


def _build_notebook_cells(dataset_title: str, dataset_name: str, processed_file: str, date_columns: list[str]) -> list[dict[str, Any]]:
    parse_dates_literal = repr(date_columns)
    setup_code = f"""from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "dataset").exists() and (candidate / "src").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PREPROCESS_DIR = PROJECT_ROOT / "dataset" / "processed_data"
DATASET_FILE = PREPROCESS_DIR / "{processed_file}"
CATALOG_PATH = PREPROCESS_DIR / "dataset_catalog.json"

with CATALOG_PATH.open("r", encoding="utf-8") as handle:
    catalog = json.load(handle)

metadata = next(item for item in catalog["datasets"] if item["name"] == "{dataset_name}")
parse_dates = {parse_dates_literal}

df = pd.read_csv(DATASET_FILE, parse_dates=parse_dates or None)
df.head()
"""

    metadata_code = """pd.DataFrame(
    {
        "Metric": [
            "Source dataset",
            "Source file",
            "Processed file",
            "Processed rows",
            "Rows loaded in notebook",
            "Columns in processed file",
            "Platform coverage",
            "Date columns",
        ],
        "Value": [
            metadata["title"],
            metadata["source_file"],
            metadata["processed_file"],
            metadata["processed_rows"],
            len(df),
            metadata["column_count"],
            ", ".join(metadata["platforms"]) or "Unknown",
            ", ".join(metadata["date_columns"]) or "None",
        ],
    }
)
"""

    quality_code = """quality = (
    pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_ratio": df.isna().mean().round(4),
            "unique_values": df.nunique(dropna=False),
        }
    )
    .sort_values(["missing_ratio", "unique_values"], ascending=[False, False])
)
quality.head(15)
"""

    categorical_code = """candidate_columns = [col for col in ["platform", "record_type", "region", "sentiment", "content_type", "primary_topic"] if col in df.columns]
display_frames = []
for column in candidate_columns:
    counts = df[column].fillna("Unknown").astype(str).value_counts().head(10)
    display_frames.append(pd.DataFrame({"column": column, "value": counts.index, "count": counts.values}))
pd.concat(display_frames, ignore_index=True) if display_frames else pd.DataFrame(columns=["column", "value", "count"])
"""

    time_series_code = """time_columns = [col for col in ["post_date", "published_at", "trending_date"] if col in df.columns]
if time_columns:
    time_col = time_columns[0]
    trend = (
        df.dropna(subset=[time_col])
        .assign(year_month=lambda frame: frame[time_col].dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)
        .size()
        .rename(columns={"size": "records"})
    )
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=trend, x="year_month", y="records", marker="o", linewidth=2.2, color="#D96F32")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Monthly record volume based on {time_col}")
    plt.xlabel("Month")
    plt.ylabel("Records")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
else:
    print("No date column is available in this processed dataset.")
"""

    numeric_code = """numeric_candidates = [col for col in ["views", "likes", "shares", "comments", "dislikes", "total_engagement", "engagement_rate", "text_length", "word_count"] if col in df.columns]
if numeric_candidates:
    summary = df[numeric_candidates].describe().T.sort_values("mean", ascending=False)
    display(summary)
    if len(numeric_candidates) >= 2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_candidates].corr(numeric_only=True), annot=True, cmap="YlOrBr", fmt=".2f")
        plt.title("Correlation snapshot")
        plt.tight_layout()
        plt.show()
else:
    print("No numeric features were found in this processed dataset.")
"""

    topic_code = """if "primary_topic" in df.columns:
    topics = (
        df.loc[df["primary_topic"].fillna("no_explicit_topic") != "no_explicit_topic", "primary_topic"]
        .astype(str)
        .value_counts()
        .head(12)
        .sort_values()
    )
    if not topics.empty:
        plt.figure(figsize=(10, 6))
        topics.plot(kind="barh", color="#254D6E")
        plt.title("Most visible explicit topics")
        plt.xlabel("Records")
        plt.ylabel("Topic")
        plt.tight_layout()
        plt.show()
    else:
        print("The dataset does not expose explicit hashtags or tags often enough for a topic chart.")
"""

    return [
        {"cell_type": "markdown", "source": f"# {dataset_title}\n\nProfile notebook for `{processed_file}`.\n"},
        {"cell_type": "markdown", "source": "This notebook is generated from the preprocessing metadata so it stays aligned with the latest cleaned dataset and can be rerun after each pipeline refresh.\n"},
        {"cell_type": "code", "source": setup_code},
        {"cell_type": "code", "source": metadata_code},
        {"cell_type": "code", "source": quality_code},
        {"cell_type": "code", "source": categorical_code},
        {"cell_type": "code", "source": time_series_code},
        {"cell_type": "code", "source": numeric_code},
        {"cell_type": "code", "source": topic_code},
    ]


def _build_multisource_notebook_cells(notebook_title: str, notebook_summary: str) -> list[dict[str, Any]]:
    setup_code = """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "dataset").exists() and (candidate / "src").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PREPROCESS_DIR = PROJECT_ROOT / "dataset" / "processed_data"
UNION_FILE = PREPROCESS_DIR / "social_media_analysis_ready.csv"
SOURCE_METRICS_FILE = PREPROCESS_DIR / "dashboard_source_metrics.csv"
MONTHLY_METRICS_FILE = PREPROCESS_DIR / "dashboard_monthly_metrics.csv"
TOPIC_METRICS_FILE = PREPROCESS_DIR / "dashboard_topic_metrics.csv"
CATALOG_PATH = PREPROCESS_DIR / "dataset_catalog.json"

with CATALOG_PATH.open("r", encoding="utf-8") as handle:
    catalog = json.load(handle)

source_metrics = pd.read_csv(SOURCE_METRICS_FILE)
monthly_metrics = pd.read_csv(MONTHLY_METRICS_FILE)
topic_metrics = pd.read_csv(TOPIC_METRICS_FILE)
union_df = pd.read_csv(UNION_FILE, parse_dates=["published_at", "trend_date"])

len(union_df), source_metrics.head()
"""

    overview_code = """pd.DataFrame(catalog["datasets"])[["title", "processed_rows", "platforms", "date_columns", "processed_file"]]
"""

    source_code = """platform_mix = (
    source_metrics.groupby(["platform"], as_index=False)[["record_count", "total_engagement"]]
    .sum()
    .sort_values("record_count", ascending=False)
)

plt.figure(figsize=(10, 5))
sns.barplot(data=platform_mix, x="record_count", y="platform", color="#254D6E")
plt.title("Record coverage by platform")
plt.xlabel("Records")
plt.ylabel("Platform")
plt.tight_layout()
plt.show()
"""

    monthly_code = """dated_monthly = monthly_metrics.loc[monthly_metrics["year_month"] != "Unknown"].copy()
trend = dated_monthly.groupby("year_month", as_index=False)[["record_count", "total_engagement"]].sum()

plt.figure(figsize=(12, 5))
sns.lineplot(data=trend, x="year_month", y="record_count", marker="o", color="#D96F32", linewidth=2.2)
plt.xticks(rotation=45, ha="right")
plt.title("Cross-source activity trend")
plt.xlabel("Month")
plt.ylabel("Records")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""

    sentiment_code = """sentiment_mix = (
    source_metrics.loc[source_metrics["sentiment"] != "Unknown"]
    .groupby("sentiment", as_index=False)["record_count"]
    .sum()
    .sort_values("record_count", ascending=False)
)

if not sentiment_mix.empty:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=sentiment_mix, x="sentiment", y="record_count", palette="crest")
    plt.title("Sentiment coverage across text datasets")
    plt.xlabel("Sentiment")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.show()
"""

    topic_code = """top_topics = topic_metrics.sort_values(["record_count", "total_engagement"], ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_topics, x="record_count", y="primary_topic", hue="platform")
plt.title("Most visible explicit topics across tagged datasets")
plt.xlabel("Records")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()
"""

    return [
        {"cell_type": "markdown", "source": f"# {notebook_title}\n\n{notebook_summary}\n"},
        {"cell_type": "code", "source": setup_code},
        {"cell_type": "code", "source": overview_code},
        {"cell_type": "code", "source": source_code},
        {"cell_type": "code", "source": monthly_code},
        {"cell_type": "code", "source": sentiment_code},
        {"cell_type": "code", "source": topic_code},
    ]


def _build_preprocessing_notebook_cells() -> list[dict[str, Any]]:
    setup_code = """from pathlib import Path
import json
import sys

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "dataset").exists() and (candidate / "src").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import DATASET_SPECS

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_data"
RAW_DIR = PROJECT_ROOT / "dataset" / "raw_data"
CATALOG_PATH = PROCESSED_DIR / "dataset_catalog.json"

with CATALOG_PATH.open("r", encoding="utf-8") as handle:
    catalog = json.load(handle)

pd.DataFrame(catalog["datasets"])[["title", "source_file", "processed_file", "processed_rows"]]
"""

    supported_code = """pd.DataFrame(
    {
        "dataset_name": [spec.name for spec in DATASET_SPECS],
        "source_file": [spec.source_file for spec in DATASET_SPECS],
        "processed_file": [spec.processed_file for spec in DATASET_SPECS],
        "chunked_reader": ["yes" if spec.chunk_size else "no" for spec in DATASET_SPECS],
    }
)
"""

    file_code = """pd.concat(
    [
        pd.Series(sorted(path.name for path in RAW_DIR.glob("*.csv")), name="raw_files_present"),
        pd.Series(sorted(path.name for path in PROCESSED_DIR.glob("*_processed.csv")), name="processed_outputs_present"),
    ],
    axis=1,
)
"""

    summary_code = """summary_assets = sorted(
    path.name for path in PROCESSED_DIR.glob("*")
    if path.is_file() and not path.name.endswith("_processed.csv")
)
pd.DataFrame({"stored_assets": summary_assets})
"""

    return [
        {
            "cell_type": "markdown",
            "source": "# Preprocessing\n\nThis notebook documents the project pipeline, supported raw sources, and stored processed outputs under the academic folder structure.\n",
        },
        {"cell_type": "code", "source": setup_code},
        {"cell_type": "code", "source": supported_code},
        {"cell_type": "code", "source": file_code},
        {"cell_type": "code", "source": summary_code},
    ]


def _write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    import nbformat as nbf

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            notebook["cells"].append(nbf.v4.new_markdown_cell(cell["source"]))
        else:
            notebook["cells"].append(nbf.v4.new_code_cell(cell["source"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


def generate_analysis_notebooks(catalog_path: Path | str = PREPROCESS_DIR / "dataset_catalog.json") -> list[Path]:
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Dataset catalog not found at {catalog_path}")

    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    DATASET_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for notebook_path in DATASET_NOTEBOOK_DIR.glob("*.ipynb"):
        notebook_path.unlink()

    created_paths: list[Path] = []
    for dataset in catalog["datasets"]:
        notebook_path = DATASET_NOTEBOOK_DIR / dataset["notebook_file"]
        cells = _build_notebook_cells(
            dataset_title=dataset["title"],
            dataset_name=dataset["name"],
            processed_file=dataset["processed_file"],
            date_columns=dataset["date_columns"],
        )
        _write_notebook(notebook_path, cells)
        created_paths.append(notebook_path)

    data_understanding_cells = _build_multisource_notebook_cells(
        "Data Understanding",
        "Cross-dataset understanding notebook for the processed social media analytics layer.",
    )
    data_understanding_path = NOTEBOOKS_DIR / "data_understanding.ipynb"
    _write_notebook(data_understanding_path, data_understanding_cells)
    created_paths.append(data_understanding_path)

    visualization_cells = _build_multisource_notebook_cells(
        "Visualization",
        "Visualization notebook for platform, topic, and time-based analysis across the processed datasets.",
    )
    visualization_path = NOTEBOOKS_DIR / "visualization.ipynb"
    _write_notebook(visualization_path, visualization_cells)
    created_paths.append(visualization_path)

    preprocessing_cells = _build_preprocessing_notebook_cells()
    preprocessing_path = NOTEBOOKS_DIR / "preprocessing.ipynb"
    _write_notebook(preprocessing_path, preprocessing_cells)
    created_paths.append(preprocessing_path)
    return created_paths


def build_analysis_assets(
    raw_dir: Path | str = RAW_DATA_DIR,
    output_dir: Path | str = PREPROCESS_DIR,
    generate_notebooks: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    _cleanup_previous_outputs(output_dir)
    selected_specs = [spec for spec in DATASET_SPECS if (raw_dir / spec.source_file).exists()]
    skipped_specs = [spec for spec in DATASET_SPECS if spec not in selected_specs]
    if not selected_specs:
        raise FileNotFoundError(f"No supported raw datasets were found in: {raw_dir}")
    if skipped_specs:
        skipped_names = ", ".join(spec.source_file for spec in skipped_specs)
        _print_progress(f"Skipping missing supported datasets: {skipped_names}", verbose=verbose)

    combined_path = output_dir / "social_media_analysis_ready.csv"
    monthly_summary: pd.DataFrame | None = None
    source_summary: pd.DataFrame | None = None
    topic_summary: pd.DataFrame | None = None
    topic_timeline_summary: pd.DataFrame | None = None
    combined_header_written = False

    catalog: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_data_dir": str(raw_dir),
        "preprocess_dir": str(output_dir),
        "datasets": [],
    }

    monthly_group_columns = ["source_dataset", "platform", "record_type", "region", "sentiment", "year_month"]
    source_group_columns = ["source_dataset", "platform", "record_type", "region", "sentiment"]
    topic_group_columns = ["source_dataset", "platform", "record_type", "region", "sentiment", "primary_topic"]
    topic_timeline_group_columns = [
        "source_dataset",
        "platform",
        "record_type",
        "region",
        "sentiment",
        "year_month",
        "primary_topic",
    ]

    total_specs = len(selected_specs)
    for dataset_index, spec in enumerate(selected_specs, start=1):
        source_path = raw_dir / spec.source_file

        _print_progress(
            f"[{dataset_index}/{total_specs}] Processing {spec.title} from '{spec.source_file}'",
            verbose=verbose,
        )

        processed_output_path = output_dir / spec.processed_file
        processed_header_written = False
        row_offset = 0
        raw_rows = 0
        processed_rows = 0
        platforms_seen: set[str] = set()
        sentiments_seen: set[str] = set()
        date_columns_seen: set[str] = set()
        preview_records: list[dict[str, Any]] = []
        processed_columns: list[str] = []

        read_iterator = _create_read_iterator(source_path, spec)

        for chunk_index, chunk in enumerate(read_iterator, start=1):
            raw_rows += len(chunk)
            processed_df, canonical_df = spec.processor(chunk, row_offset=row_offset)
            row_offset += len(chunk)
            if processed_df.empty:
                continue

            _append_csv(processed_df, processed_output_path, include_header=not processed_header_written)
            processed_header_written = True

            _append_csv(canonical_df, combined_path, include_header=not combined_header_written)
            combined_header_written = True

            monthly_chunk = _aggregate_chunk(canonical_df, monthly_group_columns)
            source_chunk = _aggregate_chunk(canonical_df, source_group_columns)
            topic_candidates = canonical_df.loc[
                canonical_df["primary_topic"].fillna(NO_TOPIC_VALUE).ne(NO_TOPIC_VALUE)
            ].copy()
            topic_chunk = _aggregate_chunk(topic_candidates, topic_group_columns)
            dated_topic_candidates = topic_candidates.loc[
                topic_candidates["year_month"].fillna(UNKNOWN_VALUE).ne(UNKNOWN_VALUE)
            ].copy()
            topic_timeline_chunk = _aggregate_chunk(dated_topic_candidates, topic_timeline_group_columns)

            monthly_summary = _merge_aggregate_frames(monthly_summary, monthly_chunk, monthly_group_columns)
            source_summary = _merge_aggregate_frames(source_summary, source_chunk, source_group_columns)
            topic_summary = _merge_aggregate_frames(topic_summary, topic_chunk, topic_group_columns)
            topic_timeline_summary = _merge_aggregate_frames(
                topic_timeline_summary,
                topic_timeline_chunk,
                topic_timeline_group_columns,
            )

            processed_rows += len(processed_df)
            processed_columns = processed_df.columns.tolist()
            if "platform" in processed_df.columns:
                platforms_seen.update(processed_df["platform"].dropna().astype(str).unique())
            if "sentiment" in processed_df.columns:
                sentiments_seen.update(processed_df["sentiment"].dropna().astype(str).unique())
            date_columns_seen.update([column for column in processed_df.columns if "date" in column])
            if not preview_records:
                preview_records = _metadata_preview(processed_df)

            if spec.chunk_size:
                _print_progress(
                    f"    chunk {chunk_index}: {raw_rows:,} raw rows processed",
                    verbose=verbose,
                )

        catalog["datasets"].append(
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "source_file": spec.source_file,
                "processed_file": spec.processed_file,
                "notebook_file": spec.notebook_file,
                "raw_rows": raw_rows,
                "processed_rows": processed_rows,
                "column_count": len(processed_columns),
                "columns": processed_columns,
                "platforms": sorted(platforms_seen),
                "sentiments": sorted(sentiments_seen),
                "date_columns": sorted(date_columns_seen),
                "preview": preview_records,
            }
        )
        _print_progress(
            f"    completed {spec.name}: {processed_rows:,} processed rows written to '{processed_output_path.name}'",
            verbose=verbose,
        )

    monthly_summary = _finalize_summary(monthly_summary if monthly_summary is not None else pd.DataFrame())
    source_summary = _finalize_summary(source_summary if source_summary is not None else pd.DataFrame())
    topic_summary = _finalize_summary(topic_summary if topic_summary is not None else pd.DataFrame())
    topic_timeline_summary = _finalize_summary(
        topic_timeline_summary if topic_timeline_summary is not None else pd.DataFrame()
    )

    if not topic_summary.empty:
        topic_summary = topic_summary.sort_values(
            ["source_dataset", "platform", "record_count", "total_engagement"],
            ascending=[True, True, False, False],
        )
        topic_summary["topic_rank"] = topic_summary.groupby(["source_dataset", "platform"]).cumcount() + 1
        topic_summary = topic_summary.loc[topic_summary["topic_rank"] <= 25].reset_index(drop=True)

    monthly_summary.to_csv(output_dir / "dashboard_monthly_metrics.csv", index=False)
    source_summary.to_csv(output_dir / "dashboard_source_metrics.csv", index=False)
    topic_summary.to_csv(output_dir / "dashboard_topic_metrics.csv", index=False)
    topic_timeline_summary.to_csv(output_dir / "dashboard_topic_timeline_metrics.csv", index=False)

    with (output_dir / "dataset_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)

    notebook_paths: list[Path] = []
    if generate_notebooks:
        notebook_paths = generate_analysis_notebooks(output_dir / "dataset_catalog.json")

    _print_progress(
        f"Finished preprocessing. Combined analysis file: '{combined_path.name}'",
        verbose=verbose,
    )

    return {
        "catalog": catalog,
        "combined_path": combined_path,
        "monthly_summary_path": output_dir / "dashboard_monthly_metrics.csv",
        "source_summary_path": output_dir / "dashboard_source_metrics.csv",
        "topic_summary_path": output_dir / "dashboard_topic_metrics.csv",
        "topic_timeline_summary_path": output_dir / "dashboard_topic_timeline_metrics.csv",
        "notebook_paths": notebook_paths,
    }


def ensure_preprocessed_assets(
    raw_dir: Path | str = RAW_DATA_DIR,
    preprocess_dir: Path | str = PREPROCESS_DIR,
    force: bool = False,
    generate_notebooks: bool = True,
) -> None:
    preprocess_dir = Path(preprocess_dir)
    required_files = [
        preprocess_dir / "social_media_analysis_ready.csv",
        preprocess_dir / "dashboard_monthly_metrics.csv",
        preprocess_dir / "dashboard_source_metrics.csv",
        preprocess_dir / "dashboard_topic_metrics.csv",
        preprocess_dir / "dashboard_topic_timeline_metrics.csv",
        preprocess_dir / "dataset_catalog.json",
    ]
    if force or not all(path.exists() for path in required_files):
        build_analysis_assets(raw_dir=raw_dir, output_dir=preprocess_dir, generate_notebooks=generate_notebooks)


def _required_dashboard_asset_paths(preprocess_dir: Path) -> list[Path]:
    return [
        preprocess_dir / "dataset_catalog.json",
        preprocess_dir / "dashboard_monthly_metrics.csv",
        preprocess_dir / "dashboard_source_metrics.csv",
        preprocess_dir / "dashboard_topic_metrics.csv",
        preprocess_dir / "dashboard_topic_timeline_metrics.csv",
    ]


def load_dashboard_assets(preprocess_dir: Path | str = PREPROCESS_DIR) -> dict[str, Any]:
    preprocess_dir = Path(preprocess_dir)
    catalog_path = preprocess_dir / "dataset_catalog.json"
    monthly_path = preprocess_dir / "dashboard_monthly_metrics.csv"
    source_path = preprocess_dir / "dashboard_source_metrics.csv"
    topic_path = preprocess_dir / "dashboard_topic_metrics.csv"
    topic_timeline_path = preprocess_dir / "dashboard_topic_timeline_metrics.csv"

    required_paths = _required_dashboard_asset_paths(preprocess_dir)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Stored dashboard assets are missing ({missing_list}). Run `python src/preprocessing.py` once to create them."
        )

    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    return {
        "catalog": catalog,
        "monthly_metrics": pd.read_csv(monthly_path),
        "source_metrics": pd.read_csv(source_path),
        "topic_metrics": pd.read_csv(topic_path),
        "topic_timeline_metrics": pd.read_csv(topic_timeline_path),
    }


def load_analysis_ready_sample(
    preprocess_dir: Path | str = PREPROCESS_DIR,
    nrows: int | None = None,
) -> pd.DataFrame:
    preprocess_dir = Path(preprocess_dir)
    analysis_ready_path = preprocess_dir / "social_media_analysis_ready.csv"
    if not analysis_ready_path.exists():
        raise FileNotFoundError(
            f"Stored combined analysis file is missing ({analysis_ready_path}). Run `python src/preprocessing.py` once to create it."
        )
    return pd.read_csv(
        analysis_ready_path,
        nrows=nrows,
        parse_dates=["published_at", "trend_date"],
    )


def load_processed_dataset(name_or_path: str | Path, preprocess_dir: Path | str = PREPROCESS_DIR) -> pd.DataFrame:
    preprocess_dir = Path(preprocess_dir)
    candidate = Path(name_or_path)
    dataset_path = candidate if candidate.exists() else preprocess_dir / str(name_or_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        fallback = PREPROCESS_DIR / "viral_social_media_trends_processed.csv"
        if fallback.exists():
            return pd.read_csv(fallback, parse_dates=["post_date"])
        raise FileNotFoundError(f"Dataset not found at {path}")
    raw = pd.read_csv(path)
    processed, _ = transform_viral_social_media_trends(raw)
    return processed


if __name__ == "__main__":
    build_analysis_assets(verbose=True)
