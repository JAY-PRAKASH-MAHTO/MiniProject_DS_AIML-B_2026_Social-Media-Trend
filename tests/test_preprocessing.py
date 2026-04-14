from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    build_analysis_assets,
    load_dashboard_assets,
    transform_indian_youtube_trending,
    transform_regional_youtube_trending,
    transform_twitter_train,
)


class PreprocessingPipelineTests(unittest.TestCase):
    def test_transform_twitter_train_maps_sentiment_and_generates_ids(self) -> None:
        raw = pd.DataFrame(
            {
                "sentence": ["i am sad today", "i am very happy today"],
                "sentiment": [0, 1],
            }
        )

        processed, canonical = transform_twitter_train(raw, row_offset=10)

        self.assertEqual(processed["record_id"].tolist(), ["tw_train_00000011", "tw_train_00000012"])
        self.assertEqual(processed["sentiment"].tolist(), ["negative", "positive"])
        self.assertTrue((processed["word_count"] > 0).all())
        self.assertEqual(canonical["platform"].unique().tolist(), ["Twitter"])

    def test_transform_indian_youtube_trending_derives_topic_and_engagement(self) -> None:
        raw = pd.DataFrame(
            {
                "video_id": ["abc123"],
                "title": ["New music launch"],
                "publishedAt": ["2024-01-01T00:00:00Z"],
                "channelId": ["channel-1"],
                "channelTitle": ["Demo Channel"],
                "categoryId": [10],
                "trending_date": ["2024-01-03T00:00:00Z"],
                "tags": ["music|launch|india"],
                "view_count": [1000],
                "likes": [200],
                "dislikes": [10],
                "comment_count": [50],
                "thumbnail_link": ["http://example.com/img.jpg"],
                "comments_disabled": [False],
                "ratings_disabled": [False],
                "description": ["A new music release."],
            }
        )

        processed, canonical = transform_indian_youtube_trending(raw)

        self.assertEqual(processed.loc[0, "primary_topic"], "music")
        self.assertEqual(processed.loc[0, "total_engagement"], 250)
        self.assertEqual(processed.loc[0, "trend_lag_days"], 2)
        self.assertEqual(canonical.loc[0, "platform"], "YouTube")

    def test_transform_regional_youtube_trending_parses_classic_dates_and_region(self) -> None:
        raw = pd.DataFrame(
            {
                "video_id": ["us_video_1"],
                "trending_date": ["17.14.11"],
                "title": ["Classic trending sample"],
                "channel_title": ["Demo Creator"],
                "category_id": [24],
                "publish_time": ["2017-11-13T17:13:01.000Z"],
                "tags": ["comedy|sample"],
                "views": [1500],
                "likes": [230],
                "dislikes": [7],
                "comment_count": [33],
                "comments_disabled": [False],
                "ratings_disabled": [False],
                "description": ["Sample description"],
            }
        )

        processed, canonical = transform_regional_youtube_trending(
            raw,
            dataset_name="youtube_trending_united_states",
            region="United States",
            source_file="USvideos.csv",
        )

        self.assertEqual(processed.loc[0, "primary_topic"], "comedy")
        self.assertEqual(processed.loc[0, "total_engagement"], 263)
        self.assertEqual(processed.loc[0, "region"], "United States")
        self.assertEqual(str(processed.loc[0, "year_month"]), "2017-11")
        self.assertEqual(canonical.loc[0, "source_file"], "USvideos.csv")

    def test_build_analysis_assets_creates_expected_outputs(self) -> None:
        temp_root = Path(".tmp_test_workspace")
        temp_root.mkdir(parents=True, exist_ok=True)
        base_dir = temp_root / "preprocessing_pipeline_case"
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)

        raw_dir = base_dir / "raw"
        preprocess_dir = base_dir / "preprocess"
        raw_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "Post_ID": ["post_1"],
                "Post_Date": ["2024-01-10"],
                "Platform": ["Instagram"],
                "Hashtag": ["#Launch"],
                "Content_Type": ["Reel"],
                "Region": ["India"],
                "Views": [5000],
                "Likes": [400],
                "Shares": [50],
                "Comments": [20],
                "Engagement_Level": ["High"],
            }
        ).to_csv(raw_dir / "Cleaned_Viral_Social_Media_Trends.csv", index=False)

        pd.DataFrame(
            {
                "video_id": ["video_1"],
                "title": ["Launch recap"],
                "publishedAt": ["2024-01-08T00:00:00Z"],
                "channelId": ["channel_a"],
                "channelTitle": ["News Hub"],
                "categoryId": [22],
                "trending_date": ["2024-01-09T00:00:00Z"],
                "tags": ["launch|recap"],
                "view_count": [9000],
                "likes": [600],
                "dislikes": [5],
                "comment_count": [120],
                "thumbnail_link": ["http://example.com/thumb.jpg"],
                "comments_disabled": [False],
                "ratings_disabled": [False],
                "description": ["Daily recap video"],
            }
        ).to_csv(raw_dir / "IN_youtube_trending_data.csv", index=False)

        pd.DataFrame(
            {
                "textID": ["tweet_1"],
                "text": ["#launch is going great"],
                "selected_text": ["going great"],
                "sentiment": ["positive"],
            }
        ).to_csv(raw_dir / "Tweets.csv", index=False)

        pd.DataFrame(
            {
                "sentence": ["i feel bad", "i feel great"],
                "sentiment": [0, 1],
            }
        ).to_csv(raw_dir / "twitter_train.csv", index=False)

        pd.DataFrame(
            {
                "Comment": ["this launch looks solid"],
                "Sentiment": ["positive"],
            }
        ).to_csv(raw_dir / "YoutubeCommentsDataSet.csv", index=False)

        pd.DataFrame(
            {
                "video_id": ["us_video_1"],
                "trending_date": ["17.14.11"],
                "title": ["United States trend"],
                "channel_title": ["Demo US Channel"],
                "category_id": [24],
                "publish_time": ["2017-11-13T07:30:00.000Z"],
                "tags": ["trend|usa"],
                "views": [12000],
                "likes": [900],
                "dislikes": [12],
                "comment_count": [150],
                "comments_disabled": [False],
                "ratings_disabled": [False],
                "description": ["US classic trending dataset row"],
            }
        ).to_csv(raw_dir / "USvideos.csv", index=False)

        try:
            result = build_analysis_assets(raw_dir=raw_dir, output_dir=preprocess_dir, generate_notebooks=False)

            self.assertTrue((preprocess_dir / "social_media_analysis_ready.csv").exists())
            self.assertTrue((preprocess_dir / "dashboard_monthly_metrics.csv").exists())
            self.assertTrue((preprocess_dir / "dashboard_source_metrics.csv").exists())
            self.assertTrue((preprocess_dir / "dashboard_topic_metrics.csv").exists())
            self.assertTrue((preprocess_dir / "dashboard_topic_timeline_metrics.csv").exists())
            self.assertEqual(len(result["catalog"]["datasets"]), 6)

            with (preprocess_dir / "dataset_catalog.json").open("r", encoding="utf-8") as handle:
                catalog = json.load(handle)
            self.assertEqual(len(catalog["datasets"]), 6)

            source_metrics = pd.read_csv(preprocess_dir / "dashboard_source_metrics.csv")
            self.assertEqual(int(source_metrics["record_count"].sum()), 7)
            self.assertIn("youtube_trending_united_states", source_metrics["source_dataset"].tolist())

            assets = load_dashboard_assets(preprocess_dir=preprocess_dir)
            self.assertIn("source_metrics", assets)
            self.assertIn("topic_timeline_metrics", assets)
            self.assertFalse(assets["source_metrics"].empty)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
