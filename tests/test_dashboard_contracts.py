from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_contracts import (
    build_top_trends_calendar,
    get_dashboard_layout_contract,
    select_reliable_platform_leader,
)


class DashboardContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topic_timeline = pd.DataFrame(
            {
                "year_month": [
                    "2023-01",
                    "2023-01",
                    "2023-01",
                    "2023-01",
                    "2023-02",
                    "2023-02",
                    "2023-02",
                    "2023-02",
                    "2023-02",
                ],
                "primary_topic": [
                    "music",
                    "gaming",
                    "fitness",
                    "no_explicit_topic",
                    "music",
                    "gaming",
                    "fitness",
                    "education",
                    "",
                ],
                "record_count": [120, 240, 90, 50, 200, 235, 70, 180, 40],
                "total_views": [1500, 5200, 700, 100, 2600, 5300, 600, 1700, 80],
                "total_engagement": [150, 260, 120, 5, 360, 255, 110, 340, 4],
            }
        )
        self.analysis_source = Path("src/analysis.py").read_text(encoding="utf-8")
        self.contract_source = Path("src/dashboard_contracts.py").read_text(encoding="utf-8")
        self.main_source = Path("main.py").read_text(encoding="utf-8")

    def test_layout_contract_exposes_expected_sections(self) -> None:
        contract = get_dashboard_layout_contract()

        self.assertEqual(
            contract["hero_pills"],
            ("Last refresh", "Datasets in current view", "Visible platforms"),
        )
        self.assertEqual(
            contract["tabs"],
            ("Overview", "Comparison Lab", "Trends & Sentiment", "Dataset Coverage"),
        )
        self.assertEqual(len(contract["metric_cards"]), 6)

    def test_top_trends_filters_out_non_topics(self) -> None:
        result = build_top_trends_calendar(self.topic_timeline, trend_lens="total_engagement", top_n=3)

        self.assertFalse(result.empty)
        self.assertNotIn("no_explicit_topic", result["primary_topic"].str.lower().tolist())
        self.assertNotIn("", result["primary_topic"].tolist())

    def test_top_trends_returns_max_three_topics_per_month(self) -> None:
        result = build_top_trends_calendar(self.topic_timeline, trend_lens="total_engagement", top_n=3)

        monthly_counts = result.groupby("year_month")["primary_topic"].count()
        self.assertTrue((monthly_counts <= 3).all())

    def test_top_trends_are_observed_only(self) -> None:
        result = build_top_trends_calendar(self.topic_timeline, trend_lens="record_count", top_n=3)

        expected_months = set(self.topic_timeline["year_month"].unique().tolist())
        self.assertTrue(set(result["year_month"].unique().tolist()).issubset(expected_months))
        self.assertTrue((result["topic_origin"] == "Observed").all())

    def test_top_trends_have_stable_ranks_per_month(self) -> None:
        result = build_top_trends_calendar(self.topic_timeline, trend_lens="total_views", top_n=3)

        for _, month_frame in result.groupby("year_month"):
            expected_ranks = list(range(1, len(month_frame) + 1))
            self.assertEqual(month_frame.sort_values("rank")["rank"].tolist(), expected_ranks)
            self.assertEqual(month_frame.iloc[0]["trend_lens"], "total_views")

    def test_top_trends_can_fill_missing_months(self) -> None:
        result = build_top_trends_calendar(
            self.topic_timeline,
            trend_lens="total_engagement",
            top_n=3,
            fill_missing_months=True,
            start="2023-01",
            end="2023-03",
        )

        self.assertEqual(set(result["year_month"].unique().tolist()), {"2023-01", "2023-02", "2023-03"})
        march_rows = result[result["year_month"] == "2023-03"]
        self.assertFalse(march_rows.empty)
        self.assertTrue((march_rows["topic_origin"] == "Modeled").all())

    def test_analysis_uses_updated_tab_label(self) -> None:
        self.assertIn("DASHBOARD_LAYOUT_CONTRACT[\"tabs\"]", self.analysis_source)
        self.assertIn("Trends & Sentiment", self.contract_source)
        self.assertNotIn("Topics & Sentiment", self.analysis_source)

    def test_analysis_removes_hero_clutter_pills(self) -> None:
        self.assertNotIn("Top activity source:", self.analysis_source)
        self.assertNotIn("Trending topic:", self.analysis_source)

    def test_analysis_removes_methodology_expander(self) -> None:
        self.assertNotIn("with st.expander(\"Methodology\"", self.analysis_source)

    def test_select_reliable_platform_leader_ignores_thin_coverage(self) -> None:
        platform_rollup = pd.DataFrame(
            {
                "platform": ["Instagram", "Twitter", "YouTube"],
                "record_count": [40, 10000, 4500],
                "total_views": [1500, 600000, 350000],
                "engagement_rate": [0.11, 0.05, 0.06],
            }
        )

        leader, share, mode = select_reliable_platform_leader(platform_rollup)

        self.assertIn(leader, {"Twitter", "YouTube"})
        self.assertGreater(share, 0.03)
        self.assertEqual(mode, "reliable")

    def test_select_reliable_platform_leader_uses_fallback_when_all_thin(self) -> None:
        platform_rollup = pd.DataFrame(
            {
                "platform": ["Instagram", "TikTok"],
                "record_count": [80, 70],
                "total_views": [500, 450],
                "engagement_rate": [0.13, 0.10],
            }
        )

        leader, _, mode = select_reliable_platform_leader(platform_rollup)

        self.assertEqual(leader, "Instagram")
        self.assertEqual(mode, "fallback")

    def test_analysis_uses_non_crashing_filter_fallback(self) -> None:
        self.assertIn("if column not in filtered.columns", self.analysis_source)
        self.assertIn("Reverting to full-scope summary data to keep the dashboard running.", self.analysis_source)

    def test_analysis_uses_safe_figure_renderer(self) -> None:
        self.assertIn("def render_figure(fig", self.analysis_source)
        self.assertIn("st.pyplot(fig, use_container_width=True)", self.analysis_source)
        self.assertIn("plt.close(fig)", self.analysis_source)
        self.assertNotIn("def render_figure(fig: plt.Figure) -> None:\n    render_figure(fig)", self.analysis_source)

    def test_analysis_uses_control_deck_form_state(self) -> None:
        self.assertIn("with st.form(\"control_deck_form\"", self.analysis_source)
        self.assertIn("form_submit_button(\"Apply Filters\"", self.analysis_source)
        self.assertIn("form_submit_button(\"Reset\"", self.analysis_source)
        self.assertIn("control_deck_state", self.analysis_source)

    def test_main_launcher_executes_dashboard_module_fresh_on_each_rerun(self) -> None:
        self.assertIn("from runpy import run_module", self.main_source)
        self.assertIn('run_module("src.analysis")', self.main_source)
        self.assertNotIn("from src.analysis import *", self.main_source)

    def test_format_topic_label_sanitization_logic_present(self) -> None:
        self.assertIn('unicodedata.normalize("NFKD"', self.analysis_source)
        self.assertIn('.encode("ascii", "ignore").decode("ascii")', self.analysis_source)
        self.assertIn('cleaned = "Topic"', self.analysis_source)


if __name__ == "__main__":
    unittest.main()
