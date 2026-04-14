---
applyTo:
  - src/analysis.py
  - src/dashboard_contracts.py
  - tests/test_dashboard_contracts.py
description: "Use when: editing dashboard layout, trend ranking logic, or dashboard validation tests."
---
# Dashboard QA Rules

Keep this dashboard aligned with media trend analysis, not topic-search summaries.

## Layout contracts
- Keep hero chips minimal and high-signal.
- Keep KPI cards concise and non-redundant.
- Keep tab naming stable to preserve navigation consistency.

## Trend logic contracts
- Trend calendar must show top 3 trends per month.
- Trend calendar must use observed data only.
- Ranking must combine momentum with at least one quality-normalized signal.
- Exclude non-topic placeholders from trend outputs.

## Testing contracts
- Include unit checks for top-3 behavior, observed-only output, rank stability, and layout contract consistency.
- Avoid tests that rely on a running Streamlit server for core logic validation.
