# Social Media Trend Analyzer

<p align="center">
  <strong>A Streamlit-powered analytics workspace for exploring cross-platform social trends, engagement patterns, sentiment signals, and topic momentum from multi-source social media datasets.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="Data Stack" src="https://img.shields.io/badge/Data-Pandas%20%2B%20NumPy-150458?logo=pandas&logoColor=white" />
  <img alt="Visualization" src="https://img.shields.io/badge/Visualization-Matplotlib%20%2B%20Seaborn-1F77B4" />
  <img alt="Testing" src="https://img.shields.io/badge/Tests-20%20passing-success" />
</p>

## Overview

This project unifies multiple raw CSV datasets into a single analysis-ready social media layer and exposes the results through an interactive Streamlit dashboard. It combines preprocessing, dataset-specific cleaning rules, summary asset generation, notebook creation, and dashboard visualizations so you can move from raw data to insight quickly.

The repository currently includes **15 processed datasets**, **2,112,276 total processed rows**, **4 supported platforms**, and **18 notebooks** for project-level and dataset-level exploration. The dashboard is designed to load from stored summary assets in `dataset/processed_data`, which keeps the experience much faster than reading every raw file on each run.

## At A Glance

| Item | Details |
| --- | --- |
| Project type | Data analytics + Streamlit dashboard |
| Core use case | Social media trend, sentiment, engagement, and topic analysis |
| Platforms covered | Instagram, TikTok, Twitter, YouTube |
| Record types | Post, tweet, video, comment |
| Current processed datasets | 15 |
| Current processed rows | 2,112,276 |
| Notebooks generated | 18 |
| Test status | 20 unit tests passing |

## What This Dashboard Helps You Explore

- Compare dataset scale, engagement, and visibility across multiple social media sources.
- Track monthly activity shifts across platforms, datasets, and regions.
- Identify dominant topics using explicit tags, momentum scoring, and engagement quality.
- Analyze sentiment distribution from tweet and YouTube comment datasets.
- Inspect how record type and geography influence coverage and performance.
- Review a catalog of processed datasets and generated notebook assets.

## Dashboard Sections

- **Overview**: Hero summary, KPI cards, platform coverage, dataset footprint, and timeline views.
- **Comparison Lab**: Heatmaps, scale-vs-efficiency mapping, region leaderboards, and record-type mix.
- **Trends & Sentiment**: Trending topic spotlight, sentiment breakdown, topic cloud, monthly momentum calendar, and explicit topic rankings.
- **Dataset Coverage**: Processed dataset inventory and notes about generated assets.

## Screenshot Gallery

### Main Views

<table>
  <tr>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000420.png" alt="Dashboard hero and filter controls" />
    </td>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000428.png" alt="Overview KPI cards and tabs" />
    </td>
  </tr>
  <tr>
    <td align="center"><strong>Filter control deck and dashboard hero</strong></td>
    <td align="center"><strong>KPI cards, leadership signals, and tab navigation</strong></td>
  </tr>
  <tr>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000441.png" alt="Monthly record volume split by dataset" />
    </td>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000622.png" alt="Source platform performance map" />
    </td>
  </tr>
  <tr>
    <td align="center"><strong>Activity timeline across datasets</strong></td>
    <td align="center"><strong>Efficiency vs scale comparison by platform</strong></td>
  </tr>
  <tr>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000658.png" alt="Trend topic cloud" />
    </td>
    <td width="50%">
      <img src="Screenshots/Screenshot%202026-04-15%20000721.png" alt="Monthly trend momentum heatmap" />
    </td>
  </tr>
  <tr>
    <td align="center"><strong>Weighted topic cloud for trend discovery</strong></td>
    <td align="center"><strong>Top-3 monthly trend momentum calendar</strong></td>
  </tr>
</table>

<details>
  <summary><strong>Extended dashboard gallery</strong></summary>

  <br />

  <table>
    <tr>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000548.png" alt="Coverage by platform" />
      </td>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000554.png" alt="Coverage by dataset" />
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Coverage by platform</strong></td>
      <td align="center"><strong>Coverage by dataset</strong></td>
    </tr>
    <tr>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000602.png" alt="Top regions by posts" />
      </td>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000607.png" alt="Total engagement over months by year" />
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Top regions by post volume</strong></td>
      <td align="center"><strong>Year-over-year monthly engagement comparison</strong></td>
    </tr>
    <tr>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000617.png" alt="Dataset vs platform heatmap" />
      </td>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000628.png" alt="Region leaderboard" />
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Dataset vs platform engagement-rate heatmap</strong></td>
      <td align="center"><strong>Region leaderboard by engagement</strong></td>
    </tr>
    <tr>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000636.png" alt="Record type mix" />
      </td>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000649.png" alt="Trending topic spotlight" />
      </td>
    </tr>
    <tr>
      <td align="center"><strong>Record type mix across the active view</strong></td>
      <td align="center"><strong>Trending topic spotlight panel</strong></td>
    </tr>
    <tr>
      <td width="50%">
        <img src="Screenshots/Screenshot%202026-04-15%20000732.png" alt="Top explicit topics by total engagement" />
      </td>
      <td width="50%"></td>
    </tr>
    <tr>
      <td align="center"><strong>Top explicit topics ranked by engagement</strong></td>
      <td></td>
    </tr>
  </table>
</details>

## Project Architecture

The project follows a straightforward pipeline:

1. Raw CSV files are stored in `dataset/raw_data/`.
2. `src/preprocessing.py` cleans and standardizes each supported dataset.
3. The pipeline generates processed datasets, dashboard summary metrics, and `dataset_catalog.json` in `dataset/processed_data/`.
4. Dataset-specific and project-level notebooks are generated in `notebooks/`.
5. The Streamlit app reads the processed summary layer to power the dashboard.

## Runtime Architecture

At runtime, the application follows a lightweight read-from-assets design so the dashboard stays responsive:

```text
User
  -> Streamlit app (`main.py`)
  -> Dashboard module (`src.analysis`)
  -> Cached asset loader (`get_dashboard_assets`)
  -> Processed summary files in `dataset/processed_data`
  -> Filtering, aggregations, and visual components
  -> Interactive dashboard views
```

When a rebuild is triggered, the refresh path is:

```text
Raw CSV files in `dataset/raw_data`
  -> `build_analysis_assets()` in `src/preprocessing.py`
  -> Regenerated processed CSVs, summary metrics, catalog, and notebooks
  -> Streamlit cache clear
  -> Dashboard reload with fresh assets
```

## Key Features

- Unified preprocessing for posts, tweets, comments, and videos under a standardized analytics schema.
- Fast dashboard startup through precomputed summary assets stored in `dataset/processed_data`.
- Interactive filtering by dataset, platform, record type, region, sentiment label, comparison metric, and trend lens.
- Trend calendar logic that ranks the top monthly topics using momentum, share, and engagement-quality signals.
- Topic and sentiment exploration with spotlight cards, word clouds, explicit-topic rankings, and mood distribution views.
- Automatic notebook generation for both project-wide analysis and individual processed datasets.
- Test-backed dashboard contracts to reduce layout regressions and ranking logic errors.

## Dataset Coverage

The preprocessing layer auto-detects supported files present in `dataset/raw_data`. The current repository includes the following dataset families:

| Dataset family | Files currently included | Coverage |
| --- | --- | --- |
| Viral social posts | `Cleaned_Viral_Social_Media_Trends.csv` | Cross-platform post performance with views, likes, shares, comments, hashtags, and regions |
| Tweet sentiment selection | `Tweets.csv` | Tweet text, selected spans, and sentiment labels |
| Twitter binary sentiment training | `twitter_train.csv` | Large tweet corpus with positive and negative sentiment coverage |
| YouTube comment sentiment | `YoutubeCommentsDataSet.csv` | Comment-level audience sentiment from YouTube discussions |
| Regional YouTube trending | `USvideos.csv`, `GBvideos.csv`, `CAvideos.csv`, `DEvideos.csv`, `FRvideos.csv`, `INvideos.csv`, `JPvideos.csv`, `KRvideos.csv`, `MXvideos.csv`, `RUvideos.csv` | Country-level trending-video analysis with views, likes, comments, tags, and trend lag |
| Cross-country YouTube snapshot | `youtube.csv` | Publish-country snapshot with publish date, day-of-week, time frame, and trending metrics |

## Generated Outputs

Running the preprocessing pipeline creates and refreshes the following assets:

- `dataset/processed_data/social_media_analysis_ready.csv`
- `dataset/processed_data/dashboard_monthly_metrics.csv`
- `dataset/processed_data/dashboard_source_metrics.csv`
- `dataset/processed_data/dashboard_topic_metrics.csv`
- `dataset/processed_data/dashboard_topic_timeline_metrics.csv`
- `dataset/processed_data/dataset_catalog.json`
- `notebooks/data_understanding.ipynb`
- `notebooks/preprocessing.ipynb`
- `notebooks/visualization.ipynb`
- `notebooks/datasets/*.ipynb`

For GitHub distribution, the repository keeps the lightweight dashboard summary assets and the raw source datasets, while the very large generated processed CSV exports are meant to be rebuilt locally with `python src/preprocessing.py`.

## Repository Structure

```text
Social Media Trend Analyzer/
|-- README.md
|-- requirements.txt
|
|-- docs/
|-- dataset/
|   |-- raw_data/
|   \-- processed_data/
|-- notebooks/
|   |-- data_understanding.ipynb
|   |-- preprocessing.ipynb
|   \-- visualization.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- analysis.py
|   \-- model.py
|-- outputs/
|   |-- graphs/
|   \-- results/
\-- report/
```

`docs/` and `report/` are intentionally left empty so you can add your abstract, problem statement, presentation, and final report later.

Additional project files such as `main.py`, `tests/`, `Screenshots/`, and helper modules are still present in the repository, but the structure above reflects the simplified academic layout.

## Tech Stack

- **Application**: Streamlit
- **Data processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Text analysis support**: NLTK, TextBlob
- **Notebook generation**: Jupyter, nbformat
- **Testing**: Python `unittest`

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "Social Media Trend Analyzer"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows**

```powershell
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Refresh processed assets from the raw datasets

```bash
python src/preprocessing.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run main.py
```

### 6. Optional: regenerate notebooks only

```bash
python src/notebook_builder.py
```

> `run.txt` contains the Windows command variants used for this project setup.

## Testing

Run the automated test suite with:

```bash
python -m unittest discover -s tests
```

The current suite validates preprocessing transformations, ID generation, sentiment mapping, regional YouTube date parsing, dashboard layout contracts, trend-calendar ranking rules, and platform-leader selection behavior.

## Typical Workflow

1. Add or update source CSV files inside `dataset/raw_data/`.
2. Run `src/preprocessing.py` to rebuild processed datasets and dashboard summary assets.
3. Launch the app with `streamlit run main.py`.
4. Explore the dashboard by dataset, platform, region, sentiment, and trend lens.
5. Use the generated notebooks for deeper offline analysis or academic submission support.

## Notes

- The dashboard reads stored summary outputs from `dataset/processed_data`, not the raw files directly.
- If you replace or add raw datasets, rerun preprocessing before expecting the dashboard to reflect the changes.
- Some charts depend on available date, sentiment, or visibility fields, so the active view may change based on the selected filters.
