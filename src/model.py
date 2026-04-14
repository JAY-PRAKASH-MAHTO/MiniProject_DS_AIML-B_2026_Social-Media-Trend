from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw_data"
PROCESSED_DATA_DIR = DATASET_DIR / "processed_data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATASET_NOTEBOOK_DIR = NOTEBOOKS_DIR / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
RESULTS_DIR = OUTPUTS_DIR / "results"
REPORT_DIR = PROJECT_ROOT / "report"


def ensure_project_directories() -> None:
    for path in [
        DOCS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        DATASET_NOTEBOOK_DIR,
        OUTPUTS_DIR,
        GRAPHS_DIR,
        RESULTS_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
