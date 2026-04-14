from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import PREPROCESS_DIR, ensure_preprocessed_assets, generate_analysis_notebooks


def build_notebooks() -> None:
    ensure_preprocessed_assets(generate_notebooks=False)
    generate_analysis_notebooks(PREPROCESS_DIR / "dataset_catalog.json")


if __name__ == "__main__":
    build_notebooks()
    print("Dataset notebooks regenerated from the latest preprocessing catalog.")
