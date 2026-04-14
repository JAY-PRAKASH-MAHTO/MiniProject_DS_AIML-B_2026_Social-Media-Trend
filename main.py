"""Streamlit app launcher.

This file is rerun by Streamlit on every interaction, so we execute the
dashboard module fresh each time instead of relying on Python's import cache.
"""

from runpy import run_module


run_module("src.analysis")
