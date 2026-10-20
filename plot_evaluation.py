"""Compatibility entry point for the V2 chart command."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("plot_portfolio.py")),run_name="__main__")
