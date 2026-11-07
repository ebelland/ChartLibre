"""Kept for a while: an older ChartLibre.exe or shortcut starts main.py. The app is ChartLibre.py now."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("ChartLibre.py")), run_name="__main__")
