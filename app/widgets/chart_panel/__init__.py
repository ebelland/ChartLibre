"""The chart panel: one figure drawn from the project, with its notes, tools and menus.

ChartPanel (panel.py): notes, picking, hover and menus. It inherits
ChartToolsMixin (tools.py): the interactive tools, the view's geometry and
zoom, and the actions on the chart.
"""
from app.widgets.chart_panel.constants import FIGURE_VIEW_OPTIONS_KEY, RESIZE_MODE_CHOICES, axis_text
from app.widgets.chart_panel.panel import ChartPanel

__all__ = ["ChartPanel", "FIGURE_VIEW_OPTIONS_KEY", "RESIZE_MODE_CHOICES", "axis_text"]
