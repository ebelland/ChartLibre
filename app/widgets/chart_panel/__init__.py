"""The chart panel: one figure drawn from the project, with its notes, tools and menus.

ChartPanel (panel.py) is built from mixins, one module per concern, as the main
window is: notes, picking, hover, tools, menus, geometry, actions.
"""
from app.widgets.chart_panel.constants import FIGURE_VIEW_OPTIONS_KEY, RESIZE_MODE_CHOICES, axis_text
from app.widgets.chart_panel.panel import ChartPanel

__all__ = ["ChartPanel", "FIGURE_VIEW_OPTIONS_KEY", "RESIZE_MODE_CHOICES", "axis_text"]
