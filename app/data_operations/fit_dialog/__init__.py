"""Data operation Fit: curve and surface fitting of chart series or table columns.

SeriesFitDialog (dialog.py) is built from mixins: model data and parameters,
running the fit, algorithm and starting point, the base-class hooks, accessory
charts and the parameter table's menu. Shared types in common.py.
"""
from app.data_operations.fit_dialog.dialog import SeriesFitDialog
from app.data_operations.fit_dialog.common import SeriesFitResult

__all__ = ["SeriesFitDialog", "SeriesFitResult"]
