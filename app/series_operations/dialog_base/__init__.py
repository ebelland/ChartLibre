"""The base of every series (data) operation dialog.

SeriesOperationDialogBase (dialog.py) is built from mixins, one module per concern:
builder hooks, layout and data source, evaluation, inputs, results, the preview
transaction. The models and jobs are in common.py.
"""
from app.series_operations.dialog_base.dialog import SeriesOperationDialogBase
from app.series_operations.dialog_base.common import OperationModel
from app.series_operations.dialog_base.common import ResultSeriesSpec
from app.series_operations.dialog_base.common import OperationJob
from app.series_operations.dialog_base.common import CallJob
from app.series_operations.dialog_base.common import SeriesOutcome
from app.series_operations.dialog_base.common import SeriesJob
from app.series_operations.dialog_base.common import generated_table_name
from app.series_operations.dialog_base.common import GENERATED_TABLE_PREFIX

__all__ = ["SeriesOperationDialogBase", "OperationModel", "ResultSeriesSpec", "OperationJob", "CallJob", "SeriesOutcome", "SeriesJob", "generated_table_name", "GENERATED_TABLE_PREFIX"]
