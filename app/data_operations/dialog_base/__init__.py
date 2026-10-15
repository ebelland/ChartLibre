"""The base of every series (data) operation dialog.

DataOperationDialogBase (dialog.py) is built from mixins, one module per concern:
builder hooks, layout and data source, evaluation, inputs, results, the preview
transaction. The models and jobs are in common.py.
"""
from app.data_operations.dialog_base.dialog import DataOperationDialogBase
from app.data_operations.dialog_base.common import OperationModel
from app.data_operations.dialog_base.common import ResultSeriesSpec
from app.data_operations.dialog_base.common import OperationJob
from app.data_operations.dialog_base.common import CallJob
from app.data_operations.dialog_base.common import SeriesOutcome
from app.data_operations.dialog_base.common import SeriesJob
from app.data_operations.dialog_base.common import generated_table_name
from app.data_operations.dialog_base.common import GENERATED_TABLE_PREFIX

__all__ = ["DataOperationDialogBase", "OperationModel", "ResultSeriesSpec", "OperationJob", "CallJob", "SeriesOutcome", "SeriesJob", "generated_table_name", "GENERATED_TABLE_PREFIX"]
