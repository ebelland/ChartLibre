"""The base of every series (data) operation dialog.

DataOperationDialogBase (dialog.py) is built from mixins, one module per concern:
builder hooks, layout and data source, evaluation, inputs, results, the preview
transaction. The models and jobs are in common.py.
"""
from app.plugin_api.data_operations.dialog_base.dialog import DataOperationDialogBase
from app.plugin_api.data_operations.dialog_base.common import OperationModel
from app.plugin_api.data_operations.dialog_base.common import ResultSeriesSpec
from app.plugin_api.data_operations.dialog_base.common import OperationJob
from app.plugin_api.data_operations.dialog_base.common import CallJob
from app.plugin_api.data_operations.dialog_base.common import SeriesOutcome
from app.plugin_api.data_operations.dialog_base.common import SeriesJob
from app.plugin_api.data_operations.dialog_base.common import generated_table_name
from app.plugin_api.data_operations.dialog_base.common import GENERATED_TABLE_PREFIX

__all__ = ["DataOperationDialogBase", "OperationModel", "ResultSeriesSpec", "OperationJob", "CallJob", "SeriesOutcome", "SeriesJob", "generated_table_name", "GENERATED_TABLE_PREFIX"]
