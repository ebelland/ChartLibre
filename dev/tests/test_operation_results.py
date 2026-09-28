"""Every series operation's result is an OperationResult."""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pandas as pd
import pytest

import app.series_operations as package
from app.series_operations.filter_dialog import FilterResult
from app.series_operations.results import OperationResult, TableResult


def _result_classes() -> list[type]:
    found = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.name.endswith("_dialog"):
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and name.endswith("Result"):
                found.append(cls)
    return found


@pytest.mark.parametrize("cls", _result_classes(), ids=lambda cls: cls.__name__)
def test_every_result_is_a_complete_operation_result(cls: type) -> None:
    assert issubclass(cls, OperationResult)
    assert not inspect.isabstract(cls), f"{cls.__name__} leaves {cls.__abstractmethods__} unimplemented"


def test_a_table_result_must_say_what_its_table_is() -> None:
    class Incomplete(TableResult):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_the_common_properties_and_report() -> None:
    result = FilterResult(
        source_name="signal", result_name="signal - IIR", model="IIR filter",
        x=np.arange(3.0), y=np.zeros(3), metadata={"order": 4},
    )
    assert result.model_name == "IIR filter"
    assert result.series == ("signal",)
    assert result.parameters == {"order": 4}
    assert isinstance(result.to_df(), pd.DataFrame)
    report = result.to_html()
    assert "IIR filter" in report and "signal" in report and "order" in report
