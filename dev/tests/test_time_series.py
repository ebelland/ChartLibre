"""AN-05: Time Series - autocorrelations, STL, ARIMA and ETS forecasts, checked against statsmodels and known series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import time_series as ts


def _ar1(n: int = 300, phi: float = 0.7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal()
    return values


def _monthly(n: int = 96) -> pd.DataFrame:
    months = pd.date_range("2016-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(5)
    values = 10 + 0.05 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "sales": values})


def test_the_autocorrelations_are_statsmodels(qapp=None) -> None:
    from statsmodels.tsa.stattools import acf, pacf

    y = _ar1()
    result = ts.analyse(pd.DataFrame({"y": y}), "y", model=ts.NONE, lags=10)
    assert result.correlations["ACF"].to_numpy() == pytest.approx(acf(y, nlags=10, fft=True)[1:])
    assert result.correlations["PACF"].to_numpy() == pytest.approx(pacf(y, nlags=10, method="ywm")[1:])
    assert result.correlations["Prob>Q"].iloc[0] < 1e-10  # an AR(1) is anything but white noise
    assert result.forecast.empty and "Fitted" not in result.rows


def test_arima_finds_the_ar_coefficient_and_forecasts_toward_the_mean() -> None:
    result = ts.analyse(pd.DataFrame({"y": _ar1()}), "y", model=ts.ARIMA, order=(1, 0, 0), horizon=30)
    phi = result.parameters.set_index("Term").loc["ar.L1", "Estimate"]
    assert phi == pytest.approx(0.7, abs=0.08)
    forecast = result.forecast
    assert len(forecast) == 30 and (forecast["Lower 95%"] < forecast["Forecast"]).all()
    assert abs(forecast["Forecast"].iloc[-1] - result.parameters.set_index("Term").loc["const", "Estimate"]) < 0.2
    assert result.forecast["Time"].iloc[0] == 301  # the row order, carried on
    assert result.summary["Residuals: Ljung-Box Prob>Q"] > 0.05 and not result.notes


def test_monthly_dates_give_a_season_of_twelve_stl_and_an_ets_forecast() -> None:
    frame = _monthly()
    result = ts.analyse(frame.sample(frac=1, random_state=1), "sales", "month", model=ts.ETS, horizon=12)
    assert result.temporal and result.season == 12
    assert list(result.rows["Time"].head(2)) == ["2016-01-01", "2016-02-01"]  # sorted by date
    seasonal = result.rows["Seasonal"].to_numpy()
    assert np.corrcoef(seasonal, 3 * np.sin(2 * np.pi * np.arange(96) / 12))[0, 1] > 0.95
    assert list(result.forecast["Time"].head(2)) == ["2024-01-01", "2024-02-01"]
    expected = 10 + 0.05 * np.arange(96, 108) + 3 * np.sin(2 * np.pi * np.arange(96, 108) / 12)
    assert np.abs(result.forecast["Forecast"].to_numpy() - expected).max() < 1.0


def test_what_cannot_be_done_is_said() -> None:
    with pytest.raises(ValueError, match="too few"):
        ts.analyse(pd.DataFrame({"y": np.arange(5.0)}), "y")
    with pytest.raises(ValueError, match="neither numbers nor dates"):
        ts.analyse(pd.DataFrame({"y": np.arange(20.0), "t": ["a"] * 20}), "y", "t")
    short = ts.analyse(pd.DataFrame({"y": np.arange(20.0)}), "y", model=ts.NONE, season=12)
    assert short.season == 0 and short.notes


def test_ok_writes_the_tables_and_draws_the_series_its_correlations_and_parts(qapp, tmp_path) -> None:
    import time

    from app.data.sqlite_repo import SqliteRepo
    from app.table_operations.time_series_dialog import TimeSeriesDialog

    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo.import_dataframe(_monthly(), table_name="shop", normalize_columns=False)
    dialog = TimeSeriesDialog(repo=repo, table="shop")
    try:
        dialog.roles_widget.set_casting({"response": ["sales"], "time": ["month"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 60
        while dialog._task is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert len(repo.table_frame("_shop_time_series_forecast")) == 12
        assert {"Fitted", "Trend", "Seasonal"} <= set(repo.table_frame("_shop_time_series").columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Time Series", "Stem Plot", "Stem Plot", "Time Series"]
        assert len(figure.axes[0].series) == 6  # the values, fitted, forecast, its two bounds, the trend
    finally:
        dialog.close()
        repo.close()


def test_mid_month_dates_are_monthly_too() -> None:
    frame = _monthly()
    frame["month"] = pd.to_datetime(frame["month"]).map(lambda d: d.replace(day=15)).dt.strftime("%Y-%m-%d")
    result = ts.analyse(frame, "sales", "month", horizon=3)
    assert result.season == 12 and list(result.forecast["Time"]) == ["2024-01-15", "2024-02-15", "2024-03-15"]
