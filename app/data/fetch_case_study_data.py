"""Download the real datasets behind the case-study demo, once.

The other demo tables are either shipped CSVs or exact-by-construction
synthetic data (see demo_project.py). These four are neither: they are
the actual published observations behind four results a physicist, a
climatologist or a seismologist would recognise, pulled from the
institutions that maintain them.

Run once, from the repository root::

    python -m app.data.fetch_case_study_data

It writes tidy CSVs into ``sample data/`` - one per subject, already
cleaned of each source's own sentinel values and header preamble - and
:mod:`app.data.demo_project` then builds from those files offline, like
every other demo. Downloading at build time instead would make a demo
that cannot be rebuilt on a train, and would silently change under
whoever rebuilt it: three of these series grow every month.

The snapshot is therefore deliberate. Re-run this module to refresh it.

Sources, all public and citable:

``co2_mauna_loa.csv``
    NOAA Global Monitoring Laboratory, monthly mean CO2 at Mauna Loa -
    the Keeling curve, running since 1958.
``sunspots_monthly.csv``
    WDC-SILSO, Royal Observatory of Belgium: monthly mean sunspot
    number, the longest continuous record in observational astronomy.
``global_temperature.csv``
    NASA GISS Surface Temperature Analysis (GISTEMP v4), annual global
    land-ocean anomaly against the 1951-1980 mean.
``earthquakes_recent.csv``
    USGS, every magnitude 2.5+ earthquake of the last month.
"""
from __future__ import annotations

import io
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from app.logs.logger import applogger

#: Where the cleaned CSVs land - the same folder every other demo table
#: is read from.
SAMPLE_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "sample data"

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
SUNSPOTS_URL = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php"
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
EARTHQUAKES_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv"
)

#: Some of these hosts refuse a request with no User-Agent at all.
_HEADERS = {"User-Agent": "ChartLibre demo data fetcher"}


def _download(url: str, *, timeout: int = 60) -> bytes:
    """Return *url*'s body, or raise - a partial demo is worse than none."""
    applogger.info("Fetching %s", url)
    with urlopen(Request(url, headers=_HEADERS), timeout=timeout) as response:
        return response.read()


def fetch_co2() -> pd.DataFrame:
    """Monthly mean CO2 at Mauna Loa, as date / ppm / deseasonalised ppm.

    The file carries ~40 lines of '#' preamble and uses -9.99 / -0.99 as
    "no value", which would otherwise plot as a cliff down to -10 ppm.
    """
    raw = _download(CO2_URL)
    frame = pd.read_csv(io.BytesIO(raw), comment="#")
    frame = frame.rename(
        columns={"average": "co2_ppm", "deseasonalized": "co2_trend_ppm"}
    )
    frame["date"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=15)
    )
    for column in ("co2_ppm", "co2_trend_ppm"):
        frame.loc[frame[column] < 0, column] = pd.NA
    out = frame[["date", "co2_ppm", "co2_trend_ppm"]].dropna(subset=["co2_ppm"])
    return out.reset_index(drop=True)


def fetch_sunspots() -> pd.DataFrame:
    """Monthly mean sunspot number, as date / number.

    SILSO ships semicolon-separated columns with no header, and -1 where
    a month has no observation.
    """
    raw = _download(SUNSPOTS_URL)
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        header=None,
        names=[
            "year",
            "month",
            "decimal_year",
            "sunspot_number",
            "std_dev",
            "observations",
            "definitive",
        ],
    )
    frame = frame[frame["sunspot_number"] >= 0].copy()
    frame["date"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=15)
    )
    out = frame[["date", "sunspot_number"]]
    return out.reset_index(drop=True)


def fetch_global_temperature() -> pd.DataFrame:
    """Annual global land-ocean temperature anomaly, as year / degrees C.

    GISTEMP's first line is a title, not a header, and it writes '***'
    for a month or season it cannot compute yet. Only the annual mean
    (its 'J-D' column) is kept here.
    """
    raw = _download(GISTEMP_URL)
    frame = pd.read_csv(io.BytesIO(raw), skiprows=1, na_values=["***"])
    frame = frame.rename(columns={"J-D": "anomaly_c"})
    out = frame[["Year", "anomaly_c"]].rename(columns={"Year": "year"})
    out = out.dropna(subset=["anomaly_c"])
    out["anomaly_c"] = out["anomaly_c"].astype(float)
    return out.reset_index(drop=True)


def fetch_earthquakes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The last month of magnitude 2.5+ earthquakes, and their size law.

    Two frames: the catalogue itself, and the Gutenberg-Richter counts
    derived from it - for each magnitude bin, how many earthquakes were
    *at least* that large. That cumulative count is the quantity the law
    is stated in, and log10 of it against magnitude is the straight line
    the demo fits.
    """
    raw = _download(EARTHQUAKES_URL)
    catalogue = pd.read_csv(io.BytesIO(raw))
    catalogue = catalogue[
        ["time", "latitude", "longitude", "depth", "mag", "place"]
    ].dropna(subset=["mag"])
    catalogue["time"] = pd.to_datetime(catalogue["time"], format="ISO8601", utc=True)
    catalogue["time"] = catalogue["time"].dt.tz_localize(None)
    catalogue = catalogue.sort_values("time").reset_index(drop=True)

    # One bin per reported magnitude: the scale is published to one
    # decimal, so a finer bin would only add empty ones.
    magnitudes = np.round(np.sort(catalogue["mag"].unique()), 1)
    counts = np.array(
        [int((catalogue["mag"] >= magnitude - 1e-9).sum()) for magnitude in magnitudes]
    )
    law = pd.DataFrame(
        {
            "magnitude": magnitudes,
            "count_at_least": counts,
            # log10 of the cumulative count is what turns the power law
            # into the straight line the demo fits. Kept as its own
            # column rather than left to a log axis: the fit runs on the
            # numbers, and it is log10(N) that is linear in magnitude.
            "log10_count": np.log10(counts),
        }
    )
    return catalogue, law


def main() -> None:
    """Fetch every case-study dataset and write it under sample data/."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []

    co2 = fetch_co2()
    co2.to_csv(SAMPLE_DATA_DIR / "co2_mauna_loa.csv", index=False)
    written.append(("co2_mauna_loa.csv", len(co2)))

    sunspots = fetch_sunspots()
    sunspots.to_csv(SAMPLE_DATA_DIR / "sunspots_monthly.csv", index=False)
    written.append(("sunspots_monthly.csv", len(sunspots)))

    temperature = fetch_global_temperature()
    temperature.to_csv(SAMPLE_DATA_DIR / "global_temperature.csv", index=False)
    written.append(("global_temperature.csv", len(temperature)))

    catalogue, law = fetch_earthquakes()
    catalogue.to_csv(SAMPLE_DATA_DIR / "earthquakes_recent.csv", index=False)
    written.append(("earthquakes_recent.csv", len(catalogue)))
    law.to_csv(SAMPLE_DATA_DIR / "earthquake_size_law.csv", index=False)
    written.append(("earthquake_size_law.csv", len(law)))

    for name, rows in written:
        print(f"{name}: {rows} rows")


if __name__ == "__main__":
    main()
