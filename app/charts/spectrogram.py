"""Spectrogram: how the frequency content of a signal changes with time.

A spectrum says which frequencies a signal holds; it cannot say when. A chirp,
a motor run up to speed or a bearing that starts to rattle all have a spectrum
that is the same blur whatever the order of the samples. The spectrogram cuts
the signal into overlapping windows, takes each one's spectrum, and draws the
power as a colour over time (x) and frequency (y).

  https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.logs.logger import applogger
from app.plugin_api.charts.base import BaseAxisRenderer, SeriesData
from app.utils.coercion import to_numbers

#: Options this renderer consumes itself, stripped before the rest reaches ``ax.specgram``.
_RENDERER_ONLY_KWARGS: tuple[str, ...] = ("sampling_rate", "colorbar", "colorbar_label")


class SpectrogramAxisRenderer(BaseAxisRenderer):
    """Renderer for the short-time spectrum of one signal.

    Role columns:
        y   required: the signal, one sample per row.
        x   optional: the time of each sample. Its median step gives the
            sampling rate and its first value where the picture starts;
            without it the rate is the Sampling rate option (1 by default).
    """

    Name: str = "Spectrogram"
    Category: str = "Pairwise data"
    Description: str = (
        "Frequency content of a signal over time, as colour - when a "
        "frequency appears, not only whether it does."
    )
    Keywords: str = "specgram stft short time fourier frequency time spettrogramma segnale vibration"
    Goals: list[str] = ["signal"]
    Link: str = "https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html"

    RequiredRoles: list[str] = ["y"]
    OptionalRoles: list[str] = ["x"]

    #: A second signal would be drawn over the first one's colours.
    MaxSeries: int | None = 1

    Kwargs: dict[str, Any] = {
        "NFFT": {
            "default": 256,
            "type": int,
            "min": 8,
            "max": 65536,
            "group": "Windows",
            "description": "Samples in each window. Longer resolves frequency finer and time coarser.",
        },
        "noverlap": {
            "default": 128,
            "type": int,
            "min": 0,
            "group": "Windows",
            "description": "Samples shared by neighbouring windows; below NFFT. More overlap gives a smoother picture of time.",
        },
        "sampling_rate": {
            "default": None,
            "type": float,
            "min": 0.0,
            "group": "Windows",
            "description": "Samples per unit of time. Empty reads it from the x column, or takes 1.",
        },
        "scale": {
            "default": "dB",
            "type": ["dB", "linear"],
            "group": "Appearance",
            "description": "Power in decibels, or as it is.",
        },
        "cmap": {
            "default": "viridis",
            "type": str,
            "group": "Appearance",
            "description": "Colormap the power is mapped through.",
        },
        "colorbar": {
            "default": False,
            "type": bool,
            "group": "Colorbar",
            "description": "Add a colorbar for the power scale.",
        },
        "colorbar_label": {
            "default": None,
            "type": str,
            "group": "Colorbar",
            "description": "Label written alongside the colorbar.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(series, reason="a second spectrogram would paint over the first")
        if sd is None:
            return
        if "y" not in sd.df.columns:
            applogger.warning("Series '%s' skipped: Spectrogram needs a y role.", sd.name)
            return

        values = to_numbers(sd.df["y"]).to_numpy(dtype=float)
        times = to_numbers(sd.df["x"]).to_numpy(dtype=float) if "x" in sd.df.columns else None
        finite = np.isfinite(values)
        if times is not None:
            finite &= np.isfinite(times)
            times = times[finite]
        values = values[finite]

        style = self.merge_style(axis_options, sd.style or {})
        kwargs = self.get_kwargs(style)
        for key in _RENDERER_ONLY_KWARGS:
            kwargs.pop(key, None)

        nfft = int(kwargs.get("NFFT") or 256)
        if values.size < nfft:
            applogger.info(
                "Series '%s' skipped: %d samples is fewer than a window of %d.",
                sd.name,
                values.size,
                nfft,
            )
            return
        kwargs["NFFT"] = nfft
        kwargs["noverlap"] = min(int(kwargs.get("noverlap") or 0), nfft - 1)

        rate = self.opt("sampling_rate", style)
        start = 0.0
        if times is not None and times.size > 1:
            step = float(np.median(np.diff(times)))
            if not rate and step > 0:
                rate = 1.0 / step
            start = float(times[0])
        kwargs["Fs"] = float(rate) if rate else 1.0
        kwargs["xextent"] = (start, start + values.size / kwargs["Fs"])

        try:
            _spectrum, _freqs, _bins, image = ax.specgram(values, **kwargs)
        except Exception:  # noqa: BLE001 - Matplotlib with the user's settings: a bad one costs this element, not the chart
            applogger.exception("Spectrogram failed to draw series '%s'.", sd.name)
            return
        ax.set_aspect("auto")

        if bool(self.opt("colorbar", style)):
            self.add_colorbar(ax, image, style)

        self.apply_annotations(ax, axis_options)
