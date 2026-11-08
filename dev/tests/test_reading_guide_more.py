"""The "How to read this" boxes added for the analyses that had none (dev/plans/explanations_and_line_editor_plan.txt)."""
from __future__ import annotations

from app.analysis.peaks import Peak
from app.data_operations.peaks_dialog import PeakResult, SeriesPeaksDialog


def _peak(x: float, prominence: float, width: float) -> Peak:
    nan = float("nan")
    return Peak(x=x, y=prominence, prominence=prominence, width=width, left_x=x - width / 2, right_x=x + width / 2,
                **{name: nan for name in Peak.__dataclass_fields__ if name not in
                   ("x", "y", "prominence", "width", "left_x", "right_x")})


def test_peaks_says_how_many_and_describes_the_most_prominent() -> None:
    result = PeakResult(source_name="s", result_name="r", model="m",
                        peaks=[_peak(2.0, 1.0, 0.5), _peak(5.0, 3.0, 0.8)])
    lines = SeriesPeaksDialog.how_to_read(None, [result])
    assert len(lines) == 1 and "2 peaks kept" in lines[0] and "x = 5" in lines[0] and "0.8" in lines[0]
    empty = PeakResult(source_name="s", result_name="r", model="m", peaks=[])
    assert "No peak passed" in SeriesPeaksDialog.how_to_read(None, [empty])[0]
