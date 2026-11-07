"""The SymPy workpad: its filter, the project's tables read and written, charts - on a real project, no window."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.workpad.engine import UnsafeCommand, Workpad, check


@pytest.fixture
def repo(tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    repo.import_dataframe(pd.DataFrame({"species": ["A", "A", "B", "B"], "mass": [1.0, 2.0, 3.0, 4.0]}),
                          table_name="Birds", normalize_columns=False)
    yield repo
    repo.close()


@pytest.mark.parametrize("source", [
    "import os", "from os import path", "__import__('os')", "eval('1')", "exec('x=1')", "open('f')",
    "x.__class__", "().__class__.__bases__", "getattr(x, 'y')", "sympify('x')", "a = '__class__'",
    "def f(_x): return 1", "globals()",
])
def test_the_filter_refuses_what_could_reach_python(source):
    with pytest.raises(UnsafeCommand):
        check(source)


@pytest.mark.parametrize("source", [
    "x = symbols('x')", "diff(sin(x)*x, x)", "[i**2 for i in range(3)]", "f = lambda t: t + 1",
    "def g(t):\n    return t * 2", "for i in range(2):\n    y = i",
])
def test_the_filter_lets_ordinary_commands_through(source):
    check(source)


def test_a_session_keeps_its_names_and_answers_with_the_last_expression(repo):
    pad = Workpad(lambda: repo)
    pad.run("x = symbols('x')")
    _printed, value = pad.run("f = exp(-x)*sin(x)\ndiff(f, x)")
    assert str(value) == "-exp(-x)*sin(x) + exp(-x)*cos(x)"
    assert "f" in pad.names() and "diff" not in pad.names()
    printed, value = pad.run("print(2 + 2)")
    assert printed == "4\n" and value is None
    with pytest.raises(UnsafeCommand):
        pad.run("import os")
    pad.safe = False
    pad.run("import math")
    assert "math" in pad.names()


def test_get_reads_columns_with_names_matched_loosely(repo):
    pad = Workpad(lambda: repo)
    pad.run("m = get('birds', 'Mass')")
    assert isinstance(pad.namespace["m"], np.ndarray)
    assert pad.run("mean(m), std(m), median(m), count(m)")[1] == (2.5, pytest.approx(1.2909944), 2.5, 4)
    assert pad.run("get('Birds', 'species')")[1] == ["A", "A", "B", "B"]
    assert set(pad.run("get('Birds')")[1]) == {"species", "mass"}
    assert pad.run("tables()")[1] == ["Birds"]
    with pytest.raises(ValueError, match="Columns: species, mass"):
        pad.run("get('Birds', 'weight')")


def test_save_writes_a_new_table_and_never_overwrites(repo):
    pad = Workpad(lambda: repo)
    pad.run("x = symbols('x')\nxs = get('Birds', 'mass')")
    assert pad.run("save('Birds', {'x': xs, 'y': [x**2 for x in xs]})")[1] == "Birds 2"
    assert repo.query_df('SELECT y FROM "Birds 2"')["y"].tolist() == [1.0, 4.0, 9.0, 16.0]
    assert pad.run("store('Squares', {'n': Matrix([1, 2])})")[1] == "Squares"
    with pytest.raises(ValueError, match="different lengths"):
        pad.run("write('Bad', {'a': [1, 2], 'b': [1]})")


def test_plot_draws_values_or_a_table_s_columns(repo):
    made: list[list[int]] = []
    pad = Workpad(lambda: repo, changed=made.append)
    pad.run("xs = [1, 2, 3]\nys = [x**2 for x in xs]")
    assert "drawn" in pad.run("plot('Scatter', x=xs, y=ys)")[1]
    assert "drawn" in pad.run("plot('Box Plot', table='Birds', y='mass', group='species')")[1]
    assert sum(1 for figures in made if figures) == 2
    assert len(repo.list_figures()) == 2 if hasattr(repo, "list_figures") else True
