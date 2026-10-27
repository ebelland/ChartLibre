"""FunctionEditor: a stored function's formula, its parameters' values and its range, in three tabs; and a function
source's query read back (parse_function_sql) and rewritten (replace_function_sql)."""
from __future__ import annotations

from app.data.repo.function_sql import GridRange, function_sql, parse_function_sql, replace_function_sql
from app.widgets.function_editor import FunctionEditor

LINE = {"function_name": "Line", "source_type": "expression", "symbols": "a, b", "code_string": "a*x + b",
        "description": "a line", "from": "0", "to": "4", "points": "5", "values": '{"a": 2.0, "b": 1.0}'}


def test_the_query_of_a_function_is_read_back_and_rewritten() -> None:
    sql = function_sql("O'Brien", {"a": 2.5}, GridRange(-1.0, 3.0, 7), GridRange(0.1, 10.0, 4, "log"))
    read = parse_function_sql(f'SELECT "x" AS "x", "z" AS "z" FROM ({sql}) AS "_src"')
    assert read is not None and read.name == "O'Brien" and read.parameters == {"a": 2.5}
    assert read.x == GridRange(-1.0, 3.0, 7) and read.y == GridRange(0.1, 10.0, 4, "log")
    assert parse_function_sql('SELECT "x" FROM "t"') is None

    series = f'SELECT "x" AS "x", "y" AS "y" FROM ({function_sql("Line", {"a": 1.0}, GridRange(0, 1, 2))}) AS "_src"'
    other = function_sql("Line", {"a": 9.0}, GridRange(0, 5, 3))
    rewritten = replace_function_sql(series, other)
    assert rewritten == f'SELECT "x" AS "x", "y" AS "y" FROM ({other}) AS "_src"'


def test_the_editor_shows_a_function_and_gives_it_back(qapp) -> None:
    editor = FunctionEditor()
    try:
        assert [editor.tabs.tabText(i) for i in range(editor.tabs.count())] == ["Formula", "Parameters", "Range"]
        editor.load(LINE, values={"b": -3.0}, x=GridRange(0, 1, 3))
        assert editor.parameters() == {"a": 2.0, "b": -3.0}  # the chart's own, the stored ones for the rest
        assert editor.x_range() == GridRange(0, 1, 3) and editor.y_range() is None and editor.ndim() == 1
        assert not editor._ranges.isRowVisible(editor.y)
        data = editor.function_data()
        assert (data["function_name"], data["symbols"], data["code_string"], data["values"]) == \
            ("Line", "a,b", "a*x + b", {"a": 2.0, "b": -3.0})

        # The rows follow the declaration: a parameter added, one taken away.
        editor.symbols.setText("a, b, c")
        assert editor.parameters() == {"a": 2.0, "b": -3.0, "c": 1.0}
        editor.symbols.setText("a")
        assert editor.parameters() == {"a": 2.0}
        assert editor.problem() == "b: not among the parameters."
        editor.text.setPlainText("a*sin(x) + y")  # sympy's sin is not a parameter; y makes it a surface
        assert editor.problem() == "" and editor.ndim() == 2 and editor._ranges.isRowVisible(editor.y)

        editor.set_formula_visible(False)  # New Chart: values and range only
        assert not editor.tabs.isTabVisible(0)
    finally:
        editor.deleteLater()


def test_code_declares_its_parameters_in_its_def(qapp) -> None:
    editor = FunctionEditor()
    try:
        editor.load({"function_name": "Bump", "source_type": "code", "symbols": "",
                     "code_string": "def Bump(h, w):\n    f_numeric = h * np.exp(-(x / w) ** 2)\n    return f_numeric\n"})
        assert editor.declared_parameters() == ["h", "w"] and list(editor.parameters()) == ["h", "w"]
        assert not editor._form.isRowVisible(editor.symbols)  # a def lists its own
        editor.text.setPlainText("def Other(h):\n    return h\n")
        assert "def Bump" in editor.problem()
    finally:
        editor.deleteLater()
