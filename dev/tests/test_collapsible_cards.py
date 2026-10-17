"""H-10: a titled frame can open and close from its title; the Query Builder's Build form follows its remembered combo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from app.styles.style import TitledCard


def test_a_collapsible_card_opens_and_closes_from_its_arrow_and_its_title(qapp) -> None:
    host = QWidget()
    card = TitledCard(host, "More options", collapsible=True, collapsed=True)
    host.show()
    qapp.processEvents()
    assert card.collapsed and not card.card.isVisibleTo(host)
    assert card.disclosure.arrowType() == Qt.ArrowType.RightArrow
    card.disclosure.click()
    assert not card.collapsed and card.disclosure.arrowType() == Qt.ArrowType.DownArrow
    release = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(2, 2), QPointF(2, 2),
                          Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(card.title_label, release)
    assert card.collapsed
    plain = TitledCard(host, "Always open")
    assert plain.disclosure is None and not plain.collapsed
    host.close()


def test_the_query_builder_build_form_follows_its_remembered_operation(qapp, tmp_path: Path) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.dialogs import query_builder_dialog as qb
    from app.utils.dialog_state import save_dialog_state

    repo = SqliteRepo(db_path=tmp_path / "q.dhub")
    repo.import_dataframe(pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]}), table_name="t", normalize_columns=False)
    first = qb.QueryBuilderDialog(repo)
    first._operation_combo.setCurrentIndex(len(qb.FORMS) - 1)
    save_dialog_state(first, qb.STATE_KEY)
    first.close()
    again = qb.QueryBuilderDialog(repo)
    try:
        assert again._operation_combo.currentIndex() == len(qb.FORMS) - 1
        assert again._operation_stack.currentIndex() == len(qb.FORMS) - 1
    finally:
        again.close()
        repo.close()


def test_select_in_build_writes_the_old_snippets_and_clear_empties_the_editor(qapp, tmp_path: Path) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.dialogs import query_builder_dialog as qb
    from app.dialogs.query_operations import SelectForm

    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    repo.import_dataframe(pd.DataFrame({"g": ["a", "b", "a"], "v": [3.0, 1.0, 2.0]}), table_name="t",
                          normalize_columns=False)
    dialog = qb.QueryBuilderDialog(repo)
    try:
        dialog._operation_combo.setCurrentIndex(0)
        form = dialog._operation_forms[0]
        assert isinstance(form, SelectForm)
        assert [form.columns_list.item(i).text() for i in range(form.columns_list.count())] == ["g", "v"]
        form.where_column.setCurrentText("g")
        form.value.setText("a")
        form.order_column.setCurrentText("v")
        assert dialog.write_operation_sql()
        assert dialog.sql == 'SELECT *\nFROM "t"\nWHERE "g" = \'a\'\nORDER BY "v" ASC'
        assert dialog._result_view.model().rowCount() == 2
        dialog._editor.clear()
        assert dialog.sql == ""
    finally:
        dialog.close()
        repo.close()
