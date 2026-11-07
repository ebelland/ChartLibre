"""ChartLibre Developer: the contributors' tools in one window - not part of the app, not packaged.

    python.nosync/bin/python3 dev/ChartLibreDeveloper.py

Three columns as the app's Project page: writing code (the creators, localization, the Qt style), checking it
(tests, NIST, size, chart previews) and shipping it (bundle, libraries, manuals). The long ones run one at a
time in the log below, with Stop. In English only: it is for people working on the code.
"""
from __future__ import annotations

import csv
import os
import shlex
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.qt_plugins import ensure_visible_plugins  # noqa: E402

ensure_visible_plugins()  # before Qt: macOS may have flagged the environment's Qt plugins hidden

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import apply_platform_style, ensure_icon_theme  # noqa: E402
from app.utils.config import get_value, set_value  # noqa: E402
from app.utils.machine_translation import DEEPL_KEY_SETTING  # noqa: E402
from app.widgets.nav_bar import NavButton, NavPanel  # noqa: E402

#: The one environment: .python, a link to python.nosync (out of iCloud), as the launchers use it.
PYTHON = str(ROOT / "python.nosync" / "bin" / "python3") if (ROOT / "python.nosync").exists() else sys.executable
RESULTS = ROOT / "dev" / "test_results"
JUNIT = RESULTS / "junit.xml"
#: Outside the Desktop: iCloud tags files there while codesign reads them (build_bundle --work).
BUILD_WORK = Path.home() / "Library" / "Caches" / "ChartLibre-build"


class DeveloperWindow(QMainWindow):
    """The tools as buttons in three columns; the log of the one running below them."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChartLibre Developer")
        self._process: QProcess | None = None
        self._finished: Callable[[int], None] | None = None
        self._result: Path | None = None

        columns = QWidget(self)
        row = QHBoxLayout(columns)
        row.setContentsMargins(12, 12, 12, 0)
        row.setSpacing(12)
        row.addWidget(self._code_column(), 1)
        row.addWidget(self._check_column(), 1)
        row.addWidget(self._ship_column(), 1)

        # The log as a titled card of the app's pages, its buttons in the title's row.
        log_page = NavPanel(self)
        log_box = QWidget(log_page)
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self._status = QLabel("Ready.", log_box)
        self._status.setProperty("muted", True)
        bar.addWidget(self._status, 1)
        self._open_button = QPushButton("Open result", log_box)
        self._open_button.setEnabled(False)
        self._open_button.clicked.connect(self._open_result)
        bar.addWidget(self._open_button)
        self._stop_button = QPushButton("Stop", log_box)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._stop)
        bar.addWidget(self._stop_button)
        log_layout.addLayout(bar)
        self._log = QPlainTextEdit(log_box)
        self._log.setReadOnly(True)
        self._log.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._log.setFont(QFont("Menlo" if sys.platform == "darwin" else "Consolas", 11))
        self._log.setMaximumBlockCount(20000)
        log_layout.addWidget(self._log)
        log_page.add_frame("Log", log_box, stretch=1)

        split = QSplitter(Qt.Orientation.Vertical, self)
        split.addWidget(columns)
        split.addWidget(log_page)
        # The page the app's panels sit on (the toolboxPage rule of each style sheet).
        for page in (split, columns, log_page):
            page.setProperty("toolboxPage", True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.resize(1180, 820)
        split.setSizes([400, 420])  # the columns whole, without a scroll bar

    # ------------------------------------------------------------------
    # The three columns
    # ------------------------------------------------------------------

    def _code_column(self) -> NavPanel:
        from dev.developer.data_operation_helper import DataOperationBuilderDialog
        from dev.developer.function_creator_helper import FunctionCreatorDialog
        from dev.developer.localization_helper import EditLocalizationDialog
        from dev.developer.renderer_helper import RendererHelperDialog

        page = NavPanel(self)
        page.add_buttons("Write code", [
            NavButton("edit_localization", lambda: EditLocalizationDialog(self).exec(), described=True),
            NavButton("data_operation_builder", lambda: DataOperationBuilderDialog(self).exec(), described=True),
            NavButton("function_creator", lambda: FunctionCreatorDialog(self).exec(), described=True),
            NavButton("renderer_helper", lambda: RendererHelperDialog(self).exec(), described=True),
            NavButton("settings", self._qss_editor, text="Qt style editor",
                      tooltip="Edit one of app/styles/*.qss and try it live on the app's widgets, light or dark.",
                      described=True),
        ])
        # The one secret here: kept in user.json on this computer, sent to nobody but DeepL.
        key = QLineEdit(str(get_value(DEEPL_KEY_SETTING, "") or ""), self)
        key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        key.setPlaceholderText("Free plan keys end in :fx")
        key.setToolTip("Used by Edit Localization to translate with DeepL; empty: Google or MyMemory.")
        key.editingFinished.connect(lambda: set_value(DEEPL_KEY_SETTING, key.text().strip()))
        page.add_frame("DeepL API key", key)
        return page

    def _qss_editor(self) -> None:
        from dev.developer.qss_editor import QssEditorDialog

        QssEditorDialog(self).exec()

    def _check_column(self) -> NavPanel:
        page = NavPanel(self)
        page.add_buttons("Check", [
            NavButton("run", lambda: self._run_tests(changed_only=False), text="Run the tests",
                      tooltip="The whole suite, 6 at a time; a report of what failed at the end.", described=True),
            NavButton("run", lambda: self._run_tests(changed_only=True), text="Run the changed tests",
                      tooltip="Only the test files changed since the last commit.", described=True),
            NavButton("column_stats", lambda: self._script("NIST report", ["dev/tools/nist_report.py"]),
                      text="NIST report",
                      tooltip="The correct digits ChartLibre gets on NIST's reference datasets.", described=True),
            NavButton("database_info", self._project_size, text="Project size",
                      tooltip="Lines of code and files now, and at each commit (dev/appsize.csv).", described=True),
            NavButton("reload", lambda: self._script("Chart previews", ["dev/tools/make_chart_previews.py"],
                                                     result=ROOT / "app" / "charts" / "previews"),
                      text="Make chart previews",
                      tooltip="Draw again the example of every chart type that New plot shows.", described=True),
        ])
        return page

    def _ship_column(self) -> NavPanel:
        page = NavPanel(self)
        page.add_buttons("Ship", [
            NavButton("export_rows", self._build_bundle, text="Build the bundle",
                      tooltip="The installable package for this computer (.dmg on a Mac), in dist/.", described=True),
            NavButton("reload", lambda: self._dependencies(apply=False), text="Check library updates",
                      tooltip="What a library update would change; nothing is installed.", described=True),
            NavButton("reload", lambda: self._dependencies(apply=True), text="Update Python libraries",
                      tooltip="Install the latest releases and write them into requirements.txt; then run the "
                              "tests.", described=True),
            NavButton("project_report", self._compile_manual, text="Compile the user manual",
                      tooltip="docs/manual/user_manual.typ to PDF with Typst.", described=True),
            NavButton("operation_history", self._developer_manual, text="Developer manual",
                      tooltip="dev/DEVELOPMENT.md: how the code is laid out and how to extend it.", described=True),
        ])
        return page

    # ------------------------------------------------------------------
    # Running one long job
    # ------------------------------------------------------------------

    def _start(self, title: str, program: str, arguments: list[str], *, result: Path | None = None,
               finished: Callable[[int], None] | None = None, env: dict[str, str] | None = None) -> None:
        """Run *program* in the log; one job at a time."""
        if self._process is not None:
            QMessageBox.information(self, "Busy", "Another job is running: stop it first.")
            return
        self._log.clear()
        self._log.appendPlainText(f"$ {shlex.join([program, *arguments])}\n")
        self._status.setText(f"{title}: running...")
        self._result = result
        self._finished = finished
        self._open_button.setEnabled(False)
        process = QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        for name, value in (env or {}).items():
            environment.insert(name, value)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(lambda code, _status, title=title: self._done(title, code))
        process.errorOccurred.connect(lambda error, title=title: self._failed(title, error))
        self._process = process
        self._stop_button.setEnabled(True)
        process.start(program, arguments)

    def _script(self, title: str, arguments: list[str], **kwargs) -> None:
        self._start(title, PYTHON, arguments, **kwargs)

    def _read(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertPlainText(text)
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _done(self, title: str, code: int) -> None:
        self._read()
        self._process = None
        self._stop_button.setEnabled(False)
        self._status.setText(f"{title}: {'done' if code == 0 else f'failed (exit {code})'}.")
        self._open_button.setEnabled(self._result is not None and self._result.exists())
        finished, self._finished = self._finished, None
        if finished is not None:
            finished(code)

    def _failed(self, title: str, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._log.appendPlainText(f"\nCould not start: {self._process.program() if self._process else ''}")
            self._process = None
            self._stop_button.setEnabled(False)
            self._status.setText(f"{title}: could not start.")

    def _stop(self) -> None:
        if self._process is not None:
            self._log.appendPlainText("\nStopping...")
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()

    def _open_result(self) -> None:
        if self._result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result)))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # The jobs
    # ------------------------------------------------------------------

    def _run_tests(self, *, changed_only: bool) -> None:
        files = ["dev/tests"]
        if changed_only:
            import subprocess

            out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
            files = sorted({line[3:].split(" -> ")[-1] for line in out.splitlines()
                            if line[3:].startswith("dev/tests/test_") and line[3:].endswith(".py")})
            if not files:
                QMessageBox.information(self, "Tests", "No test file changed since the last commit.")
                return
        RESULTS.mkdir(parents=True, exist_ok=True)
        self._start("Tests", PYTHON, ["-m", "pytest", "-q", "-n", "6", f"--junitxml={JUNIT}", *files],
                    env={"QT_QPA_PLATFORM": "offscreen"}, finished=lambda _code: self._test_report())

    def _test_report(self) -> None:
        """What failed, from the JUnit file pytest wrote: each test, where, and the first line of why."""
        try:
            suites = ET.parse(JUNIT).getroot()
        except (OSError, ET.ParseError):
            return
        cases = list(suites.iter("testcase"))
        failed = [case for case in cases if case.find("failure") is not None or case.find("error") is not None]
        skipped = sum(1 for case in cases if case.find("skipped") is not None)
        lines = ["", "=" * 70, f"Report: {len(cases) - len(failed) - skipped} passed, {len(failed)} failed, "
                 f"{skipped} skipped."]
        for case in failed:
            problem = case.find("failure") if case.find("failure") is not None else case.find("error")
            message = (problem.get("message") or "").splitlines()[0][:200] if problem is not None else ""
            where = f"{case.get('file') or case.get('classname', '').replace('.', '/') + '.py'}"
            lines.append(f"  FAILED {where}::{case.get('name')}")
            if message:
                lines.append(f"         {message}")
        self._log.appendPlainText("\n".join(lines))
        self._status.setText(f"Tests: {len(failed)} failed of {len(cases)}.")

    def _project_size(self) -> None:
        """Today's size, and the last commits' from dev/appsize.csv."""
        def count(folder: str) -> tuple[int, int]:
            files = list((ROOT / folder).rglob("*.py"))
            return sum(len(path.read_text(errors="replace").splitlines()) for path in files), len(files)

        app_lines, app_files = count("app")
        test_lines, test_files = count("dev/tests")
        lines = [f"Now: app/ {app_lines:,} lines in {app_files} files; dev/tests {test_lines:,} lines in "
                 f"{test_files} files.", ""]
        table = ROOT / "dev" / "appsize.csv"
        if table.exists():
            with table.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            lines.append("Last commits (dev/appsize.csv):")
            for row in rows[-15:]:  # the whole history is in the chart
                lines.append("  " + "  ".join(f"{key}={value}" for key, value in row.items()))
        self._log.clear()
        self._log.appendPlainText("\n".join(lines))
        self._status.setText("Project size.")
        if table.exists():
            self._size_chart(rows)

    def _size_chart(self, rows: list[dict[str, str]]) -> None:
        """Lines of code and of tests at each commit, against its date, in a window of its own."""
        from datetime import datetime

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        dates = [datetime.strptime(row["date"], "%Y-%m-%d %H:%M") for row in rows]
        figure = Figure(figsize=(8, 4.5), layout="constrained")
        axes = figure.add_subplot()
        for key, label in (("lines", "app/ lines"), ("code_lines", "app/ code lines"), ("test_lines", "test lines")):
            axes.plot(dates, [int(row[key]) for row in rows], label=label, linewidth=1.6)
        axes.set_ylabel("lines")
        axes.set_title(f"ChartLibre's size, {len(rows)} commits")
        axes.grid(alpha=0.3)
        axes.legend()
        figure.autofmt_xdate()
        dialog = QDialog(self)
        dialog.setWindowTitle("Project size")
        layout = QVBoxLayout(dialog)
        layout.addWidget(FigureCanvasQTAgg(figure))
        dialog.resize(900, 520)
        dialog.show()  # not modal: the log stays usable

    def _build_bundle(self) -> None:
        BUILD_WORK.mkdir(parents=True, exist_ok=True)
        self._script("Bundle", ["dev/tools/build_bundle.py", "--work", str(BUILD_WORK)], result=ROOT / "dist")

    def _dependencies(self, *, apply: bool) -> None:
        arguments = ["dev/tools/update_dependencies.py", "--python", PYTHON]
        if not apply:
            self._script("Library updates", [*arguments, "--dry-run"])
            return
        answer = QMessageBox.question(
            self, "Update libraries",
            "Install the latest releases into .python and write them into requirements.txt?\n"
            "The tests run afterwards; git restores requirements.txt if something breaks.")
        if answer == QMessageBox.StandardButton.Yes:
            self._script("Library update", [*arguments, "--write-pins", "--test"])

    def _compile_manual(self) -> None:
        import shutil

        typst = shutil.which("typst") or str(Path.home() / ".local" / "bin" / "typst")
        if not Path(typst).exists():
            QMessageBox.information(self, "Typst", "Typst is not installed: https://typst.app/open-source "
                                                   "(brew install typst).")
            return
        source = ROOT / "docs" / "manual" / "user_manual.typ"
        self._start("User manual", typst, ["compile", str(source)], result=source.with_suffix(".pdf"))

    def _developer_manual(self) -> None:
        path = ROOT / "dev" / "DEVELOPMENT.md"
        dialog = QDialog(self)
        dialog.setWindowTitle("Developer manual")
        layout = QVBoxLayout(dialog)
        view = QTextBrowser(dialog)
        view.setOpenExternalLinks(True)
        view.setMarkdown(path.read_text(encoding="utf-8"))
        layout.addWidget(view)
        edit = QPushButton("Open in the editor", dialog)
        edit.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
        layout.addWidget(edit, 0, Qt.AlignmentFlag.AlignRight)
        dialog.resize(900, 700)
        dialog.exec()


def main() -> int:
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    app.setApplicationName("ChartLibre Developer")
    # As ChartLibre.py starts: the same sheet, palette and icon theme.
    apply_platform_style(app)
    ensure_icon_theme()
    window = DeveloperWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
