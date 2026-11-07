"""How well an assistant provider does the work: 20 requests on the Palmer penguins, each scored.

    .python/bin/python3 dev/tools/assistant_eval.py ollama           # one provider (ollama, apple, claude, demo)
    .python/bin/python3 dev/tools/assistant_eval.py ollama --only 3  # one case, to look at it
    .python/bin/python3 dev/tools/assistant_eval.py ollama --level read_only

Run by hand, one provider at a time and never beside the test suite: a local model takes the memory of a small
Mac. Each case runs on a fresh copy of the penguin demo, in a fresh conversation, with the same system prompt and
tools the panel gives (app.ai.assistant); the tools run without a window and every change is accepted.

A case passes when a tool it accepts ran without error, with the columns it names among its arguments, and its
outcome is there: a figure drawn, a number in the answer, or an answer at all. Printed: each case's verdict and why,
then the score; written: dev/test_results/assistant_eval/<provider>-<date>.json (calls, answers, times, memory).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEMO = ROOT / "demo" / "Penguins - species compared across four chart types.dhub"
OUT = ROOT / "dev" / "test_results" / "assistant_eval"
#: The steps one request may take (the panel allows 25; a small model that loops is stopped sooner).
MAX_TURNS = 8
PROVIDERS = {"ollama": "Ollama", "apple": "Apple Foundation Models", "claude": "Claude", "demo": "Demo"}


@dataclass(frozen=True)
class Case:
    request: str
    tools: tuple[str, ...]  # any of these, run without error
    columns: tuple[str, ...] = ()  # each named in that call's arguments
    outcome: str = "answer"  # "figure", "number" or "answer"


CASES: tuple[Case, ...] = (
    Case("Which tables are in this project?", ("list_tables",), (), "answer"),
    Case("Describe the penguins table.", ("describe_table",), (), "answer"),
    Case("How many penguins are there of each species?", ("query", "describe_table", "compare_groups"),
         ("species",), "number"),
    Case("What is the average body mass of the penguins?", ("query", "describe_table"), ("body_mass_g",), "number"),
    Case("Is body mass different between species?", ("compare_groups", "run_analysis"),
         ("body_mass_g", "species"), "number"),
    Case("Do the islands differ in flipper length?", ("compare_groups", "run_analysis"),
         ("flipper_length_mm", "island"), "number"),
    Case("Are males heavier than females?", ("compare_groups", "run_analysis"), ("body_mass_g", "sex"), "number"),
    Case("What affects body mass the most?", ("find_impact", "run_analysis"), ("body_mass_g",), "answer"),
    Case("Does bill length depend on species and sex?", ("find_impact", "run_analysis"),
         ("bill_length_mm", "species", "sex"), "answer"),
    Case("Draw a box plot of body mass by species.", ("create_chart",), ("body_mass_g", "species"), "figure"),
    Case("Make a histogram of flipper length.", ("create_chart",), ("flipper_length_mm",), "figure"),
    Case("Plot bill length against bill depth.", ("create_chart",), ("bill_length_mm", "bill_depth_mm"), "figure"),
    Case("Show a scatter of flipper length versus body mass coloured by species.", ("create_chart",),
         ("flipper_length_mm", "body_mass_g", "species"), "figure"),
    Case("Which chart types show a distribution?", ("list_charts",), (), "answer"),
    Case("Which analyses can I run on this data?", ("list_analyses",), (), "answer"),
    Case("Run descriptive statistics on body mass.", ("run_analysis",), ("body_mass_g",), "number"),
    Case("Is there a correlation between flipper length and body mass?", ("run_analysis", "query"),
         ("flipper_length_mm", "body_mass_g"), "number"),
    Case("What is the heaviest penguin's species and island?", ("query",), ("body_mass_g",), "answer"),
    Case("How many penguins have a missing sex?", ("query", "describe_table"), ("sex",), "number"),
    Case("Fit a straight line of body mass on flipper length.", ("run_analysis",),
         ("body_mass_g", "flipper_length_mm"), "number"),
)


@dataclass
class Result:
    index: int
    request: str
    passed: bool = False
    why: str = ""
    calls: list[dict] = field(default_factory=list)
    answer: str = ""
    figures: int = 0
    seconds: float = 0.0
    error: str = ""


def free_memory_mb() -> int:
    """Free and inactive memory, in MB (vm_stat); 0 where there is no vm_stat."""
    try:
        text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    page = int(re.search(r"page size of (\d+)", text).group(1)) if "page size" in text else 16384
    pages = sum(int(m) for m in re.findall(r"Pages (?:free|inactive):\s+(\d+)", text))
    return pages * page // (1024 * 1024)


def judge(case: Case, result: Result) -> tuple[bool, str]:
    good = [call for call in result.calls if call["name"] in case.tools and not call["error"]]
    if not good:
        tried = ", ".join(f"{call['name']}{' (error)' if call['error'] else ''}" for call in result.calls) or "none"
        return False, f"no {'/'.join(case.tools)} without error (calls: {tried})"
    arguments = " ".join(json.dumps(call["input"]).lower() for call in good)
    missing = [column for column in case.columns if column.lower() not in arguments]
    if missing:
        return False, f"columns not used: {', '.join(missing)}"
    if case.outcome == "figure" and not result.figures:
        return False, "no figure drawn"
    if case.outcome == "number" and not re.search(r"\d", result.answer):
        return False, "no number in the answer"
    if not result.answer.strip() and case.outcome != "figure":
        return False, "no answer"
    return True, "ok"


def run_case(provider, index: int, case: Case, level: str, api_key: str, folder: Path) -> Result:
    from app.ai.assistant import system_prompt
    from app.ai.tools import AssistantTools
    from app.data.sqlite_repo import SqliteRepo
    from app.plugin_api.ai.provider import ACCURATE, ToolResult

    copy = folder / f"case{index}.dhub"
    shutil.copy2(DEMO, copy)
    repo = SqliteRepo(db_path=copy)
    result = Result(index, case.request)
    made: list[int] = []
    tools = AssistantTools(repo, confirm=lambda _question: True, changed=lambda figures: made.extend(figures),
                           current_figure=lambda: None)
    if provider.MAX_RESULT:
        tools.max_result = provider.MAX_RESULT
    if provider.READ_ONLY:
        level = "read_only"
    model = dict(provider.models())[ACCURATE]
    started = time.monotonic()
    try:
        session = provider.start(api_key=api_key, system=system_prompt(provider, tools, level),
                                 tools=tools.specs(level))
        pieces: list[str] = []
        turn = session.send(model=model, user_text=case.request, tool_results=(), on_text=pieces.append,
                            cancel=threading.Event())
        for _step in range(MAX_TURNS):
            if turn.stop != "tools":
                break
            results = []
            for call in turn.tool_calls:
                content, is_error = tools.run(call.name, call.input, level)
                result.calls.append({"name": call.name, "input": dict(call.input), "error": is_error,
                                     "result": content[:300]})
                results.append(ToolResult(call.id, content, is_error))
            turn = session.send(model=model, user_text=None, tool_results=tuple(results), on_text=pieces.append,
                                cancel=threading.Event())
        result.answer = "".join(pieces)
    except Exception as error:  # the provider's failure is the case's verdict
        result.error = f"{type(error).__name__}: {error}"
    finally:
        result.seconds = round(time.monotonic() - started, 1)
        result.figures = len(made)
        repo.close()
    result.passed, result.why = (False, result.error) if result.error else judge(case, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", choices=sorted(PROVIDERS))
    parser.add_argument("--only", type=int, default=0, help="run only case N (1-20)")
    parser.add_argument("--level", default="analyses", choices=["read_only", "analyses", "everything"])
    options = parser.parse_args()

    from app.utils.qt_plugins import ensure_visible_plugins

    ensure_visible_plugins()  # macOS may have flagged Qt's plugins hidden
    from PySide6.QtWidgets import QApplication

    from app.ai import settings
    from app.widgets.assistant_panel import _load_provider

    app = QApplication.instance() or QApplication([])  # some analyses build their dialog off screen
    del app
    provider = _load_provider(PROVIDERS[options.provider])
    if provider is None or provider.Name != PROVIDERS[options.provider]:
        print(f"No provider {PROVIDERS[options.provider]} installed.")
        return 1
    if provider.unavailable():
        print(f"{provider.Name} is not available: {provider.unavailable()}")
        return 1
    api_key = settings.api_key(provider.KEY_NAME) if provider.KEY_NAME else ""

    cases = [(index, case) for index, case in enumerate(CASES, 1) if not options.only or index == options.only]
    memory_before = free_memory_mb()
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="assistant_eval_") as folder:
        for index, case in cases:
            result = run_case(provider, index, case, options.level, api_key, Path(folder))
            results.append(result)
            mark = "PASS" if result.passed else "FAIL"
            print(f"{index:2}. {mark} {result.seconds:6.1f}s  {case.request}")
            if not result.passed:
                print(f"     {result.why}")
    passed = sum(result.passed for result in results)
    print(f"\n{provider.Name} ({dict(provider.models())}): {passed} of {len(results)} "
          f"({100 * passed // max(1, len(results))} %), free memory {memory_before} -> {free_memory_mb()} MB")

    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / f"{options.provider}-{datetime.now():%Y%m%d-%H%M}.json"
    report.write_text(json.dumps({"provider": provider.Name, "models": dict(provider.models()),
                                  "level": options.level, "passed": passed, "cases": len(results),
                                  "results": [asdict(result) for result in results]}, indent=1, default=str),
                      encoding="utf-8")
    print(f"Written: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
