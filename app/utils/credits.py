"""Who and what this application is made of.

Everything here is read rather than written down: the libraries come from
``requirements.txt`` and their versions and licences from the installed
packages' own metadata.  A hand-kept list would be wrong within a release -
it would name a version nobody has installed, or miss a dependency added
last week - and being wrong is worse than being absent, because a credits
page is a licence statement as much as a thank-you.

The exception is what has no metadata to read - the style files copied into
the tree, the demo data, the system icons - listed by hand in ``BUNDLED``,
``ICONS`` and ``DATA`` because nothing else would name them at all.

No Qt here.  The dialog that shows it is one screen of formatting; what is
worth testing is this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

#: The project's dependency list, one directory above ``app``.
REQUIREMENTS_PATH: Path = Path(__file__).resolve().parent.parent.parent / "requirements.txt"

#: Whose project this is.
AUTHOR: str = "Artafasio Pippoz"

#: The assistants that wrote code alongside the author, and what each did.
#: Named because they did the work, and because someone reading the source a
#: year from now is entitled to know how it was written.
ASSISTANTS: tuple[tuple[str, str, str], ...] = (
    (
        "Claude",
        "Anthropic",
        "Series operations, the fit engine, the renderers and most of the "
        "test suite, written in conversation.",
    ),
    (
        "GitHub Copilot",
        "GitHub",
        "Completions throughout, and the first draft of much of the dialog "
        "plumbing.",
    ),
)

@dataclass(frozen=True, slots=True)
class Work:
    """Someone else's work that ships inside this project, or that it draws.

    The one list here that *is* written down: none of these is a package
    with metadata to read. The style files and themes were copied into the
    tree, the demo data was downloaded once and saved as CSV, and the icons
    are the operating system's own - so nothing else would ever name them.
    """

    name: str
    author: str
    use: str
    license: str
    url: str


#: Copied into the source tree, under their authors' licences. The MIT ones
#: ask for their notice to travel with the files; it does, beside them.
BUNDLED: tuple[Work, ...] = (
    Work(
        "SciencePlots",
        "John D. Garrett",
        "The Matplotlib styles in mplstyles/: science, journals, colours, "
        "languages.",
        "MIT",
        "https://github.com/garrettj403/SciencePlots",
    ),
    Work(
        "QSS themes",
        "Jaime A. Quiroga P. (GTRONICK)",
        "The optional Qt themes in app/styles/ and their images.",
        "MIT",
        "https://github.com/GTRONICK/QSS",
    ),
)

#: Drawn by name from the system at run time; nothing is copied.
ICONS: tuple[Work, ...] = (
    Work(
        "SF Symbols",
        "Apple",
        "Toolbar, menu and navigation icons on macOS.",
        "Apple's SF Symbols licence",
        "https://developer.apple.com/sf-symbols/",
    ),
    Work(
        "Segoe Fluent Icons",
        "Microsoft",
        "The same icons on Windows.",
        "Part of Windows",
        "https://learn.microsoft.com/windows/apps/design/style/segoe-fluent-icons-font",
    ),
    Work(
        "freedesktop icon themes",
        "The GNOME, KDE and Papirus projects",
        "The same icons on Linux (Adwaita, Breeze, Papirus).",
        "LGPL / GPL",
        "https://specifications.freedesktop.org/icon-naming-spec/latest/",
    ),
)

#: The real datasets under ``dev/demo/sample data/`` that the demo projects are
#: built from. The made-up ones - Lissajous curves, the synthetic signals,
#: the invented employees - are nobody's to credit.
DATA: tuple[Work, ...] = (
    Work(
        "Palmer penguins",
        "Horst, Hill and Gorman; Palmer Station LTER",
        "Bill, flipper and body-mass measurements of three species.",
        "CC0 1.0",
        "https://allisonhorst.github.io/palmerpenguins/",
    ),
    Work(
        "Mauna Loa CO2",
        "NOAA GML and Scripps Institution of Oceanography",
        "The Keeling curve, monthly since 1958.",
        "Public domain",
        "https://gml.noaa.gov/ccgg/trends/",
    ),
    Work(
        "Sunspot number",
        "WDC-SILSO, Royal Observatory of Belgium, Brussels",
        "Monthly mean sunspot number since 1749.",
        "CC BY-NC 4.0",
        "https://www.sidc.be/SILSO/",
    ),
    Work(
        "GISTEMP v4",
        "NASA Goddard Institute for Space Studies",
        "Global land-ocean temperature anomaly since 1880.",
        "Public domain",
        "https://data.giss.nasa.gov/gistemp/",
    ),
    Work(
        "Earthquake catalogue",
        "U.S. Geological Survey",
        "One month of the global feed, for the Gutenberg-Richter law.",
        "Public domain",
        "https://earthquake.usgs.gov/earthquakes/feed/",
    ),
    Work(
        "Yeast",
        "Kenta Nakai; UCI Machine Learning Repository",
        "Protein localisation measurements.",
        "CC BY 4.0",
        "https://archive.ics.uci.edu/dataset/110/yeast",
    ),
    Work(
        "Anscombe's quartet",
        "F. J. Anscombe, The American Statistician, 1973",
        "Four datasets with the same statistics and different pictures.",
        "Published data",
        "https://doi.org/10.1080/00031305.1973.10478966",
    ),
    Work(
        "Antibiotic effectiveness",
        "Will Burtin, 1951",
        "Sixteen bacteria against penicillin, streptomycin and neomycin.",
        "Published data",
        "https://en.wikipedia.org/wiki/Will_Burtin",
    ),
)


#: A requirement line: name, optional extras, optional specifier, optional
#: environment marker after a semicolon.
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"(?P<specifier>[^;]*)"
    r"(?:;\s*(?P<marker>.*))?$"
)


class DistributionMetadata(Protocol):
    """The two lookups this module needs from a package's metadata.

    Structural rather than nominal on purpose: what ``importlib.metadata``
    hands back is a private adapter class in some versions and an
    ``email.message.Message`` in others, and a test wants to pass a plain
    Message of its own. Naming the two methods says what is actually required
    and satisfies all three.
    """

    def get(self, name: str, failobj: Any = None) -> Any: ...

    def get_all(self, name: str, failobj: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class Requirement:
    """One line of requirements.txt, taken apart."""

    name: str
    specifier: str = ""
    marker: str = ""


@dataclass(frozen=True, slots=True)
class Package:
    """A dependency as the running installation actually has it."""

    name: str
    required: str
    installed: str
    summary: str
    license: str
    marker: str = ""

    @property
    def present(self) -> bool:
        """True when the package is installed in this interpreter."""
        return bool(self.installed)


def parse_requirements(text: str) -> list[Requirement]:
    """Return the requirements in *text*, in the order they are written.

    Comments, blank lines and the ``-r``/``-e`` directives are skipped; extras
    and environment markers are kept, because "macOS only" is part of the
    answer to what this application depends on.
    """
    requirements: list[Requirement] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        match = _REQUIREMENT_RE.match(line)
        if match is None:
            continue
        requirements.append(
            Requirement(
                name=match.group("name"),
                specifier=(match.group("specifier") or "").strip(),
                marker=(match.group("marker") or "").strip(),
            )
        )
    return requirements


def describe(name: str, *, required: str = "", marker: str = "") -> Package:
    """Return what the installed distribution says about itself.

    A package that is not installed still gets a row: an optional dependency
    that is absent is a fact about this installation, and hiding it would make
    the list look shorter than the project's actual dependencies.
    """
    try:
        distribution = metadata.metadata(name)
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        return Package(name, required, "", "", "", marker)

    return Package(
        name=str(distribution.get("Name") or name),
        required=required,
        installed=str(installed),
        summary=str(distribution.get("Summary") or "").strip(),
        license=license_of(distribution),
        marker=marker,
    )


def license_of(distribution: DistributionMetadata) -> str:
    """Return a licence short enough to put in a table cell.

    Three sources, in this order, because packaging has changed its mind
    twice: the SPDX ``License-Expression`` where a package has adopted it, the
    OSI classifier where it has not, and the free-text ``License`` field only
    when it is short. That last guard matters - matplotlib, SciPy, pandas and
    scikit-image all put their *entire licence text* in that field, and a
    table cell is not where anyone reads it.
    """
    expression = str(distribution.get("License-Expression") or "").strip()
    if expression:
        return expression

    for classifier in distribution.get_all("Classifier") or []:
        text = str(classifier)
        if text.startswith("License ::"):
            # "License :: OSI Approved :: BSD License" -> "BSD License"
            return text.rsplit("::", 1)[-1].strip()

    free_text = str(distribution.get("License") or "").strip()
    if free_text and "\n" not in free_text and len(free_text) <= 60:
        return free_text
    return ""


def packages(path: Path | None = None) -> list[Package]:
    """Return every dependency in requirements.txt, described."""
    source = path or REQUIREMENTS_PATH
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        # A packaged build may not ship requirements.txt. The credits then
        # name the author and the assistants and say nothing about libraries,
        # which is better than refusing to open.
        return []

    return [
        describe(requirement.name, required=requirement.specifier, marker=requirement.marker)
        for requirement in parse_requirements(text)
    ]
