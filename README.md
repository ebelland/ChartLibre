# ChartLibre

**A free, open-source desktop application for turning data into scientific charts — without writing code, and without a licence fee.**

Scientific charting software is usually the expensive part of a small lab's or a student's toolchain. ChartLibre covers the everyday ground those packages are bought for — import a dataset, query it, plot it, fit it, export it at publication resolution — as a plain MIT-licensed desktop application you install and use for free, on Windows and macOS.¹

It is built on Matplotlib, so the charts are Matplotlib charts: the same output a Python script would produce, reached through a GUI instead of code.

![The ChartLibre main window](docs/manual/screenshot_main_window.png)

<table>
  <tr>
    <td><img src="docs/images/gallery_keeling.png" alt="The Keeling curve, CO2 at Mauna Loa since 1958"></td>
    <td><img src="docs/images/gallery_anscombe.png" alt="Anscombe's quartet on a 2x2 grid"></td>
    <td><img src="docs/images/gallery_ishikawa.png" alt="An Ishikawa (fishbone) diagram"></td>
  </tr>
  <tr>
    <td><img src="docs/images/gallery_spectrum.png" alt="The sunspot record's amplitude spectrum, peaking at 11 years"></td>
    <td><img src="docs/images/gallery_penguins.png" alt="Violin plots of penguin flipper length by species"></td>
    <td><img src="docs/images/gallery_surface.png" alt="A 3D surface on a regular grid"></td>
  </tr>
</table>

<sub>All six are demo projects that ship with the application, drawn by ChartLibre itself in its default style.</sub>

## Download

| [**⬇ Download for Mac**](https://github.com/ebelland/ChartLibre/releases/latest/download/ChartLibre-Mac.dmg) | [**⬇ Download for Windows**](https://github.com/ebelland/ChartLibre/releases/latest/download/ChartLibre-Windows-Setup.exe) |
|:---:|:---:|
| Macs with an Apple chip (M1, M2, M3, M4 ...) | Windows 10 and 11 |

Free, no account, nothing else to install. Then follow the [installation steps](#installation) below — two minutes, once.

---

## Why it exists

If you need to plot measurements, fit a curve, find peaks in a spectrum or build a control chart, the usual options are a commercial package (Origin, SigmaPlot, Prism, Igor Pro, KaleidaGraph), a spreadsheet that was never meant for it, or writing Matplotlib by hand every time.

ChartLibre aims at the first of those without the price tag:

- **Free of charge, and free to modify** — MIT licence, no tiers, no seats, no activation, no account.
- **No code required** — every chart, fit and transformation is a dialog. The data stays queryable in SQL when you want that power.
- **Your file stays yours** — a project is one `.dhub` file (a SQLite database) holding *both* the data and every chart definition built on it. Copy it, email it, commit it to git. No proprietary container, no cloud, no lock-in: any SQLite tool can read the data back out.
- **Publication output** — export to PNG, JPEG, **SVG** and **PDF** at a DPI you choose.

> **Status: early, and not thoroughly tested.** ChartLibre is at version 0.1.0 and under active development. The feature list below is what is implemented and covered by the automated test suite, not a roadmap — but it has had little testing by real users on real data yet. Expect rough edges, and check results that matter.
>
> **Collaborators wanted.** Testers, bug reports, fixes, translations, a Linux check, packaging: see [Contributing](#contributing).

> **How it was made.** ChartLibre was written mostly by [Claude](https://www.anthropic.com/claude), Anthropic's AI model, working in conversation with the author, who designed it, set its direction and tested it. The *Credits* window in the application says the same.

---

## What it does

### Charts — 35 types

| Family | Types |
|---|---|
| **Pairwise (x, y)** | Bar, Horizontal Bar, Broken Bar (+ vertical), Scatter, Fill Between, Stack Plot, Stem, Table, Text, Time Series, Timeline |
| **Statistical distributions** | Histogram, Box, Violin, ECDF, Stairs, Pareto, Pie, Hexbin, Event Plot |
| **Gridded data** | Contour, Heatmap, Quiver, Stream Plot, Wind Barbs |
| **Irregularly gridded** | Contour (scattered), Triangular Mesh, Triangular Colour Mesh |
| **3D and volumetric** | Surface, Surface (scattered), 3D Scatter, 3D Line, 3D Bar |
| **Diagrams** | Fishbone (Ishikawa) |

Multiple axes per figure, multi-axis layout presets (shared scale, main + secondary, overlapping), annotations and reference lines, and an interactive canvas for measuring, annotating and selecting points.

### Analysis — 19 series operations

Peaks · Roots · Calculus · Statistics · Outliers · Clustering · Regression · GP Regression · Transform · Decomposition · Control Chart · Geometry · Smoothing · Spectral Analysis · Filtering · Baseline Correction · Fit · Interpolation · Function

Each previews its result on the chart before anything is committed, and a long calculation runs in the background with a *Stop* button.

![The Series Operations page](docs/manual/screenshot_series_operations.png)

Highlights:

- **Fit** — Gaussian, exponential, polynomial, your own functions; standard errors, t, p and 95 % intervals for every parameter, confidence and prediction bands on the chart, residual and measured-vs-fit charts. Validated against the NIST StRD certified values.
- **Statistics** — descriptive statistics, normality, one-sample and paired tests, independent samples (Welch, Student, Mann-Whitney, Cohen's d), one-way ANOVA with Welch, Kruskal-Wallis and Tukey HSD, correlation, distribution fitting.
- **Geometry** — rotate, translate, roto-translate, scale, mirror or shear a series in 2D or 3D; the result is a live query on the original data, not a copy.
- **Filtering / Spectral Analysis** — Butterworth, Chebyshev, Bessel, FIR; PSD, coherence, cross-correlation, Hilbert envelope.
- **Control Chart** — I-MR, X-bar-R, X-bar-S, p, np, c, u with rule violations flagged.
- **Outliers / Regression / Clustering** — classical criteria plus scikit-learn models (Isolation Forest, LOF, One-Class SVM, RANSAC, Huber, Random Forest, k-means, DBSCAN).
- **Surfaces** — Fit, Peaks, Roots and Calculus also work on `z = f(x, y)` series: surface models, 2D local extrema, traced level curves, gradient and volume.

### Data in and out

- **Import** — CSV/TSV/TXT, Excel (`.xlsx`, `.xlsm`, `.xls`), JSON, XML, clipboard paste, web URLs, and databases: SQLite, another `.dhub`, **PostgreSQL** and **MySQL** (a table or your own `SELECT`).
- **Re-read** — any imported table keeps its link and can be refreshed from the original source in one click. Passwords are never stored in the project file.
- **Export** — tables to CSV or Excel; charts to PNG, JPEG, SVG or PDF.
- **SQL Query Builder** — write, test and save queries that behave like tables anywhere a table is accepted.

### Make it yours

- **Matplotlib style editor** — edit rcParams and save `.mplstyle` files.
- **Extensible without forking** — drop a Python file into `user/charts/`, `user/series_operations/` or `user/functions/` and it appears in the application. GUI scaffolding tools generate a working skeleton for each.
- **Localisation** — English, Italian and French (complete), with a built-in catalogue editor for adding more.

---

## Installation

You do not need Python, a GitHub account or anything else: download one file, open it, done. Everything ChartLibre needs is inside it.

### On a Mac

1. **Download** [ChartLibre-Mac.dmg](https://github.com/ebelland/ChartLibre/releases/latest/download/ChartLibre-Mac.dmg) and double-click it in your *Downloads* folder. A window opens with a **ChartLibre** folder and an **Applications** folder.
2. **Drag** the ChartLibre folder onto the Applications folder, and wait for the copy to finish.
3. **Open** *Applications → ChartLibre → ChartLibre.app* with a double-click.
4. **The first time only**, the Mac says *Apple could not verify "ChartLibre.app" is free of malware*. This is because ChartLibre is free software and is not registered with Apple — it does not mean anything is wrong. To allow it:
   - click **Done**;
   - open **System Settings** (the grey gear in the Dock, or  menu → System Settings), then **Privacy & Security**;
   - scroll to the bottom: next to *"ChartLibre.app" was blocked*, click **Open Anyway**, and type your Mac password;
   - click **Open**.

   From then on ChartLibre opens with a double-click like any other application. To keep it in the Dock, right-click its icon while it runs and choose *Options → Keep in Dock*.

<details>
<summary>The "Open Anyway" button is not there</summary>

It appears only for about an hour after the warning. Double-click ChartLibre.app again, click **Done**, and look again in *Privacy & Security*. Or, if you are comfortable with the Terminal app, paste this line there and press Return — it lets the Mac open ChartLibre without asking:

```bash
xattr -dr com.apple.quarantine /Applications/ChartLibre
```
</details>

### On Windows

1. **Download** [ChartLibre-Windows-Setup.exe](https://github.com/ebelland/ChartLibre/releases/latest/download/ChartLibre-Windows-Setup.exe) and open it from your *Downloads* folder.
2. **The first time only**, Windows may show a blue box, *Windows protected your PC*, because ChartLibre is not registered with Microsoft. Click **More info**, then **Run anyway**.
3. **Follow the installer**: choose the language, accept the licence, tick *Create a desktop shortcut* if you want one, and click **Install**. No administrator password is needed.
4. **Open** ChartLibre from the Start menu (or the desktop icon).

To remove it: *Settings → Apps → Installed apps → ChartLibre → Uninstall*. Your own projects are kept.

<details>
<summary>Without installing: the portable version</summary>

[ChartLibre-Windows-Portable.zip](https://github.com/ebelland/ChartLibre/releases/latest/download/ChartLibre-Windows-Portable.zip) is the same application in a folder: unzip it anywhere (a USB stick works too) and double-click `ChartLibre.exe` inside. Delete the folder to remove it.
</details>

### Your first steps

When ChartLibre opens it asks whether to create a new project, open one, or **load a demo**. Start with one of the twenty-six demos: each opens a project with data and charts already made, which is the quickest way to see what ChartLibre does. The [user manual](docs/manual/user_manual.pdf) explains every window.

<details>
<summary>Mac with an Intel processor, Linux, or the source code</summary>

The ready-made downloads are for Apple-chip Macs and Windows. On an Intel Mac or on Linux, use the source folder instead — it downloads what it needs the first time it starts:

1. On this page, click the green **Code** button, then **Download ZIP**, and unzip it. The `ChartLibre` folder can live anywhere.
2. Start it:
   - **Intel Mac**: double-click `ChartLibre.app` (the first time, allow it as described for the Mac above);
   - **Linux**: open a terminal in the folder and run `./ChartLibre.sh`. It also adds ChartLibre to the applications menu.

The first launch asks, then downloads its own copy of Python (a standalone build from [python-build-standalone](https://github.com/astral-sh/python-build-standalone), checked against its SHA-256 before it runs) and the scientific libraries into the folder itself — about 400 MB to download and 1.7 GB on disk, a few minutes, an internet connection. Every launch after that opens the application straight away, offline. Nothing is installed anywhere else: deleting the folder removes ChartLibre.

On Linux, Qt needs a few system libraries a desktop usually has; if one is missing, the first launch names it and the command to install it (typically `sudo apt install libxcb-cursor0`).

**If ChartLibre stops starting**, delete the hidden `.python` folder inside the ChartLibre folder and start it again: it downloads Python and the libraries again.

With your own Python 3.11 or newer, from a terminal:

```bash
git clone https://github.com/ebelland/ChartLibre.git
cd ChartLibre
python3 install.py        # creates .venv and installs requirements.txt
.venv/bin/python3 main.py # on Windows: .venv\Scripts\python.exe main.py
```
</details>


---

## Documentation

| | |
|---|---|
| [**User manual**](docs/manual/user_manual.pdf) | The whole application, chapter by chapter (PDF). |
| [**Developer guide**](dev/DEVELOPMENT.md) | Architecture, the render pipeline, and how to add a chart type or an operation. |
| [`todo.txt`](todo.txt) | Known issues and what is being worked on, ranked. |

## Built with

[PySide6](https://doc.qt.io/qtforpython/) (Qt 6) · [Matplotlib](https://matplotlib.org/) · [NumPy](https://numpy.org/) · [pandas](https://pandas.pydata.org/) · [SciPy](https://scipy.org/) · [scikit-learn](https://scikit-learn.org/) · [statsmodels](https://www.statsmodels.org/) · SQLite

## Credits

ChartLibre also ships, or draws on, other people's work:

- **[SciencePlots](https://github.com/garrettj403/SciencePlots)** by John D. Garrett — the Matplotlib styles in `mplstyles/` (MIT, see [`mplstyles/LICENSE-SciencePlots`](mplstyles/LICENSE-SciencePlots)).
- **Icons**: [SF Symbols](https://developer.apple.com/sf-symbols/) on macOS and [Segoe Fluent Icons](https://learn.microsoft.com/windows/apps/design/style/segoe-fluent-icons-font) on Windows, both drawn from the system rather than shipped.
- **Demo data** in `dev/demo/sample data/`:
  [Palmer penguins](https://allisonhorst.github.io/palmerpenguins/) (Horst, Hill & Gorman; Palmer Station LTER, CC0) ·
  [Mauna Loa CO₂](https://gml.noaa.gov/ccgg/trends/) (NOAA GML and Scripps) ·
  [sunspot number](https://www.sidc.be/SILSO/) (WDC-SILSO, Royal Observatory of Belgium, Brussels, CC BY-NC 4.0) ·
  [GISTEMP v4](https://data.giss.nasa.gov/gistemp/) (NASA GISS) ·
  [earthquake catalogue](https://earthquake.usgs.gov/earthquakes/feed/) (USGS) ·
  [Yeast](https://archive.ics.uci.edu/dataset/110/yeast) (Kenta Nakai, UCI Machine Learning Repository, CC BY 4.0) ·
  [Anscombe's quartet](https://doi.org/10.1080/00031305.1973.10478966) (F. J. Anscombe, 1973) ·
  antibiotic effectiveness (Will Burtin, 1951).

The same list, with every library's installed version and licence, is in the application under *Credits*.

## Contributing

ChartLibre is looking for collaborators. The most useful help right now:

- **Use it on your own data** and [open an issue](https://github.com/ebelland/ChartLibre/issues) for anything wrong, confusing or missing.
- **Check results** against another tool you trust — a fit, a test, a filter.
- **Try it on Linux**, where it has never been launched.
- **Translate** it: the catalogue editor is built in (*Developer → Edit Localization*).
- **Package it** as a real installer for macOS and Windows.

Pull requests are welcome too. The test suite runs with:

```bash
python -m pytest dev/tests -q          # add QT_QPA_PLATFORM=offscreen when headless
```

Please read [`dev/DEVELOPMENT.md`](dev/DEVELOPMENT.md) first — it explains the conventions the codebase actually follows.

## Notes

¹ **Linux: ChartLibre is barely tested there.** `./ChartLibre.sh` downloads its own Python and libraries like the other launchers. Qt needs a few system libraries a desktop usually has; the first launch says which are missing (typically `sudo apt install libxcb-cursor0`). Reports either way are welcome.

## Licence

[MIT](LICENSE) © 2026 Enrico Bellandi. Free to use, modify and redistribute.

*Product names mentioned above are trademarks of their respective owners and are named only to describe the kind of tool ChartLibre is an alternative to.*
