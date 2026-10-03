#import "helpers.typ": *

= Getting started <getting-started>

== Installing ChartLibre

ChartLibre is a Python application. It needs *Python 3.11 or newer*, installed once from #link("https://www.python.org/downloads/")[python.org] (on Windows, tick *Add python.exe to PATH* in the installer).

Copy the `ChartLibre` folder anywhere you like, then start it:

#defs(
  [*macOS*], [Double-click `ChartLibre.app`.],
  [*Windows*], [Double-click `ChartLibre.exe` (or `ChartLibre.bat`).],
  [*Linux*], [Run `python3 install.py` once in the folder: it prepares ChartLibre and adds it to the applications menu. Then start it from the menu, or with `./ChartLibre.sh`. On Ubuntu and Debian, Python's `venv` module may need installing first: `sudo apt install python3-venv`.],
)

The first launch asks before installing the libraries ChartLibre needs (Qt, Matplotlib, NumPy, SciPy, ...). It installs them inside the folder itself, which needs an internet connection and takes a few minutes; every later launch starts straight away. Nothing is installed anywhere else, so deleting the folder removes ChartLibre completely.

#note[
  *The first time*, macOS may say it cannot check the app: open *System Settings → Privacy & Security* and click *Open Anyway* (or right-click the app and choose *Open*). Windows may show *Windows protected your PC*: click *More info*, then *Run anyway*.

  *If ChartLibre stops starting*, run `python3 install.py --recreate` in its folder: it throws away the installed libraries and installs them again.
]

== Opening a project

On first launch, or whenever no project is open, ChartLibre asks what to open:

#defs(
  [*New*], [Creates an empty `.dhub` project at a path you choose.],
  [*Open*], [Opens an existing `.dhub` file.],
  [*Load demo*], [Opens one of the demo projects (see below).],
)

The last project is reopened automatically next time, and the *File* page lists the recent ones.

== Saving, and undoing

A project is a database, and every change is written to it as it is made: there is no unsaved work to lose. The commands on the *File* page are:

#defs(
  [*Save*], [Makes sure every change is in the `.dhub` file itself right now (Ctrl+S / Cmd+S). Useful before copying the file elsewhere.],
  [*Save As…*], [Saves the project under a new name and switches to the copy, leaving the original untouched.],
)

*Undo* (in the main menu, and in the macOS Edit menu) takes back the last recorded action — an import, a table edit, a series operation, deleting a figure, hiding points, applying a template — up to the last ten, and names the action it is about to undo. Settings changed in Chart Options are not recorded: change them back by hand.

== Demo projects <demos>

*Load demo* (on the *File* page) opens a ready-made project with data and charts already built. Twenty-six demos ship with ChartLibre. The window lists them with a picture of each one's first chart and a short summary, so you can see what you are opening.

Between them they cover every chart family and give most series operations a dataset built for them: a sparse curve for *Interpolation*, a drifting spectrum for *Baseline Correction*, a three-tone signal for *Filtering* and *Spectral Analysis*, a Gaussian bump and a saddle for the surface operations, the NIST reference data sets for *Fit*, and a *Diagnostic plots* demo with Q-Q and P-P plots, a survival study, a forest plot, a mosaic and a two-factor experiment — whose notes suggest which *Statistics* model to try on each.

A demo opens as a copy in your home folder, under the demo's own name, so you can change it freely. Loading the same demo again replaces that copy with a fresh one.

#use[Exploring ChartLibre before you have data of your own; copying a chart's settings or a series' query into a real project.]
