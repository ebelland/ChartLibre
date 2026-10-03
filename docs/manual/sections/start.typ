#import "helpers.typ": *

= Getting started <getting-started>

== Installing ChartLibre

Nothing needs installing first — not even Python. Copy the `ChartLibre` folder anywhere you like (unzip it, if you downloaded it), then start it:

#defs(
  [*macOS*], [Double-click `ChartLibre.app`.],
  [*Windows*], [Double-click `ChartLibre.exe` (or `ChartLibre.bat`).],
  [*Linux*], [Open a terminal in the folder and run `./ChartLibre.sh`. The first launch also adds ChartLibre to the applications menu, so later it starts from there.],
)

The first launch asks, then downloads what ChartLibre needs into the folder itself: its own copy of Python and the scientific libraries (Qt, Matplotlib, NumPy, SciPy, ...) — about 400 MB to download, 1.7 GB on disk. It needs an internet connection and takes a few minutes; ChartLibre opens by itself when it is ready. Every later launch starts straight away and needs no internet. Nothing is installed anywhere else, so deleting the folder removes ChartLibre completely.

#note[
  *The first time*, macOS may say it cannot check the app: open *System Settings → Privacy & Security* and click *Open Anyway* (or right-click the app and choose *Open*). Windows may show *Windows protected your PC*: click *More info*, then *Run anyway*.

  *On Linux*, Qt needs a few system libraries that a desktop usually has. If one is missing, the first launch says which and how to install it — typically `sudo apt install libxcb-cursor0`.

  *If ChartLibre stops starting*, delete the hidden `.venv` folder inside the ChartLibre folder (on macOS, Cmd+Shift+. shows hidden files in Finder) and open ChartLibre again: it reinstalls the libraries.
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
