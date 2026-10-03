#import "helpers.typ": *

= Settings, styles and the log <settings-chapter>

== Settings

*Settings* (the ChartLibre menu on macOS, the rail on Windows and Linux) holds the preferences of this installation — they belong to the computer, not to a project:

#defs(
  [*App style*], [The look of the interface. The platform's native style, plus any Qt styles installed on the system (Breeze, Fusion, ...) and *Browse…* for a Qt stylesheet (`.qss`) of your own.],
  [*Language*], [*Auto* follows the operating system's language, falling back to English. Each language is listed under its own name: English, *Italiano* and *Français* are complete; *Español* and *Deutsch* are only started and still read mostly English.],
  [*Default export format*], [The format *Save figure* proposes: PNG, JPEG, SVG or PDF.],
  [*SQL guard*], [*Refuse SQL that would change the database* (on by default, shown in red): `UPDATE`, `DELETE`, `DROP` and the like are refused when a query is saved or typed and when a project is opened. Switch it off only if you know why you need to.],
  [*DeepL API key*], [Used by *Edit Localization* to translate the interface with DeepL. The key is kept on this computer only. Without one, Edit Localization translates with Google or MyMemory. Free-plan keys end in `:fx`.],
)

The style and the language are applied the next time ChartLibre starts, as the window says: changing them in a running application would leave half the window in the old style or language.

== Chart styles

A *Matplotlib style* sets the base look of every chart in a figure: colours, fonts, line widths, grid, tick direction. Each figure picks its style in *Figure options → Style*: Matplotlib's built-in styles, the *SciencePlots* styles for journal figures, or a `.mplstyle` file of your own.

The *Edit* button beside the style opens the *Matplotlib Style Editor*:

- a *style stack* to combine several styles (each overriding the previous one);
- a *parameter editor* to add, change or reset individual settings (rcParams), with an editor suited to each — colour picker, check box, number, line style, marker, colormap, font list;
- a live *preview* of the current figure with the edited style;
- *Save* writes the result as a `.mplstyle` file, which can then be chosen for any figure.

== The log <log>

The *Log viewer* (on the *Developer* page) shows what ChartLibre has done and what went wrong: projects opened, imports, operations, rows a chart could not use, errors. It is the first place to look when a chart is emptier than expected, and the thing to attach to a bug report.

*Clear log* (in red) empties the log after asking for confirmation.

== Credits

*Credits* (in the ChartLibre menu on macOS, the *Menu* button elsewhere) names the application and its version, and the open-source libraries it is built on — Qt and PySide6, Matplotlib, NumPy, pandas, SciPy, statsmodels, scikit-learn and others, each with its installed version and licence. It also credits the work of others that ships with ChartLibre or that it draws on — the SciencePlots styles by John D. Garrett, the system icon sets — and the sources of the demo data: the Palmer penguins, NOAA's Mauna Loa CO₂ record, WDC-SILSO's sunspot number, NASA's GISTEMP, the USGS earthquake catalogue, the UCI Yeast dataset, Anscombe's quartet, Will Burtin's antibiotic data, the NIST StRD reference data sets and others. Each name links to its source.
