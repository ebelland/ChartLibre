# ChartLibre user manual

The manual is written in [Typst](https://typst.app); `user_manual.pdf` is the
built copy the application opens from its *Help* menu.

## Layout

| Path | What it is | Edited by |
|---|---|---|
| `user_manual.typ` | Page setup, cover, contents; includes the chapters in order | hand |
| `sections/*.typ` | One file per chapter | hand |
| `sections/helpers.typ` | Shared helpers: `#note`, `#defs`, `#roles`, `#mpl`, `#refs`, `#example`, … | hand |
| `sections/equivalents.typ` | Appendix: the same task in JMP, SPSS, Minitab, Origin, Excel | `python -m dev.manual.make_equivalents` |
| `screenshots/` | The application's windows | `python dev/manual/make_screenshots.py` |
| `examples/` | A chart for every chart type, data operation and analysis | `python dev/manual/make_examples.py` |
| `logos/` | ChartLibre and Matplotlib logos | hand |

Run the scripts from the project root. They open the application off screen
on copies of the demos and of `user.json`, so nothing of yours is touched.

## Build

```bash
typst compile docs/manual/user_manual.typ
```

or `typst watch docs/manual/user_manual.typ` to rebuild on every save.
Commit the PDF together with the sources it was built from.
