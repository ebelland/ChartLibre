#import "helpers.typ": *

= Data <data-chapter>

== The Tables page

The *Tables* page lists the project's tables and saved queries (marked with a *Q*), and shows a preview of the selected one below the list.

The list has four columns:

#defs(
  [*(no title)*], [A narrow first column with a badge: a chain for a table linked to its source (it can be refreshed with *Update link*), *Q* for a saved query, nothing otherwise.],
  [*Table*], [The table's name.],
  [*Notes*], [Free text: double-click to write what the table is, where it came from, what was done to it. Notes are kept in the project and appear in reports.],
  [*Source*], [Where its data comes from: a file name for a file or web import, "connection → table" for a database import, the SQL itself for a saved query (the full text in the tooltip), nothing for a table with no link.],
)

Tables written by data operations start with `_`. *Show generated tables* hides or shows them, so the list can show only your own data.

Right-clicking a table offers:

#defs(
  [*Plot*], [Opens New plot on this table.],
  [*Query Builder…* / *Edit…*], [Opens the Query Builder; on a saved query, opens that query for editing.],
  [*Rename…*, *Duplicate table*, *Delete…*], [Manage the table. Several tables can be selected and deleted at once.],
  [*Export → CSV…* / *Export → XLSX…*], [Save the table as a CSV text file or an Excel workbook.],
  [*Update link*], [Re-read the table from its source — the same file, database table, query or URL — replacing its contents.],
  [*History…*], [The operations applied to or from this table (see @history).],
)

=== The preview

The preview shows the selected table's rows. Each column header carries the column's type in brackets: (S) text, (I) whole numbers, (F) decimal numbers. Right-click a cell for:

#defs(
  [*Copy*], [Copy the selected cells.],
  [*Statistics of a column*], [Count, empty cells, distinct values, minimum, maximum, mean and median of the column.],
  [*Histogram and statistics*], [With exactly one column selected: a new figure with that column's histogram, shown at once, and its statistics — mean with its 95% confidence interval, standard deviation, quartiles, skewness, kurtosis — in the figure's notes. Hidden rows are left out.],
  [*Hide rows* / *Select rows*], [Mark the rows equal to the selected cell, different from it, higher, lower, or where the column is empty — in the `Hide` column, which every chart skips, or in the `Selected` column, which only marks them. The rows stay in the table.],
  [*Export rows…*], [Save the selected rows as CSV or Excel.],
  [*New chart from selected columns*], [Opens New plot with the selected columns as x, y and z.],
  [*Edit table…*], [Opens the table editor.],
)

== Importing data <importing>

*Import* (on the *File* page) opens the import window with four sources:

#defs(
  [*Open*], [A local file: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xlsm`, `.xls`, `.json`, `.xml`. For a workbook with several sheets, you choose the sheet.],
  [*Paste*], [Whatever table is on the clipboard — for example cells copied from a spreadsheet.],
  [*Database*], [A table, or the result of a query, from another database: SQLite, another ChartLibre project, PostgreSQL or MySQL.],
  [*Web*], [Whatever an http(s) address returns — a CSV download, a JSON API, a spreadsheet.],
)

Whatever the source, the window shows a preview before anything is written, and the left panel sets how it is read:

#defs(
  [*Source*], [Where the data comes from.],
  [*Read options*], [The destination table name, the header row, rows to skip, the delimiter and the text encoding. These describe how to read *text*, so they are disabled for a database source, which has its own column types.],
  [*Columns*], [The type of each column (number, text, date, ...) — change it when a column was guessed wrong.],
)

*Database* opens its own window: pick an engine, give a file path or a host, port, user and password (*Connect* then lists the server's databases, and picking one lists its tables), choose a table — or tick *Use a query* and write a `SELECT` for a join, a filter or an aggregate — and confirm. The last connection is remembered, except its password.

*Web* has a menu of ready-made public datasets grouped by subject. The menu is the file `web_sources.json` in the ChartLibre folder, beside `config.json`: a list of entries with a `name`, a `url`, and optionally a `category` and a `description`. Open it in any text editor to add, change or remove datasets — the menu reads it again each time it opens. A mistake in the file (a missing comma, say) leaves its entries out and is named in the log; it never stops the import window. Picking one fills in the address without downloading, so it can be checked first; *Fetch* downloads it. *Add source* saves an address of your own to the menu, and *Delete source* removes one you added.

#note[
  Importing is a one-time read: the table does not change when the original file, database or web page changes. To refresh it, right-click the table and choose *Update link*. Every source except a paste can be refreshed this way.
]

#note[
  A PostgreSQL or MySQL password is never saved — not in the project, not anywhere on disk — because a `.dhub` file is easily copied or shared. *Update link* asks for it again. *Web* only fetches `http://` and `https://` addresses, never a local file.
]

== Editing a table by hand <table-editor>

The preview is read-only. To change the data, right-click it and choose *Edit table…*:

#defs(
  [*Cells*], [Double-click and type. An emptied cell becomes NULL (not measured) rather than an empty text, which would turn a numeric column into text.],
  [*Rows*], [*Add row* at the end, *Insert row above* the selected one, *Delete rows* with a selected cell.],
  [*Columns*], [Type a name and pick a type, then *Add column* (at the end) or *Insert before selected*; *Rename selected*; *Delete selected*.],
  [*Managed columns*], [`Hide` marks rows every chart skips; `Selected` marks rows for your own queries and operations (`WHERE "Selected" = 1`, or *Group by* `Selected` in New plot), with no effect on drawing; `ClusterId` is written by Clustering. Mark rows by a comparison on the selected column, or create, reset or invert the columns, from the editor's last button.],
)

Every change is written as it is made, yet *Cancel* still puts the table back: the editor takes a snapshot before the first change. *OK* keeps the changes (and the snapshot stays in the undo history), *Cancel* restores the snapshot after asking.

#note[
  Inserting a row *above* another renumbers the rows below, and inserting a column *before* another rebuilds the table. Neither is offered when the table's row id is one of its own columns (an `INTEGER PRIMARY KEY`); add at the end instead.
]

== The Query Builder <query-builder>

The *Query Builder* (in the *Data* section of Data operations, or from the table list's menu) writes, tests and saves SQL queries. A saved query appears in the table list and can feed any chart, exactly like a table — but its rows are computed each time it is read, so it never goes out of date and takes no space.

The window has:

- on the left, the *saved queries* (pick one to edit, *Add*, *Delete*); *Build*, the Tables operations of JMP and the like, each a form that writes its SQL; the project's *tables* with their *fields*, and three snippets;
- on the right, the *SQL* editor over the *result preview*.

*Build* — choose the operation, fill its form, press *Write SQL*: the statement goes into the editor and runs, so the preview shows the result at once. It is ordinary SQL, to read, change, save as a query or as a table:

#defs(
  [*Join*], [Rows of two tables matched on one or two pairs of columns. *Keep*: inner (matched rows only), left, right or full outer (every row, empty where there is no match). A column both tables have gets the suffix `_2`; *Add a Match column* says where each row came from (Both, Left only, Right only).],
  [*Union*], [Two or more tables one under the other (concatenate), columns matched by name — empty where a table lacks one — with a *Source* column naming each row's table. UNION ALL keeps repeated rows; *Drop repeated rows* writes UNION.],
  [*Stack*], [Wide to long: each chosen column becomes a row of two, *Label* (the column's name) and *Data* (its value); the other columns are repeated. Optionally without the empty values.],
  [*Split*], [Long to wide, the opposite of Stack: the values of *Split by* become columns, filled from *Values*. With *Group by* columns, one row per combination of them; with none, the n-th value of every label goes in row n. *Combine* says how two values for the same row and label are combined.],
  [*Transpose*], [Columns become rows and rows columns; the new columns are named by a label column, or Row 1, Row 2...],
  [*Recode*], [A column's values replaced by new ones, typed against each distinct value; the ones left empty stay as they are. Into a new column, or the column itself.],
  [*Bin*], [A numeric column cut into intervals: equal width, equal counts (quantiles) or your edges, labelled `[a, b)`, 1, 2, 3... or by midpoint. A new column.],
  [*Group*], [One row per combination of the grouping columns, with the row count and a statistic (sum, average, minimum, maximum, count, distinct count) of the chosen columns.],
)

The snippets act on the selected table and field:

#defs(
  [*Select*], [`SELECT` the chosen field from the table.],
  [*Order*], [`ORDER BY` the field, with an explicit direction.],
  [*Filter*], [`WHERE` with the comparison written out.],
)

*Run* shows the first 500 rows of the result. *OK* saves the statement as a query — read again each time, so it follows its tables — and closes the window, or stays open showing why it could not. *Save as table* writes what the statement returns as a new table instead: fixed rows, undoable, for a result that should not change when its sources do.

#note[
  By default ChartLibre refuses any SQL that would change the database — `UPDATE`, `DELETE`, `DROP` and the like — when a query is saved or typed. The data is changed only through the import window, the table editor and the operations, which all keep the undo history. The guard can be switched off in Settings (see @settings-chapter).
]

=== Useful SQL

#code[```
-- Filter and convert units
SELECT time AS x, temperature - 273.15 AS y FROM log WHERE sensor = 'A'

-- Monthly means from a dated table
SELECT strftime('%Y-%m', date) AS x, AVG(value) AS y FROM readings GROUP BY x

-- Recode a number into categories for a Box Plot
SELECT value, CASE WHEN dose < 10 THEN 'low' ELSE 'high' END AS "group" FROM trial
```]

Names that are SQL keywords, such as `group`, must be quoted with double quotes.
