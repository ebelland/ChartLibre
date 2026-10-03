#import "helpers.typ": *

= Data <data-chapter>

== The Tables page

The *Tables* page lists the project's tables and saved queries (marked with a *Q*), and shows a preview of the selected one below the list.

The list has four columns:

#defs(
  [*Table*], [The table's name.],
  [*Link*], [How the table is linked to its source; saved queries show a *Q* here.],
  [*Source*], [Where its data comes from: a file name for a file or web import, "connection → table" for a database import, the SQL itself for a saved query (the full text in the tooltip), nothing for a table with no link.],
  [*Notes*], [Free text: double-click to write what the table is, where it came from, what was done to it. Notes are kept in the project and appear in reports.],
)

Tables written by series operations start with `_`. *Show generated tables* hides or shows them, so the list can show only your own data.

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
  [*Hide rows*], [Hide rows equal to the selected cell, different from it, higher, lower, or where the column is empty — by marking them in the `Hide` column, which every chart skips. The rows stay in the table.],
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

*Web* has a menu of ready-made public datasets grouped by subject. Picking one fills in the address without downloading, so it can be checked first; *Fetch* downloads it. *Add source* saves an address of your own to the menu, and *Delete source* removes one you added.

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
  [*Managed columns*], [`Hide` marks rows every chart skips; `ClusterId` is written by Clustering. Create, reset or invert them here.],
)

Every change is written as it is made, yet *Cancel* still puts the table back: the editor takes a snapshot before the first change. *OK* keeps the changes (and the snapshot stays in the undo history), *Cancel* restores the snapshot after asking.

#note[
  Inserting a row *above* another renumbers the rows below, and inserting a column *before* another rebuilds the table. Neither is offered when the table's row id is one of its own columns (an `INTEGER PRIMARY KEY`); add at the end instead.
]

== The Query Builder <query-builder>

The *Query Builder* (in the *Data* section of Series Operations, or from the table list's menu) writes, tests and saves SQL queries. A saved query appears in the table list and can feed any chart, exactly like a table — but its rows are computed each time it is read, so it never goes out of date and takes no space.

The window has:

- on the left, the *saved queries* (add, delete, pick one to edit) and the project's *tables* with their *fields*;
- in the middle, the *SQL* editor and the query's *name*;
- *snippet* buttons that insert ready-made SQL for the selected table and field:

#defs(
  [*Select*], [`SELECT` the chosen fields from the table.],
  [*Join*], [Opens *Build a JOIN*: the main table and field, the table and field to join on, and the join type.],
  [*Union*], [Opens *Build a UNION*: two tables, stacked, using the fields they have in common.],
  [*Summary*], [Count, average, minimum and maximum, grouped by a field.],
  [*Order*], [`ORDER BY` the field, with an explicit direction.],
  [*Filter*], [`WHERE` with the comparison written out.],
)

Placeholders in a snippet are written in `<angle brackets>`, which is not valid SQL, so a snippet left unfinished fails instead of running on the wrong column. *Run* shows the first 500 rows of the result below the editor. *Save query* saves it under its name; *OK* saves and closes the window — and stays open if the query could not be saved, showing why.

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
