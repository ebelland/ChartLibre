"""The repository's parts: SqliteRepo (app.data.sqlite_repo) and the two mixins it inherits.

- ``tables``: TablesMixin - the user tables, their flags, and edits made by hand;
- ``project``: ProjectMixin - table tools, saved queries, maintenance, the
  operations record and the project info;
- ``_common``: the errors, reports, SQL guards and decorators shared with the
  rest of the application; ``function_sql``: the SQL functions registered on
  each connection.

Each mixin keeps no state of its own (``__slots__ = ()``); the fields, the
connection, the caches and the transactions stay on SqliteRepo.
"""
