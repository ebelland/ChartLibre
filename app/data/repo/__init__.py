"""The repository's helpers: what SqliteRepo (app.data.sqlite_repo) is built on.

``_common`` holds the errors, reports, SQL guards and decorators the
repository shares with the rest of the application; ``function_sql`` the SQL
functions it registers on each connection. The class itself, with all its
methods, is ``SqliteRepo`` in app/data/sqlite_repo.py.
"""
