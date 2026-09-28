"""The calculations behind the series operations, without any Qt.

Each module takes plain arrays and parameters and returns plain results, so
the same code runs behind a dialog, in a test, in a script or on a worker
thread. A dialog's job is to collect the parameters and present the result;
the arithmetic lives here. See todo.txt R-01.
"""
