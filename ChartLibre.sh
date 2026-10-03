#!/bin/sh
# ChartLibre launcher for Linux (dev/tools/make_launcher.py). It works from
# wherever the ChartLibre folder is: the first launch installs the libraries
# into .venv inside that folder, every later one just starts the application.
# install.py adds it to the applications menu.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python3 ]; then
  python3 install.py || exit 1
fi
exec .venv/bin/python3 main.py "$@"
