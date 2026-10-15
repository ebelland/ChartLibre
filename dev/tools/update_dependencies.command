#!/bin/sh
# macOS / Linux: upgrade ChartLibre's libraries (double-click in Finder, or run in a terminal).
# Any options go to update_dependencies.py: --dry-run, --write-pins, --test, --python PATH.
cd "$(dirname "$0")/../.." || exit 1
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .python/bin/python3 ]; then PY=.python/bin/python3
else PY=python3; fi
"$PY" dev/tools/update_dependencies.py "$@"
status=$?
printf '\nPress Return to close.'; read -r _
exit $status
