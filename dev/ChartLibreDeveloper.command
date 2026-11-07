#!/bin/bash
# Double-click in Finder (or run in a terminal): opens ChartLibre Developer with the project's environment.
cd "$(dirname "$0")/.." || exit 1
exec python.nosync/bin/python3 dev/ChartLibreDeveloper.py "$@"
