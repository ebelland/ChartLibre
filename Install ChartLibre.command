#!/bin/bash
# Double-click to install ChartLibre's libraries into .venv inside this folder
# (install.py does the work). On Linux, run:  python3 install.py
cd "$(dirname "$0")" || exit 1

PY=""
CANDIDATES=$(ls -d /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 2>/dev/null | sort -t. -k2,2nr)
for c in $CANDIDATES /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3)"; do
  if [ -x "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "ChartLibre needs Python 3.11 or newer."
  echo "Install it from https://www.python.org/downloads/ and open this file again."
  open "https://www.python.org/downloads/" 2>/dev/null
  read -r -p "Press Return to close. " _
  exit 1
fi

"$PY" install.py "$@"
status=$?
echo
read -r -p "Press Return to close. " _
exit $status
