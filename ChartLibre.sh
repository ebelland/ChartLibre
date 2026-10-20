#!/bin/sh
# ChartLibre launcher for Linux (dev/tools/make_launcher.py). It works from
# wherever the ChartLibre folder is. The first launch downloads Python and
# the libraries into .python inside that folder (a ready-made package
# already has them); every later launch just starts the application.
# install.py adds it to the applications menu.
cd "$(dirname "$0")" || exit 1

get_python() {  # download Python 3.15.0 into .python, once
  [ -x .python/bin/python3 ] && return 0
  case "$(uname -m)" in
    x86_64|amd64)  url="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.15.0+20261013-x86_64-unknown-linux-gnu-install_only.tar.gz"; sum="e02a368dd5dc76ffe0599e5109ad6c1ee84e597910758b0e006bab25372a8921" ;;
    aarch64|arm64) url="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.15.0+20261013-aarch64-unknown-linux-gnu-install_only.tar.gz"; sum="26a54e8fbde81bcc39ba35ad855a6a573295c023793c20fff1670a560649bcd7" ;;
    *) echo "ChartLibre has no Python for this processor ($(uname -m))."; return 1 ;;
  esac
  archive=.python-download.tar.gz
  echo "Downloading Python 3.15.0 ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$archive" "$url" || return 1
  else
    wget -O "$archive" "$url" || return 1
  fi
  echo "$sum  $archive" | sha256sum -c - || { rm -f "$archive"; echo "The download is damaged."; return 1; }
  rm -rf .python && mkdir -p .python || return 1
  tar -xzf "$archive" -C .python --strip-components 1 || { rm -rf .python; return 1; }
  rm -f "$archive"
}

if [ -f .python/CHARTLIBRE_READY ]; then
  # A ready-made package: add the menu entry the first time it runs here.
  [ -f "$HOME/.local/share/applications/chartlibre.desktop" ] || .python/bin/python3 install.py --menu-only
  exec .python/bin/python3 main.py "$@"
fi
if [ -x .venv/bin/python3 ]; then  # an install made before .python
  exec .venv/bin/python3 main.py "$@"
fi

echo "First launch: ChartLibre downloads what it needs into this folder -"
echo "Python and its scientific libraries: about 400 MB to download, 1.7 GB"
echo "on disk. This needs an internet connection and takes a few minutes."
get_python || exit 1
.python/bin/python3 install.py --runtime || exit 1
exec .python/bin/python3 main.py "$@"
