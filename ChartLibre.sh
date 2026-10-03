#!/bin/sh
# ChartLibre launcher for Linux (dev/tools/make_launcher.py). It works from
# wherever the ChartLibre folder is. The first launch downloads Python into
# .python and the libraries into .venv, both inside that folder; every later
# launch just starts the application. install.py adds it to the menu.
cd "$(dirname "$0")" || exit 1

get_python() {  # download Python 3.13.16 into .python, once
  [ -x .python/bin/python3 ] && return 0
  case "$(uname -m)" in
    x86_64|amd64)  url="https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.13.16+20261003-x86_64-unknown-linux-gnu-install_only.tar.gz"; sum="0a0272910b10417c659a9312fb3f2d7a6d774da7bd510999be7a3ba83273dc1f" ;;
    aarch64|arm64) url="https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.13.16+20261003-aarch64-unknown-linux-gnu-install_only.tar.gz"; sum="6477121f22904963a066ac8ff89983baae781510815590e894daa3c11ce86652" ;;
    *) echo "ChartLibre has no Python for this processor ($(uname -m))."; return 1 ;;
  esac
  archive=.python-download.tar.gz
  echo "Downloading Python 3.13.16 ..."
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

if [ ! -x .venv/bin/python3 ]; then
  echo "First launch: ChartLibre downloads what it needs into this folder -"
  echo "Python and its scientific libraries: about 400 MB to download, 1.7 GB"
  echo "on disk. This needs an internet connection and takes a few minutes."
  get_python || exit 1
  .python/bin/python3 install.py || exit 1
fi
exec .venv/bin/python3 main.py "$@"
