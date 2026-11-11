#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$out" "$url"
  else
    echo "Missing required command: wget or curl" >&2
    exit 1
  fi
}

require_cmd unzip
require_cmd uvx

# -------- ORL (ATT Faces) --------
download_file "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip" "att_faces.zip"

mkdir -p data/ORL
unzip att_faces.zip -d data/ORL

# Some zips already extract directly into data/ORL.
if [ -d data/ORL/att_faces ]; then
  if compgen -G "data/ORL/att_faces/*" >/dev/null; then
    mv data/ORL/att_faces/* data/ORL/
  fi
  rm -rf data/ORL/att_faces
fi
rm -f att_faces.zip

# -------- Yale Face Database --------
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Kaggle credentials not found at ~/.kaggle/kaggle.json." >&2
    echo "Download the file from your Kaggle account page and run: chmod 600 ~/.kaggle/kaggle.json" >&2
    exit 1
  fi
fi

uvx kaggle datasets download olgabelitskaya/yale-face-database

mkdir -p data/Yale
unzip yale-face-database.zip -d data/Yale

# Move contents up one level if needed.
if [ -d data/Yale/data ]; then
  if compgen -G "data/Yale/data/*" >/dev/null; then
    mv data/Yale/data/* data/Yale/
  fi
  rm -rf data/Yale/data
fi
rm -f yale-face-database.zip
