#!/usr/bin/env bash
# Deca-Auto installer for Linux
# Creates a virtual environment in .venv and installs deca_auto into it.
set -e
cd "$(dirname "$0")"

PY=""
for cand in python3 python python3.14 python3.13 python3.12; do
    if command -v "$cand" >/dev/null 2>&1 \
        && "$cand" -c "import sys; sys.exit(sys.version_info < (3, 12))" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "[ERROR] Python 3.12 or later was not found. Install it and run this script again." >&2
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] Creating virtual environment in .venv ..."
    if ! "$PY" -m venv .venv; then
        echo "[ERROR] Failed to create the virtual environment." >&2
        echo "        On Debian/Ubuntu, install the venv module first: sudo apt install python3-venv" >&2
        exit 1
    fi
else
    echo "[1/3] Using existing virtual environment in .venv"
fi

echo "[2/3] Upgrading pip ..."
.venv/bin/python -m pip install --upgrade pip

echo "[3/3] Installing deca_auto and its dependencies. This may take several minutes ..."
.venv/bin/python -m pip install -e .

echo
echo "Installation completed."
echo "To start Deca-Auto, run: bash run.sh"
