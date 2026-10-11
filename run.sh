#!/usr/bin/env bash
# Deca-Auto launcher for Linux
# Starts deca_auto from the virtual environment created by install.sh.
# Any arguments are passed through, e.g. bash run.sh --no-gui --config config.toml
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "$DIR/.venv/bin/deca-auto" ]; then
    echo "[ERROR] Deca-Auto is not installed. Run install.sh first." >&2
    exit 1
fi

exec "$DIR/.venv/bin/deca-auto" "$@"
