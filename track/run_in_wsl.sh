#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

find_venv_activate() {
	if [ -n "${WSL_VENV_PATH:-}" ] && [ -f "${WSL_VENV_PATH}/bin/activate" ]; then
		echo "${WSL_VENV_PATH}/bin/activate"
		return 0
	fi

	for candidate in "$HOME/.venv_linux" "$HOME/.venv" "$HOME/venv"; do
		if [ -f "$candidate/bin/activate" ]; then
			echo "$candidate/bin/activate"
			return 0
		fi
	done

	return 1
}

PYTHON_BIN="${WSL_PYTHON_BIN:-python3}"
if ACTIVATE_PATH="$(find_venv_activate)"; then
	source "$ACTIVATE_PATH"
	PYTHON_BIN="python3"
	VENV_DIR="$(dirname "$(dirname "$ACTIVATE_PATH")")"
	echo "[run_in_wsl] using virtualenv: $VENV_DIR"
else
	echo "[run_in_wsl] warning: no Linux virtualenv found."
	echo "[run_in_wsl] set WSL_VENV_PATH to your venv path if needed."
	echo "[run_in_wsl] example: export WSL_VENV_PATH=\$HOME/.venv_linux"
	if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
		echo "[run_in_wsl] error: python interpreter '$PYTHON_BIN' not found."
		echo "[run_in_wsl] set WSL_PYTHON_BIN to a valid interpreter, e.g. python3.10"
		exit 2
	fi
fi

is_help_mode() {
	for arg in "$@"; do
		if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
			return 0
		fi
	done
	return 1
}

check_python_modules() {
	"$PYTHON_BIN" - "$@" <<'PY'
import importlib
import sys

missing = []
for mod in sys.argv[1:]:
	try:
		importlib.import_module(mod)
	except Exception:
		missing.append(mod)

if missing:
	print("[run_in_wsl] missing modules: " + ", ".join(missing))
	raise SystemExit(1)
PY
}

ensure_requirements() {
	if check_python_modules "$@"; then
		return 0
	fi
	echo "[run_in_wsl] install dependencies with:"
	echo "[run_in_wsl]   $PYTHON_BIN -m pip install -r requirements.txt"
	exit 3
}

TASK="${1:-train}"
shift || true

case "$TASK" in
	train)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor jittor_utils numpy PIL matplotlib
		fi
		"$PYTHON_BIN" train.py "$@"
		;;
	evaluate|eval)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor jittor_utils numpy PIL
		fi
		"$PYTHON_BIN" evaluate.py "$@"
		;;
	tune)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor jittor_utils numpy PIL matplotlib
		fi
		"$PYTHON_BIN" tune.py "$@"
		;;
	prepare)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor_utils
		fi
		"$PYTHON_BIN" prepare_voc.py "$@"
		;;
	check|check_gpu)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor
		fi
		"$PYTHON_BIN" check_gpu.py "$@"
		;;
	debug)
		"$PYTHON_BIN" debug_prediction.py "$@"
		;;
	compare)
		if ! is_help_mode "$@"; then
			ensure_requirements jittor jittor_utils numpy PIL matplotlib
		fi
		"$PYTHON_BIN" compare.py "$@"
		;;
	*)
		echo "[run_in_wsl] Unknown task: $TASK"
		echo "[run_in_wsl] Supported tasks: train | evaluate | tune | prepare | check | compare"
		exit 2
		;;
esac
