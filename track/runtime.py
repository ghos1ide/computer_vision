"""Runtime helpers for Jittor startup diagnostics."""

import os
import platform


def print_startup_hint(task_name: str) -> None:
    print(f"[env] starting {task_name}")
    if os.name == "nt":
        print("[env] on Windows, first-time Jittor JIT compilation can take several minutes")


def import_jittor_or_exit(task_name: str):
    if os.name == "nt" and os.environ.get("JT_FORCE_WINDOWS_IMPORT", "0") != "1":
        print("=" * 78)
        print(f"[env] {task_name} is blocked on native Windows to avoid Jittor compile stalls")
        print("[env] this repository is configured to run Jittor tasks in WSL/Linux")
        print("[env] use one of the following commands:")
        print("[env]   bash run_in_wsl.sh check")
        print("[env]   bash run_in_wsl.sh prepare --data-root ./data")
        print("[env]   bash run_in_wsl.sh train --data-root ./data --save-dir ./runs/unet_se")
        print("[env] if you still want to try native Windows, set:")
        print("[env]   JT_FORCE_WINDOWS_IMPORT=1")
        print("=" * 78)
        raise SystemExit(1)

    try:
        import jittor as jt
        from jittor import nn
    except ModuleNotFoundError as exc:
        print("=" * 78)
        print(f"[env] failed to initialize Jittor for {task_name}")
        print(f"[env] missing python module: {exc.name}")
        print("[env] install dependencies in your current environment:")
        print("[env]   python3 -m pip install -r requirements.txt")
        print("=" * 78)
        raise SystemExit(1)
    except Exception as exc:
        message = str(exc).splitlines()
        last_line = message[-1] if message else repr(exc)

        print("=" * 78)
        print(f"[env] failed to initialize Jittor for {task_name}")
        print(f"[env] platform: {platform.platform()}")
        if os.name == "nt":
            print("[env] this is usually a Windows toolchain compatibility issue, not a code deadlock")
            print("[env] recommended workaround:")
            print("[env]   1) switch to WSL/Linux")
            print("[env]   2) activate your Linux virtual environment")
            print("[env]   3) run scripts with run_in_wsl.sh")
            print("[env] examples:")
            print("[env]   bash run_in_wsl.sh check")
            print("[env]   bash run_in_wsl.sh prepare --data-root ./data")
            print("[env]   bash run_in_wsl.sh train --data-root ./data --save-dir ./runs/unet_se")
        else:
            print("[env] unexpected runtime error. verify dependencies and local environment.")
            print("[env] try: python3 -m pip install -r requirements.txt")
        print("[env] original error (last line):")
        print(f"[env]   {last_line}")
        print("=" * 78)
        raise SystemExit(1)

    return jt, nn
