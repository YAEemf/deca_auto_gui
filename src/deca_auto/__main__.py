import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

from deca_auto.config import UserConfig, load_config
from deca_auto.utils import (
    generate_frequency_grid, get_backend, get_dtype, logger, set_log_level, transfer_to_host,
)


def find_available_port(start_port, max_attempts=10):
    """ローカルで接続を受け付けていないポートを探す。"""
    for port in range(start_port, min(start_port + max_attempts, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"ポート {start_port} 以降に空きがありません")


def _is_listening(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _open_browser_when_ready(process, url, port, timeout=60.0):
    """サーバが待受を始めたらブラウザを開く。起動に失敗したら何もしない。"""
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        if _is_listening(port):
            webbrowser.open(url)
            return
        time.sleep(0.2)
    if process.poll() is None:
        logger.warning("ブラウザを自動で開けませんでした。%s を開いてください", url)


def launch_streamlit(config, args):
    """StreamlitのGUIを起動する。

    非ヘッドレス起動では、Streamlitが初回にメールアドレス入力（宣伝メール登録）を
    端末で求め、AIエージェント用スキルの宣伝も表示する。ヘッドレスで起動して
    これらを抑止し、ブラウザはこちらで開く。
    """
    interface_path = Path(__file__).with_name("interface.py")
    port = find_available_port(config.server_port, max_attempts=11)
    url = f"http://localhost:{port}"
    command = [
        sys.executable, "-m", "streamlit", "run", str(interface_path),
        "--server.port", str(port), "--server.address", "localhost",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark" if config.dark_theme else "light",
    ]
    env = os.environ.copy()
    if args.config:
        env["DECA_CONFIG_FILES"] = ",".join(args.config)
    if args.no_search:
        env["DECA_NO_SEARCH"] = "1"
    if args.force_numpy:
        env["DECA_FORCE_NUMPY"] = "1"
    if args.cuda is not None:
        env["DECA_CUDA_DEVICE"] = str(args.cuda)
    if args.output:
        env["DECA_EXCEL_PATH"] = args.output
    env["DECA_LOG_LEVEL"] = str(logger.level)

    logger.info("起動しています... (%s)", url)
    process = subprocess.Popen(command, env=env)
    try:
        threading.Thread(
            target=_open_browser_when_ready, args=(process, url, port), daemon=True,
        ).start()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def calculate_zc(config: UserConfig, *, allow_gpu=False):
    """|Z_c|計算と表示に必要なホスト配列を返す。"""
    from deca_auto.capacitor import calculate_all_capacitor_impedances

    xp = get_backend(config.force_numpy, config.cuda)[0] if allow_gpu else np
    grid = generate_frequency_grid(
        config.f_start, config.f_stop, config.num_points_per_decade, xp,
        dtype=get_dtype(config.dtype_r),
    )
    impedances = calculate_all_capacitor_impedances(
        config, grid, xp, None, get_dtype(config.dtype_c),
    )
    return transfer_to_host(grid), {name: transfer_to_host(z) for name, z in impedances.items()}


def main():
    parser = argparse.ArgumentParser(
        description="PDN Impedance Optimization Tool - Decoupling Capacitor Auto-selection",
    )
    parser.add_argument("--gui", action="store_true", help="GUIを起動 (use_gui=Falseでも有効)")
    parser.add_argument("--no-gui", action="store_true", help="GUIを無効化 (use_gui=Trueでも無効)")
    parser.add_argument("--config", nargs="+", help="設定ファイル（複数指定可能）")
    parser.add_argument("--no-search", action="store_true", help="探索を行わずZ_cのみ計算")
    parser.add_argument("--force-numpy", action="store_true", help="NumPyを強制使用")
    parser.add_argument("--cuda", type=int, help="使用するGPU番号")
    parser.add_argument("--verbose", action="store_true", help="詳細なログ出力")
    parser.add_argument("--output", help="Excel出力先ディレクトリ")
    args = parser.parse_args()
    set_log_level(logging.DEBUG if args.verbose else logging.INFO)

    try:
        configs = [load_config(path, strict=True) for path in args.config] if args.config else [UserConfig()]
        if not args.no_gui and (args.gui or configs[0].use_gui):
            launch_streamlit(configs[0], args)
            return

        from deca_auto.main import run_optimization

        logger.info("CLIモードで実行します")
        for i, config in enumerate(configs, 1):
            if len(configs) > 1:
                logger.info("設定 %s/%s を処理中...", i, len(configs))
            if args.force_numpy:
                config.force_numpy = True
            if args.cuda is not None:
                config.cuda = args.cuda
            if args.output:
                config.excel_path = args.output
            if args.no_search:
                logger.info("--no-searchが指定されたため、Z_c計算のみ実行します")
                calculate_zc(config)
            else:
                run_optimization(config, gui_callback=None)
    except KeyboardInterrupt:
        logger.info("処理を中断しました")
    except Exception:
        logger.exception("実行エラー")
        sys.exit(1)


if __name__ == "__main__":
    main()
