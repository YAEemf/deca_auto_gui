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

from deca_auto.config import ConfigError, UserConfig, load_config
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
        "--logger.level", "warning",
        "--theme.base", "dark" if config.dark_theme else "light",
    ]
    env = os.environ.copy()
    # 起動時の "You can now view your Streamlit app" 表示を省く（未対応の版では無視される）
    env["STREAMLIT_LOGGER_HIDE_WELCOME_MESSAGE"] = "true"
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

    logger.info("GUI: %s (Ctrl+Cで終了)", url)
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


def calculate_zc(config: UserConfig, *, allow_gpu=False, report=False):
    """|Z_c|計算と表示に必要なホスト配列を返す。reportでコンデンサ一覧を出力する。"""
    from deca_auto.capacitor import calculate_all_capacitor_impedances

    xp = get_backend(config.force_numpy, config.cuda)[0] if allow_gpu else np
    grid = generate_frequency_grid(
        config.f_start, config.f_stop, config.num_points_per_decade, xp,
        dtype=get_dtype(config.dtype_r),
    )
    model_info = {}
    impedances = calculate_all_capacitor_impedances(
        config, grid, xp, None, get_dtype(config.dtype_c), model_info=model_info,
    )
    grid_host = transfer_to_host(grid)
    impedances_host = {name: transfer_to_host(z) for name, z in impedances.items()}
    if report:
        from deca_auto.main import _capacitor_rlc_params, log_capacitor_table

        params = _capacitor_rlc_params(config, list(impedances_host), impedances_host, grid_host)
        names = sorted(impedances_host, key=lambda n: params[n]['C'])
        log_capacitor_table(names, model_info, params)
    return grid_host, impedances_host


def _load_config_or_exit(path):
    """設定ファイルを読み込む。読めない場合は理由を1行で示して終了する。"""
    from tomlkit.exceptions import TOMLKitError

    if path is None:
        return UserConfig()
    try:
        return load_config(path, verbose=False, strict=True)
    except FileNotFoundError:
        logger.error("設定ファイルが見つかりません: %s", path)
    except (TOMLKitError, UnicodeDecodeError, OSError, ValueError) as exc:
        logger.error("設定ファイルを読み込めません: %s (%s)", path, exc)
    sys.exit(1)


def _load_config_quietly(path):
    """警告を出さずに読む（GUI起動判定用。警告はGUI側の読み込みで出る）"""
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return _load_config_or_exit(path)
    finally:
        logger.setLevel(previous)


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
        paths = args.config or [None]
        for path in paths:  # 実行前にすべての設定ファイルが読めることを確かめる
            if path is not None and not Path(path).is_file():
                logger.error("設定ファイルが見つかりません: %s", path)
                sys.exit(1)
        first = _load_config_quietly(paths[0])
        if not args.no_gui and (args.gui or first.use_gui):
            launch_streamlit(first, args)
            return

        from deca_auto.main import run_optimization

        for i, path in enumerate(paths, 1):
            prefix = f"[{i}/{len(paths)}] " if len(paths) > 1 else ""
            if i > 1:
                logger.info("")
            if path is not None:
                logger.info("%s設定: %s", prefix, path)
            config = _load_config_or_exit(path)
            if args.force_numpy:
                config.force_numpy = True
            if args.cuda is not None:
                config.cuda = args.cuda
            if args.output:
                config.excel_path = args.output
            if args.no_search:
                calculate_zc(config, report=True)
            else:
                run_optimization(config, gui_callback=None)
    except KeyboardInterrupt:
        logger.info("中断しました")
        sys.exit(130)
    except ConfigError:
        sys.exit(1)  # 理由は検証時に出力済み
    except Exception:
        logger.exception("実行エラー")
        sys.exit(1)


if __name__ == "__main__":
    main()
