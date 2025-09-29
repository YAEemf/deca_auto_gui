import sys
import os
import argparse
import subprocess
import socket
from pathlib import Path
from typing import List, Optional
import traceback

# 絶対パスでインポート
import numpy as np

from deca_auto.config import load_config, USER_CONFIG
from deca_auto.utils import logger, generate_frequency_grid, get_dtype
from deca_auto.main import run_optimization


def check_port(port: int) -> bool:
    """ポートが利用可能か確認"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0
    except:
        return True


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """利用可能なポートを探す"""
    for i in range(max_attempts):
        port = start_port + i
        if check_port(port):
            return port
    raise RuntimeError(f"利用可能なポートが見つかりません (試行: {start_port}-{start_port+max_attempts-1})")


def launch_streamlit(config_files: List[str] = None, no_search: bool = False):
    """StreamlitのGUIを起動"""
    try:
        # インターフェースモジュールのパスを取得
        interface_path = Path(__file__).parent / "interface.py"
        
        # ポートの確認と変更
        port = USER_CONFIG.server_port
        if not check_port(port):
            logger.info(f"ポート {port} は使用中です")
            port = find_available_port(port + 1)
            logger.info(f"ポート {port} を使用します")
        
        # Streamlitコマンド構築
        cmd = [
            sys.executable,
            "-m", "streamlit", "run",
            str(interface_path.resolve()),
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
        ]
        
        # ダークテーマ設定
        if USER_CONFIG.dark_theme:
            cmd.extend(["--theme.base", "dark"])
        
        # 環境変数に設定情報を渡す
        env = os.environ.copy()
        if config_files:
            env["DECA_CONFIG_FILES"] = ",".join(config_files)
        if no_search:
            env["DECA_NO_SEARCH"] = "1"
        
        # Streamlit起動
        logger.info(f"StreamlitのGUIを起動しています... (http://localhost:{port})")
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        logger.info("GUIを終了します")
    except Exception as e:
        logger.error(f"GUI起動エラー: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        description="PDN Impedance Optimization Tool - Decoupling Capacitor Auto-selection"
    )
    
    # オプション定義
    parser.add_argument(
        "--gui",
        action="store_true",
        help="GUIを起動 (use_gui=FalseでもGUIを有効化)"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="GUIを無効化 (use_gui=TrueでもGUIを無効化、探索速度優先)"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        help="設定ファイル（複数指定可能）"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="探索を行わずZ_cのみ計算"
    )
    parser.add_argument(
        "--force-numpy",
        action="store_true",
        help="CuPyが利用可能でもNumPyを強制使用"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="使用するGPU番号 (デフォルト: 0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログ出力"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Excel出力先ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # ロギングレベル設定
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # 設定ファイルの読み込み
    configs = []
    if args.config:
        for config_file in args.config:
            config_path = Path(config_file)
            if config_path.exists():
                config = load_config(config_path)
                configs.append(config)
                logger.info(f"設定ファイルを読み込みました: {config_path}")
            else:
                logger.error(f"設定ファイルが見つかりません: {config_path}")
                sys.exit(1)
    else:
        # デフォルト設定を使用
        configs.append(USER_CONFIG)
    
    # 設定の上書き
    for config in configs:
        if args.force_numpy:
            config.force_numpy = True
        if args.cuda is not None:
            config.cuda = args.cuda
        if args.output:
            config.excel_path = args.output
    
    # GUI使用判定
    use_gui = configs[0].use_gui  # 最初の設定を基準にする
    if args.gui:
        use_gui = True
    if args.no_gui:
        use_gui = False
    
    # 実行
    if use_gui:
        # GUI起動
        launch_streamlit(
            config_files=args.config,
            no_search=args.no_search
        )
    else:
        # CLIモード
        logger.info("CLIモードで実行します")
        try:
            for i, config in enumerate(configs):
                if len(configs) > 1:
                    logger.info(f"設定 {i+1}/{len(configs)} を処理中...")
                
                # 探索実行
                if not args.no_search:
                    run_optimization(config, gui_callback=None)
                else:
                    logger.info("--no-searchが指定されたため、Z_c計算のみ実行します")
                    # Z_c計算のみ実行する処理を追加
                    from deca_auto.capacitor import calculate_all_capacitor_impedances
                    f_grid = generate_frequency_grid(
                        config.f_start,
                        config.f_stop,
                        config.num_points_per_decade,
                        np,
                        dtype=get_dtype(config.dtype_r),
                    )
                    calculate_all_capacitor_impedances(
                        config,
                        f_grid,
                        np,
                        None,
                        get_dtype(config.dtype_c),
                    )
                
        except KeyboardInterrupt:
            logger.info("処理を中断しました")
            sys.exit(0)
        except Exception as e:
            logger.error(f"実行エラー: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
