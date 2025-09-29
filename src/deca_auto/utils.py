"""
共通ユーティリティモジュール
ログ、タイマ、補間、decimate、型変換ヘルパ、CPU <-> GPU転送、その他共通機能
"""

import time
import traceback
import logging
import sys
from typing import Optional, Union, Any, Tuple
from contextlib import contextmanager
from functools import wraps
import numpy as np

# CuPyのインポート（利用可能な場合）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("CuPyが利用できません。NumPyモードで動作します。")


# ロガー設定
def setup_logger(name: str = "deca_auto", level: int = logging.INFO) -> logging.Logger:
    """
    ロガーのセットアップ
    
    Args:
        name: ロガー名
        level: ログレベル
    
    Returns:
        logging.Logger: 設定されたロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ハンドラがない場合のみ追加
    if not logger.handlers:
        # コンソールハンドラ（stdout）
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.NOTSET)

        # フォーマッタ
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)
    else:
        for handler in logger.handlers:
            handler.setLevel(logging.NOTSET)

    logger.propagate = False

    return logger


# グローバルロガー
logger = setup_logger()


# バックエンド決定
def get_backend(force_numpy: bool = False, cuda_device: int = 0) -> Tuple[Any, str, bool]:
    """
    計算バックエンドを決定（NumPyまたはCuPy）
    
    Args:
        force_numpy: NumPyを強制使用
        cuda_device: 使用するGPU番号
    
    Returns:
        (xp, backend_name, is_gpu): バックエンドモジュール、名前、GPU使用フラグ
    """
    if force_numpy or not CUPY_AVAILABLE:
        logger.info("NumPyバックエンドを使用します")
        return np, "numpy", False
    
    try:
        # CUDAデバイスの確認
        cp.cuda.Device(cuda_device).use()
        
        # メモリ情報の取得テスト
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # デバイス情報
        device = cp.cuda.Device(cuda_device)
        props = device.attributes
        total_memory = device.mem_info[1] / 1024**3  # GB
        free_memory = device.mem_info[0] / 1024**3  # GB
        
        logger.info(f"CuPyバックエンドを使用します (GPU {cuda_device})")
        logger.info(f"GPU: {props.get('DeviceName', 'Unknown')} - Total: {total_memory:.2f}GB, Free: {free_memory:.2f}GB")
        
        return cp, "cupy", True
        
    except Exception as e:
        logger.warning(f"GPU初期化エラー: {e}")
        logger.info("NumPyバックエンドにフォールバックします")
        return np, "numpy", False


# GPU情報取得
def get_gpu_info(cuda_device: int = 0) -> Optional[dict]:
    """
    GPU情報を取得
    
    Args:
        cuda_device: GPU番号
    
    Returns:
        dict: GPU情報（利用不可の場合はNone）
    """
    if not CUPY_AVAILABLE:
        return None
    
    try:
        device = cp.cuda.Device(cuda_device)
        mem_info = device.mem_info
        props = device.attributes
        
        return {
            "device_id": cuda_device,
            "name": props.get("DeviceName", "Unknown"),
            "compute_capability": f"{props.get('ComputeCapabilityMajor', 0)}.{props.get('ComputeCapabilityMinor', 0)}",
            "total_memory_gb": mem_info[1] / 1024**3,
            "free_memory_gb": mem_info[0] / 1024**3,
            "used_memory_gb": (mem_info[1] - mem_info[0]) / 1024**3,
        }
    except Exception as e:
        logger.error(f"GPU情報取得エラー: {e}")
        traceback.print_exc()
        return None


# VRAM管理
def get_vram_budget(max_vram_ratio: float = 0.5, cuda_device: int = 0) -> Tuple[int, int]:
    """
    利用可能なVRAMバジェットを計算
    
    Args:
        max_vram_ratio: 最大使用率
        cuda_device: GPU番号
    
    Returns:
        (available_bytes, chunk_size_limit): 利用可能バイト数、チャンクサイズ上限
    """
    if not CUPY_AVAILABLE:
        # CPU使用時はメモリの一部を仮想的なバジェットとする
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            budget = int(available_memory * max_vram_ratio)
            chunk_size = min(budget // 10, 100 * 1024 * 1024)  # 最大100MB
        except ImportError:
            # psutilが無い場合はデフォルト値
            budget = 1024 * 1024 * 1024  # 1GB
            chunk_size = 100 * 1024 * 1024  # 100MB
        return budget, chunk_size
    
    try:
        device = cp.cuda.Device(cuda_device)
        free_memory = device.mem_info[0]
        
        # 利用可能メモリ
        available_bytes = int(free_memory * max_vram_ratio)
        
        # チャンクサイズ（利用可能メモリの1/10、最大1GB）
        chunk_size_limit = min(available_bytes // 10, 1024 * 1024 * 1024)
        
        return available_bytes, chunk_size_limit
        
    except Exception as e:
        logger.error(f"VRAMバジェット計算エラー: {e}")
        traceback.print_exc()
        return 100 * 1024 * 1024, 10 * 1024 * 1024  # フォールバック: 100MB, 10MB


# 配列変換ヘルパー
def ensure_numpy(array: Union[np.ndarray, Any]) -> np.ndarray:
    """
    配列をNumPy配列に変換（CuPy配列の場合はCPUに転送）
    
    Args:
        array: 入力配列
    
    Returns:
        np.ndarray: NumPy配列
    """
    if CUPY_AVAILABLE and hasattr(array, "__cuda_array_interface__"):
        # CuPy配列の場合
        return cp.asnumpy(array)
    return np.asarray(array)


def ensure_cupy(array: Union[np.ndarray, Any], dtype: Optional[np.dtype] = None) -> Any:
    """
    配列をCuPy配列に変換（利用可能な場合、そうでなければNumPy配列のまま）
    
    Args:
        array: 入力配列
        dtype: データ型
    
    Returns:
        CuPy配列またはNumPy配列
    """
    if not CUPY_AVAILABLE:
        return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)
    
    try:
        if hasattr(array, "__cuda_array_interface__"):
            # すでにCuPy配列
            return array if dtype is None else cp.asarray(array, dtype=dtype)
        else:
            # NumPy配列からCuPy配列へ
            return cp.asarray(array, dtype=dtype) if dtype else cp.asarray(array)
    except Exception as e:
        logger.warning(f"CuPy変換エラー: {e}")
        return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)


def transfer_to_device(data: Any, xp: Any) -> Any:
    """
    データを適切なデバイスに転送
    
    Args:
        data: 転送するデータ
        xp: バックエンドモジュール（np or cp）
    
    Returns:
        転送後のデータ
    """
    if xp is np:
        return ensure_numpy(data)
    else:
        return ensure_cupy(data)


# データ型ヘルパー
def get_dtype(dtype_str: str) -> np.dtype:
    """
    文字列からNumPyデータ型を取得
    
    Args:
        dtype_str: データ型文字列
    
    Returns:
        np.dtype: NumPyデータ型
    """
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    return dtype_map.get(dtype_str, np.float32)


# タイマークラス
class Timer:
    """処理時間計測用タイマー"""
    
    def __init__(self, name: str = "Timer", verbose: bool = True):
        """
        Args:
            name: タイマー名
            verbose: ログ出力フラグ
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            logger.info(f"{self.name}: {self.elapsed:.3f}秒")


# デコレータ版タイマー
def timed(func):
    """関数実行時間を計測するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"Function {func.__name__}"):
            return func(*args, **kwargs)
    return wrapper


# 対数補間
def log_interpolate(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, backend: Any = np) -> np.ndarray:
    """
    対数スケール補間：常にNumPyで安全に実行し、必要ならbackendに戻す
    """
    # まずCPU側に統一
    x_np  = ensure_numpy(x)
    xp_np = ensure_numpy(xp)
    fp_np = ensure_numpy(fp)

    # xpは単調増加・1次元に正規化
    xp_np = np.asarray(xp_np).ravel()
    fp_np = np.asarray(fp_np).ravel()
    x_np  = np.asarray(x_np).ravel()

    # xpの重複除去（np.interp要件対応）
    sort_idx = np.argsort(xp_np)
    xp_np = xp_np[sort_idx]
    fp_np = fp_np[sort_idx]
    uniq_mask = np.concatenate([[True], xp_np[1:] > xp_np[:-1]])
    xp_np = xp_np[uniq_mask]
    fp_np = fp_np[uniq_mask]

    # ゼロ・符号対策（対数補間は|fp|を使用し符号は別に保持）
    eps = 1e-300
    mag = np.maximum(np.abs(fp_np), eps)
    sign_src = np.sign(fp_np)

    log_x  = np.log10(np.maximum(x_np, eps))
    log_xp = np.log10(np.maximum(xp_np, eps))
    log_mag= np.log10(mag)

    # 安全にNumPy補間
    log_interp = np.interp(log_x, log_xp, log_mag)

    # 元の符号は周波数線形軸で補間（エッジで±1が暴れないよう外挿は端値保持）
    sign_lin = np.interp(x_np, xp_np, sign_src, left=sign_src[0], right=sign_src[-1])
    sign_lin = np.where(sign_lin >= 0, 1.0, -1.0)

    result = sign_lin * (10.0 ** log_interp)

    # backendに戻す
    if backend is not np:
        return transfer_to_device(result, backend)
    return result


# データのdecimate（間引き）
def decimate(data: np.ndarray, factor: int, axis: int = 0) -> np.ndarray:
    """
    データを間引く
    
    Args:
        data: 入力データ
        factor: 間引き係数
        axis: 間引く軸
    
    Returns:
        間引かれたデータ
    """
    if factor <= 1:
        return data
    
    # スライスで間引き
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(None, None, factor)
    return data[tuple(slices)]


# 周波数グリッド生成
def generate_frequency_grid(f_start: float, f_stop: float,
                          num_points_per_decade: int,
                          xp: Any = np,
                          dtype: Optional[Any] = None) -> np.ndarray:
    """
    対数スケールの周波数グリッドを生成
    
    Args:
        f_start: 開始周波数 [Hz]
        f_stop: 終了周波数 [Hz]
        num_points_per_decade: 10倍ごとの点数
        xp: バックエンドモジュール
    
    Returns:
        周波数グリッド
    """
    num_decades = np.log10(f_stop / f_start)
    num_points = max(2, int(np.ceil(num_decades * num_points_per_decade)))
    grid = xp.logspace(np.log10(f_start), np.log10(f_stop), num_points)
    if dtype is not None:
        grid = grid.astype(dtype, copy=False)
    return grid


# 評価帯域のマスク生成
def create_evaluation_mask(f_grid: np.ndarray, f_L: float, f_H: float,
                          xp: Any = np) -> np.ndarray:
    """
    評価帯域のブールマスクを生成
    
    Args:
        f_grid: 周波数グリッド
        f_L: 下限周波数
        f_H: 上限周波数
        xp: バックエンドモジュール
    
    Returns:
        ブールマスク
    """
    return (f_grid >= f_L) & (f_grid <= f_H)


# 目標マスクの生成
def create_target_mask(f_grid: np.ndarray, z_target: float,
                      z_custom_mask: Optional[list] = None,
                      xp: Any = np) -> np.ndarray:
    """
    目標インピーダンスマスクを生成
    
    Args:
        f_grid: 周波数グリッド
        z_target: フラット目標値
        z_custom_mask: カスタムマスク [(freq, impedance), ...]
        xp: バックエンドモジュール
    
    Returns:
        目標マスク
    """
    if z_custom_mask is None or len(z_custom_mask) == 0:
        # フラット目標
        return xp.full_like(f_grid, z_target)
    
    # カスタムマスクの対数補間
    f_points = xp.array([f for f, _ in z_custom_mask])
    z_points = xp.array([z for _, z in z_custom_mask])
    
    # 対数補間
    return log_interpolate(f_grid, f_points, z_points, backend=xp)


# バッチ処理ヘルパー
def batch_process(data: Any, batch_size: int, func: callable, 
                 *args, **kwargs) -> list:
    """
    データをバッチ処理
    
    Args:
        data: 入力データ
        batch_size: バッチサイズ
        func: 処理関数
        *args, **kwargs: 追加引数
    
    Returns:
        処理結果のリスト
    """
    results = []
    n = len(data)
    
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        result = func(batch, *args, **kwargs)
        results.append(result)
    
    return results


# メモリクリーンアップ
@contextmanager
def memory_cleanup(xp: Any = None):
    """
    メモリクリーンアップのコンテキストマネージャ
    
    Args:
        xp: バックエンドモジュール
    """
    try:
        yield
    finally:
        if xp is not None and xp is cp and CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            logger.debug("GPUメモリをクリーンアップしました")


# 安全な除算
def safe_divide(numerator: Any, denominator: Any, fill_value: float = 0.0,
                xp: Any = np) -> Any:
    """
    ゼロ除算を避ける安全な除算
    
    Args:
        numerator: 分子
        denominator: 分母
        fill_value: ゼロ除算時の値
        xp: バックエンドモジュール
    
    Returns:
        除算結果
    """
    # CuPyはerrstate未対応のため、分岐処理
    if xp is np:
        # NumPyの場合
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                mask = np.isfinite(result)
                result = np.where(mask, result, fill_value)
    else:
        # CuPyの場合（直接処理）
        result = numerator / denominator
        # NaNやInfをチェックして置換
        mask = xp.isfinite(result)
        result = xp.where(mask, result, fill_value)
    
    return result


# プログレスバー用ヘルパー
def get_progress_bar(iterable, desc: str = None, total: int = None,
                    disable: bool = False):
    """
    tqdmプログレスバーを取得
    
    Args:
        iterable: イテラブル
        desc: 説明文
        total: 総数
        disable: 無効化フラグ
    
    Returns:
        tqdmオブジェクト
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, disable=disable,
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    except ImportError:
        logger.warning("tqdmが利用できません")
        return iterable


# 結果の検証
def validate_result(result: Any, name: str = "result") -> bool:
    """
    計算結果の妥当性を検証
    
    Args:
        result: 検証する結果
        name: 結果の名前
    
    Returns:
        bool: 妥当性フラグ
    """
    xp = cp if CUPY_AVAILABLE and hasattr(result, "__cuda_array_interface__") else np
    
    if result is None:
        logger.error(f"{name}がNoneです")
        return False
    
    if hasattr(result, "shape"):
        if result.size == 0:
            logger.error(f"{name}が空です")
            return False
        
        if xp.any(xp.isnan(result)):
            logger.warning(f"{name}にNaNが含まれています")
            return False
        
        if xp.any(xp.isinf(result)):
            logger.warning(f"{name}に無限大が含まれています")
            return False
    
    return True
