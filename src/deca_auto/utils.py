import time
import traceback
import logging
import sys
from typing import Optional, Union, Any, Tuple, List
from contextlib import contextmanager
from functools import wraps
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("CuPyが利用できません。NumPyモードで動作します。")

try:
    from tomlkit.items import Item as _TomlItem
except ImportError:
    _TomlItem = None


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
    logger.disabled = False

    formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stream_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler = handler
            break

    if stream_handler is None:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(stream_handler)
    else:
        try:
            stream_handler.setStream(sys.stdout)
        except AttributeError:
            stream_handler.stream = sys.stdout

    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # 既存ハンドラがある場合もレベルとフォーマットを揃える
    for handler in logger.handlers:
        handler.setLevel(level)
        if isinstance(handler, logging.StreamHandler) and handler is not stream_handler:
            handler.setFormatter(formatter)

    logger.propagate = False

    # ルートロガーがWARNING以上の場合、情報出力が抑制されるため揃えておく
    root_logger = logging.getLogger()
    if root_logger.level > level:
        root_logger.setLevel(level)

    return logger


# グローバルロガー
logger = setup_logger()


def set_log_level(level: int):
    """ロガーと既存ハンドラのレベルを更新"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.disabled = False
    root_logger = logging.getLogger()
    if root_logger.level > level:
        root_logger.setLevel(level)


def unwrap_toml_value(value: Any) -> Any:
    """tomlkitのItemをPythonプリミティブへ展開"""
    if _TomlItem is not None and isinstance(value, _TomlItem):
        return unwrap_toml_value(value.unwrap())
    if isinstance(value, dict):
        return {k: unwrap_toml_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unwrap_toml_value(v) for v in value)
    return value


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """値をfloatに正規化"""
    v = unwrap_toml_value(value)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        try:
            return float(str(v))
        except (TypeError, ValueError):
            return default


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """値をintに正規化"""
    v = unwrap_toml_value(value)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(str(v)))
        except (TypeError, ValueError):
            return default


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
        logger.info("NumPyを使用します")
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
        
        logger.info(f"CuPyを使用します (GPU {cuda_device})")
        
        return cp, "cupy", True
        
    except Exception as e:
        logger.warning(f"GPU初期化エラー: {e}")
        logger.info("NumPyバックエンドにフォールバックします")
        return np, "numpy", False


# GPU情報取得
def _resolve_device_name(device: "cp.cuda.Device", index: int) -> str:  # type: ignore[name-defined]
    """GPU名を取得（バージョン差異を吸収）"""
    name_candidates: list[Any] = []

    try:
        attrs = device.attributes
        name_candidates.extend([
            attrs.get("Name"),
            attrs.get("DeviceName"),
        ])
    except Exception:
        pass

    try:
        props = cp.cuda.runtime.getDeviceProperties(index)
        name_candidates.append(props.get("name"))
    except Exception:
        pass

    for candidate in name_candidates:
        if candidate is None:
            continue
        if isinstance(candidate, bytes):
            try:
                return candidate.decode("utf-8")
            except Exception:
                continue
        return str(candidate)

    return f"CUDA Device {index}"


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
        name = _resolve_device_name(device, cuda_device)
        try:
            attrs = device.attributes
        except Exception:
            attrs = {}

        return {
            "device_id": cuda_device,
            "name": name,
            "compute_capability": (
                f"{attrs.get('ComputeCapabilityMajor', 0)}."
                f"{attrs.get('ComputeCapabilityMinor', 0)}"
            ),
            "total_memory_gb": mem_info[1] / 1024**3,
            "free_memory_gb": mem_info[0] / 1024**3,
            "used_memory_gb": (mem_info[1] - mem_info[0]) / 1024**3,
        }
    except Exception as e:
        logger.error(f"GPU情報取得エラー: {e}")
        traceback.print_exc()
        return None


def list_cuda_devices() -> list:
    """利用可能なCUDAデバイス一覧を取得"""
    if not CUPY_AVAILABLE:
        return []

    devices = []
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        logger.warning(f"CUDAデバイス数の取得に失敗しました: {exc}")
        return []

    for idx in range(count):
        try:
            device = cp.cuda.Device(idx)
            name = _resolve_device_name(device, idx)
            devices.append({'id': idx, 'name': name})
        except Exception as exc:
            logger.warning(f"CUDAデバイス情報の取得に失敗しました (index={idx}): {exc}")
            continue

    return devices


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
            reserved = int(available_memory * 0.1)
            budget = max(int(available_memory * max_vram_ratio) - reserved, int(available_memory * 0.1))
            chunk_size = min(max(budget // 12, 8 * 1024 * 1024), 100 * 1024 * 1024)
        except ImportError:
            # psutilが無い場合はデフォルト値
            budget = 1024 * 1024 * 1024  # 1GB
            chunk_size = 64 * 1024 * 1024  # 64MB
        return max(budget, 32 * 1024 * 1024), max(chunk_size, 4 * 1024 * 1024)

    try:
        device = cp.cuda.Device(cuda_device)
        free_memory, total_memory = device.mem_info
        used_memory = total_memory - free_memory

        ratio = min(max_vram_ratio, 0.98)
        safety_reserve = max(int(total_memory * 0.02), 256 * 1024 * 1024)
        permitted = int(total_memory * ratio) - used_memory - safety_reserve
        permitted = max(permitted, 0)

        free_cap = int(free_memory * 0.9)
        available_bytes = max(min(permitted, free_cap), 64 * 1024 * 1024)

        # チャンクサイズは保守的に設定し、余裕を確保
        chunk_size_limit = max(min(available_bytes // 12, 512 * 1024 * 1024), 8 * 1024 * 1024)

        # 万一available_bytesがfreeを上回りそうなときの保険
        available_bytes = min(available_bytes, free_cap)
        chunk_size_limit = min(chunk_size_limit, available_bytes)

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

    Notes:
        - すでにNumPy配列の場合はコピーを避ける
        - CuPy配列の場合のみCPU転送を実行
    """
    # すでにNumPy配列の場合はそのまま返す（コピー回避）
    if isinstance(array, np.ndarray):
        return array

    # CuPy配列の場合
    if CUPY_AVAILABLE and hasattr(array, "__cuda_array_interface__"):
        return cp.asnumpy(array)

    # その他の配列様オブジェクト
    return np.asarray(array)


def ensure_cupy(array: Union[np.ndarray, Any], dtype: Optional[np.dtype] = None) -> Any:
    """
    配列をCuPy配列に変換（利用可能な場合、そうでなければNumPy配列のまま）

    Args:
        array: 入力配列
        dtype: データ型

    Returns:
        CuPy配列またはNumPy配列

    Notes:
        - CuPy利用不可の場合はNumPy配列を返す
        - すでに目的のデバイス・型の場合はコピーを避ける
    """
    if not CUPY_AVAILABLE:
        return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)

    try:
        # すでにCuPy配列かつ型が一致する場合
        if hasattr(array, "__cuda_array_interface__"):
            if dtype is None or array.dtype == dtype:
                return array
            return cp.asarray(array, dtype=dtype)

        # NumPy配列からCuPy配列へ転送
        return cp.asarray(array, dtype=dtype) if dtype else cp.asarray(array)

    except Exception as e:
        logger.warning(f"CuPy変換エラー: {e}")
        return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)


def transfer_to_device(data: Any, xp: Any, dtype: Optional[np.dtype] = None) -> Any:
    """
    データを適切なデバイスに転送

    Args:
        data: 転送するデータ
        xp: バックエンドモジュール（np or cp）
        dtype: オプションのデータ型

    Returns:
        転送後のデータ

    Notes:
        - すでに目的のデバイスにある場合はコピーを避ける
        - 型変換が必要な場合のみ実行
    """
    if xp is np:
        result = ensure_numpy(data)
        if dtype is not None and result.dtype != dtype:
            return result.astype(dtype, copy=False)
        return result
    else:
        return ensure_cupy(data, dtype)


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
    対数スケール補間（GPU最適化版）

    Args:
        x: 補間する座標
        xp: データ点の座標
        fp: データ点の値
        backend: バックエンドモジュール（np or cp）

    Returns:
        補間された値

    Notes:
        - GPU使用時はCPU転送を回避し、GPU上で直接補間
        - CPU使用時は従来のNumPy補間を使用
    """
    # 共通の前処理ヘルパー
    def _prepare_data(x_arr, xp_arr, fp_arr, backend_mod):
        """データの準備と重複除去"""
        x_flat = backend_mod.asarray(x_arr).ravel()
        xp_flat = backend_mod.asarray(xp_arr).ravel()
        fp_flat = backend_mod.asarray(fp_arr).ravel()

        # ソートと重複除去
        sort_idx = backend_mod.argsort(xp_flat)
        xp_sorted = xp_flat[sort_idx]
        fp_sorted = fp_flat[sort_idx]

        if len(xp_sorted) > 1:
            diff = xp_sorted[1:] - xp_sorted[:-1]
            uniq_mask = backend_mod.concatenate([backend_mod.array([True]), diff > 0])
            xp_sorted = xp_sorted[uniq_mask]
            fp_sorted = fp_sorted[uniq_mask]

        return x_flat, xp_sorted, fp_sorted

    # 共通の補間ロジック
    def _interpolate_log(x_flat, xp_sorted, fp_sorted, backend_mod, eps=1e-300):
        """対数補間の実行"""
        # 絶対値と符号を分離
        mag = backend_mod.maximum(backend_mod.abs(fp_sorted), eps)
        sign_src = backend_mod.sign(fp_sorted)

        # 対数変換
        log_x = backend_mod.log10(backend_mod.maximum(x_flat, eps))
        log_xp = backend_mod.log10(backend_mod.maximum(xp_sorted, eps))
        log_mag = backend_mod.log10(mag)

        # 補間実行
        log_interp = backend_mod.interp(log_x, log_xp, log_mag)

        # 符号の線形補間
        sign_lin = backend_mod.interp(
            x_flat, xp_sorted, sign_src,
            left=float(sign_src[0]),
            right=float(sign_src[-1])
        )
        sign_lin = backend_mod.where(sign_lin >= 0, 1.0, -1.0)

        return sign_lin * (10.0 ** log_interp)

    # GPU版
    if CUPY_AVAILABLE and backend is cp:
        try:
            x_gpu, xp_gpu, fp_gpu = _prepare_data(x, xp, fp, backend)
            result = _interpolate_log(x_gpu, xp_gpu, fp_gpu, backend)
            return result
        except Exception as e:
            logger.warning(f"GPU補間でエラーが発生、CPUにフォールバック: {e}")
            # フォールバック

    # CPU版（またはフォールバック）
    x_cpu, xp_cpu, fp_cpu = _prepare_data(
        ensure_numpy(x),
        ensure_numpy(xp),
        ensure_numpy(fp),
        np
    )
    result = _interpolate_log(x_cpu, xp_cpu, fp_cpu, np)

    # 必要に応じてバックエンドに戻す
    if backend is not np and CUPY_AVAILABLE:
        return transfer_to_device(result, backend)
    return result


# データのdecimate（間引き）
def decimate(data: np.ndarray, factor: int, axis: int = 0) -> np.ndarray:
    """
    データを間引く

    Args:
        data: 入力データ
        factor: 間引き係数（2以上で有効、1以下は元データを返す）
        axis: 間引く軸（デフォルト: 0）

    Returns:
        間引かれたデータ

    Examples:
        >>> data = np.arange(100)
        >>> decimated = decimate(data, factor=10)
        >>> len(decimated)
        10
        >>> decimated[0], decimated[1]
        (0, 10)

        >>> data_2d = np.arange(20).reshape(10, 2)
        >>> decimated_2d = decimate(data_2d, factor=2, axis=0)
        >>> decimated_2d.shape
        (5, 2)
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


# 目標インピーダンス計算（自動モード）
def calculate_target_impedance_auto(
    v_supply: float,
    ripple_ratio: Optional[float],
    ripple_voltage: Optional[float],
    i_max: float,
    switching_activity: Optional[float],
    i_transient: Optional[float],
    design_margin: float
) -> float:
    """
    自動計算モードで目標インピーダンスを計算

    Args:
        v_supply: 電源電圧 [V]
        ripple_ratio: 許容リップル率 [%]
        ripple_voltage: 許容リップル電圧 [V]
        i_max: 最大消費電流 [A]
        switching_activity: 電流変動率 (0-1)
        i_transient: 過渡電流 [A]
        design_margin: デザインマージン [%]

    Returns:
        目標インピーダンス [Ω]

    Notes:
        理論式: Z_target = (ΔV_ripple × (1 - margin/100)) / I_transient
    """
    # 許容リップル電圧の決定
    if ripple_voltage is not None:
        delta_v = ripple_voltage
    elif ripple_ratio is not None:
        delta_v = v_supply * (ripple_ratio / 100.0)
    else:
        raise ValueError("ripple_ratio または ripple_voltage のいずれかを指定する必要があります")

    # 過渡電流の決定
    if i_transient is not None:
        i_trans = i_transient
    elif switching_activity is not None:
        i_trans = i_max * switching_activity
    else:
        # デフォルトとして i_max の 50% を使用
        i_trans = i_max * 0.5

    # デザインマージンの適用
    delta_v_design = delta_v * (1.0 - design_margin / 100.0)

    # 目標インピーダンス
    z_target = delta_v_design / i_trans

    logger.info(
        f"自動計算モード: V_supply={v_supply}V, ΔV_ripple={delta_v*1000:.2f}mV, "
        f"I_transient={i_trans}A, margin={design_margin}%, Z_target={z_target*1000:.3f}mΩ"
    )

    return z_target


# 目標マスクの生成（モード対応版）
def create_target_mask(
    f_grid: np.ndarray,
    z_target: float,
    z_custom_mask: Optional[list] = None,
    xp: Any = np,
    mode: str = "flat",
    config: Any = None
) -> np.ndarray:
    """
    目標インピーダンスマスクを生成（モード対応版）

    Args:
        f_grid: 周波数グリッド
        z_target: フラット目標値
        z_custom_mask: カスタムマスク [(freq, impedance), ...]
        xp: バックエンドモジュール
        mode: 目標インピーダンスモード ("flat", "auto", "custom")
        config: 設定オブジェクト（autoモードで必要）

    Returns:
        目標マスク

    Notes:
        - "flat": 単一値のフラット目標
        - "auto": 電源仕様から自動計算
        - "custom": カスタムマスク（対数補間）
    """
    if mode == "flat":
        # フラット目標
        return xp.full_like(f_grid, z_target)

    elif mode == "auto":
        # 自動計算モード
        if config is None:
            raise ValueError("autoモードではconfigが必要です")

        z_auto = calculate_target_impedance_auto(
            config.v_supply,
            config.ripple_ratio,
            config.ripple_voltage,
            config.i_max,
            config.switching_activity,
            config.i_transient,
            config.design_margin
        )
        return xp.full_like(f_grid, z_auto)

    elif mode == "custom":
        # カスタムマスク
        if z_custom_mask is None or len(z_custom_mask) == 0:
            logger.warning("カスタムマスクが指定されていないため、フラット目標を使用します")
            return xp.full_like(f_grid, z_target)

        # カスタムマスクの対数補間
        f_points = xp.array([f for f, _ in z_custom_mask])
        z_points = xp.array([z for _, z in z_custom_mask])

        return log_interpolate(f_grid, f_points, z_points, backend=xp)

    else:
        raise ValueError(f"未知の目標インピーダンスモード: {mode}")


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
                     disable: bool = False, dynamic: bool = True):
    """
    tqdmプログレスバーを取得（ターミナル幅に追従可）

    Args:
        iterable: イテラブル
        desc: 説明文
        total: 総数
        disable: 無効化フラグ
        dynamic: ターミナル幅に追従させる（dynamic_ncols）

    Returns:
        tqdmオブジェクト
    """
    try:
        from tqdm import tqdm
        # bar_format を指定しないか、最小限に留める
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            disable=disable,
            dynamic_ncols=dynamic,
        )
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


def create_decimated_indices(data_length: int, max_points: int) -> list:
    """
    データを間引くためのインデックスリストを作成

    Args:
        data_length: 元データの長さ
        max_points: 最大点数

    Returns:
        間引き後のインデックスリスト

    Examples:
        >>> indices = create_decimated_indices(1000, 100)
        >>> len(indices)
        100
        >>> indices[:3]
        [0, 10, 20]

    Notes:
        - data_length <= max_points の場合は全インデックスを返す
        - それ以外の場合は均等に間引いたインデックスを返す
    """
    if data_length <= max_points:
        return list(range(data_length))

    step = max(1, data_length // max_points)
    return list(range(0, data_length, step))


def parse_scientific_notation(value: Any) -> float:
    """
    科学的記数法を解析（10e3, 1.6e-19形式に対応）

    Args:
        value: 解析する値

    Returns:
        float: 解析された浮動小数点数

    Raises:
        ValueError: 無効な値または形式の場合

    Examples:
        >>> parse_scientific_notation("10e3")
        10000.0
        >>> parse_scientific_notation("1.5e-9")
        1.5e-09
        >>> parse_scientific_notation(100)
        100.0
    """
    if value is None:
        raise ValueError("値がNoneです")

    # すでに数値の場合
    if isinstance(value, (int, float)):
        return float(value)

    # 文字列に変換
    value_str = str(value).strip()
    if not value_str:
        raise ValueError("空の文字列です")

    try:
        # 通常の浮動小数点数として解析を試みる
        return float(value_str)
    except ValueError:
        # 10e3形式の場合の処理
        value_str = value_str.replace(" ", "")
        if "e" in value_str.lower():
            parts = value_str.lower().split("e")
            if len(parts) == 2:
                try:
                    base = float(parts[0]) if parts[0] else 1.0
                    exp = float(parts[1])
                    return base * (10 ** exp)
                except:
                    pass
        raise ValueError(f"無効な数値形式: {value_str}")


def get_custom_mask_freq_range(z_custom_mask: Optional[List[Tuple[float, float]]]) -> Tuple[Optional[float], Optional[float]]:
    """
    カスタムマスクから周波数範囲を取得

    Args:
        z_custom_mask: カスタムマスク [(freq, impedance), ...]

    Returns:
        (f_min, f_max): 周波数の最小値と最大値のタプル（マスクがNoneの場合は (None, None)）

    Examples:
        >>> mask = [(1e3, 10e-3), (1e6, 8e-3), (1e8, 0.45)]
        >>> get_custom_mask_freq_range(mask)
        (1000.0, 100000000.0)
        >>> get_custom_mask_freq_range(None)
        (None, None)
    """
    if not z_custom_mask or len(z_custom_mask) == 0:
        return None, None

    freqs = [f for f, _ in z_custom_mask if f is not None]
    if not freqs:
        return None, None

    return min(freqs), max(freqs)
