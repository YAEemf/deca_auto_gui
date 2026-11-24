import logging
import re
import traceback
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable
import numpy as np

from deca_auto.config import UserConfig
from deca_auto.utils import (
    logger, get_progress_bar, transfer_to_device, transfer_to_host,
    log_interpolate, validate_result, ensure_numpy, to_float
)

try:
    from PySpice.Spice.Netlist import Circuit
    PYSPICE_AVAILABLE = True
    # PySpice.Logging.setup_logging() は dictConfig で既存ロガー（deca_auto含む）を
    # 無効化し、ルートにINFOハンドラを足すため使わない
    logging.getLogger('PySpice').setLevel(logging.WARNING)
except ImportError:
    PYSPICE_AVAILABLE = False
    logger.warning("PySpiceが利用できません。RLCモデルのみ使用します")

try:
    import skrf as rf
    from skrf.vectorFitting import VectorFitting
    SKRF_AVAILABLE = True
    # scikit-rfは反復ごとにINFOログを出すため抑制する
    logging.getLogger('skrf').setLevel(logging.WARNING)
except ImportError:
    SKRF_AVAILABLE = False
    logger.warning("scikit-rfが利用できません。VectorFittingは使用できません")


AC_OVERSAMPLE = 8  # SPICE AC解析の点密度（周波数グリッド比）
VF_TOLERANCE = 0.01  # VectorFittingを採用する最大相対誤差（超える場合はAC解析結果をそのまま使う）


def parse_spice_model(model_path: Path) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    SPICEモデルファイルからサブサーキット名とピンを解析
    
    Args:
        model_path: SPICEモデルファイルパス
    
    Returns:
        (subckt_name, pins): サブサーキット名とピンリスト
    """
    try:
        with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # .SUBCKTを探す（大文字小文字不問）
        pattern = r'(?im)^\s*\.SUBCKT\s+([A-Za-z0-9_.-]+)\s+([^\r\n*;]+)'
        match = re.search(pattern, content)
        
        if match:
            subckt_name = match.group(1)
            pins_str = match.group(2)
            # ピンを解析（空白区切り）
            pins = pins_str.split()
            
            logger.debug(f"SPICEモデル解析: {subckt_name}, ピン: {pins}")
            
            # 2端子チェック
            if len(pins) != 2:
                logger.warning(f"2端子でないサブサーキット: {subckt_name} ({len(pins)}ピン)")
                return None, None
            
            return subckt_name, pins
        else:
            logger.warning(f"SUBCKTが見つかりません: {model_path}")
            return None, None
            
    except Exception as e:
        logger.error(f"SPICEモデル解析エラー: {e}")
        traceback.print_exc()
        return None, None


def simulate_ac_impedance(model_path: Path, f_grid: np.ndarray, 
                         dc_bias: float = 5.0) -> Optional[np.ndarray]:
    """
    PySpiceでマウント寄生を含まないコンデンサ単体のACインピーダンスを測定
    
    Args:
        model_path: SPICEモデルパス
        f_grid: 周波数グリッド
        dc_bias: DCバイアス電圧
    
    Returns:
        複素インピーダンス配列（失敗時はNone）
    """
    if not PYSPICE_AVAILABLE:
        return None
    
    try:
        f_grid = ensure_numpy(f_grid)
        # サブサーキット解析
        subckt_name, _ = parse_spice_model(model_path)
        if not subckt_name:
            return None
        
        # 回路作成
        circuit = Circuit('AC_Impedance_Measurement')
        
        # モデルファイルをインクルード
        abs_path = model_path.resolve()
        circuit.include(str(abs_path))
        
        # ノード定義
        n1 = 'n1'
        
        # AC電流源（1A）を使用してインピーダンスを測定
        # GND -> Iac -> n1 -> DUT -> GND
        circuit.SinusoidalCurrentSource(
            'Iac',
            circuit.gnd,  # 電流源の＋端子（電流の始点）
            n1,  # 電流源の－端子（DUTへ注入）
            frequency=100e3,
            delay=0,
            damping_factor=0,
            ac_magnitude=1
        )
        
        # DUTを測定端子に直結（マウント寄生はPDN側で加算）
        circuit.X('DUT', subckt_name, n1, circuit.gnd)
        
        # DCバイアス電圧源（必要な場合）
        bias_resistance = 1e6
        if dc_bias > 0:
            circuit.V('bias', 'vbias', circuit.gnd, dc_bias)
            circuit.R('bias', 'vbias', n1, bias_resistance)
        
        # シミュレータ作成
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        simulator.save(["all", f"v({n1})"])
        
        f_min, f_max = float(f_grid[0]), float(f_grid[-1])
        if f_max == f_min:
            analysis = simulator.ac(
                start_frequency=f_min, stop_frequency=f_max, number_of_points=1, variation='lin'
            )
        else:
            # .ac dec の number_of_points は1 decadeあたりの点数。解析点はグリッド点と一致しないため、
            # 自己共振付近の補間誤差を抑えるようグリッド密度の AC_OVERSAMPLE 倍で解析する。
            decades = np.log10(f_max / f_min)
            points_per_decade = max(1, int(np.ceil(AC_OVERSAMPLE * (len(f_grid) - 1) / decades)))
            analysis = simulator.ac(
                start_frequency=f_min, stop_frequency=f_max,
                number_of_points=points_per_decade, variation='dec'
            )
        
        freq = analysis.frequency.as_ndarray(scale=True)
        voltage = analysis[n1].as_ndarray(scale=True)
        # AC電流は1 A。バイアス抵抗に流れる電流を差し引いてDUT電流を求める。
        z_complex = voltage / (1.0 - voltage / bias_resistance) if dc_bias > 0 else voltage
        if not validate_result(z_complex, "PySpice AC結果"):
            return None

        # 周波数グリッドに補間（必要な場合）
        if len(freq) != len(f_grid) or not np.allclose(freq, f_grid, rtol=1e-3):
            # 対数補間（NumPyで実行）
            z_real_interp = log_interpolate(f_grid, freq, np.real(z_complex), np)
            z_imag_interp = log_interpolate(f_grid, freq, np.imag(z_complex), np)
            z_complex = z_real_interp + 1j * z_imag_interp
        
        # Netlistファイル出力（デバッグ用）
        netlist_file = Path(f"pyspice_{model_path.stem}.cir")
        with open(netlist_file, 'w') as f:
            f.write(str(circuit))
        logger.debug(f"Netlist出力: {netlist_file}")
        
        return z_complex
        
    except Exception as e:
        logger.error(f"PySpiceシミュレーションエラー: {e}")
        traceback.print_exc()
        return None


def apply_vector_fitting(z_samples: np.ndarray, f_grid: np.ndarray,
                        n_poles: int = 10) -> Optional[np.ndarray]:
    """
    VectorFittingを適用してインピーダンスを近似
    
    Args:
        z_samples: サンプルインピーダンス
        f_grid: 周波数グリッド
        n_poles: 極の数
    
    Returns:
        近似されたインピーダンス（失敗時はNone）
    """
    if not SKRF_AVAILABLE:
        return None
    
    try:
        z_samples = ensure_numpy(z_samples)
        f_grid = ensure_numpy(f_grid)
        if f_grid.size < 4:
            return None

        network = rf.Network(
            frequency=rf.Frequency.from_f(f_grid, unit='Hz'),
            z=z_samples.reshape(-1, 1, 1),
        )
        # 1点あたりのデータから決められる範囲に極数を抑える。
        model_order = min(n_poles, f_grid.size // 2)
        n_poles_cmplx = (model_order - 1) // 2
        vf = VectorFitting(network)
        with warnings.catch_warnings():
            # 収束しない場合の警告は、下の再現誤差チェックで判定する
            warnings.simplefilter('ignore', RuntimeWarning)
            vf.vector_fit(
                n_poles_real=model_order - 2 * n_poles_cmplx,
                n_poles_cmplx=n_poles_cmplx,
                init_pole_spacing='log',
                parameter_type='z',
                fit_proportional=True,  # jωESL を比例項で表す
            )
        z_fitted = vf.get_model_response(0, 0, f_grid)

        # 検証: 自己共振の鋭い谷などを取り逃がしたモデルは採用しない
        if not validate_result(z_fitted, "VectorFitting結果"):
            return None
        rel_err = float(np.max(np.abs(z_fitted - z_samples) / np.maximum(np.abs(z_samples), 1e-30)))
        if rel_err > VF_TOLERANCE:
            logger.info(f"VectorFittingの再現誤差 {rel_err:.1%} が許容値を超えたためAC解析結果を使用します")
            return None
        
        return z_fitted
        
    except Exception as e:
        logger.warning(f"VectorFittingエラー: {e}")
        traceback.print_exc()
        return None


def calculate_rlc_impedance(cap_config: Dict, f_grid: np.ndarray,
                           xp: Any = np, L_mntN: float = 0.2e-10) -> np.ndarray:
    """直列RLCモデルでコンデンサ単体のインピーダンスを返す。

    L_mntN は呼び出し互換性のために受け付ける。マウント寄生はPDN側で加算する。
    """
    C = to_float(cap_config.get('C'), 1e-6)
    ESR = to_float(cap_config.get('ESR'), 15e-3)
    ESL = to_float(cap_config.get('ESL'), 2e-10)

    omega = 2 * xp.pi * transfer_to_device(f_grid, xp)
    omega_safe = xp.where(omega == 0, xp.finfo(omega.dtype).tiny, omega)
    return ESR + 1j * omega * ESL - 1j / (omega_safe * max(C, 1e-24))


def estimate_capacitance_from_impedance(z_c: np.ndarray, f_grid: np.ndarray,
                                       method: str = 'median') -> float:
    """
    インピーダンスから容量値を推定（コンデンサモジュール版）

    Args:
        z_c: 複素インピーダンス配列
        f_grid: 周波数グリッド [Hz]
        method: 推定方法 ('median', 'low_freq', 'lstsq')

    Returns:
        推定容量 [F]

    Notes:
        - 'median': 全周波数帯の中央値（ロバスト）
        - 'low_freq': 低周波数領域のみ（容量性リアクタンス）
        - 'lstsq': 最小二乗法（精度高いが外れ値に弱い）
    """
    z_np = ensure_numpy(z_c)
    f_np = ensure_numpy(f_grid)
    omega = 2 * np.pi * f_np

    if method == 'median':
        # 全周波数帯から推定（中央値でロバストに）
        omega_imag = omega * np.imag(z_np)
        omega_imag_safe = np.where(np.abs(omega_imag) < 1e-30, 1e-30, omega_imag)
        c_samples = -1.0 / omega_imag_safe

        # 有効範囲でフィルタ（1pF～100mF）
        valid_mask = (c_samples >= 1e-12) & (c_samples <= 0.1)
        if np.sum(valid_mask) > 0:
            return float(np.median(c_samples[valid_mask]))
        return 1e-6  # デフォルト

    elif method == 'low_freq':
        # 低周波数領域（最初の20%）
        n_low = max(1, len(f_np) // 5)
        f_low = f_np[:n_low]
        z_low = z_np[:n_low]
        w_low = 2 * np.pi * f_low

        # 容量性領域のみ（Im(Z) < 0）
        imag_z = np.imag(z_low)
        mask = imag_z < 0
        if np.sum(mask) > 0:
            c_samples = -1.0 / (w_low[mask] * imag_z[mask])
            valid_mask = (c_samples >= 1e-12) & (c_samples <= 0.1)
            if np.sum(valid_mask) > 0:
                return float(np.median(c_samples[valid_mask]))
        return 1e-6

    elif method == 'lstsq':
        # 列の大きさを揃え、広帯域グリッドで小さい特異値が切り捨てられるのを防ぐ。
        valid = (omega > 0) & np.isfinite(omega) & np.isfinite(z_np.imag)
        w = omega[valid]
        if w.size >= 2:
            w_min, w_max = w.min(), w.max()
            design = np.column_stack((w / w_max, -w_min / w))
            try:
                coefficients, _, rank, _ = np.linalg.lstsq(design, z_np.imag[valid], rcond=None)
            except np.linalg.LinAlgError:
                return 1e-6
            inv_c = coefficients[1] * w_min
            if rank == 2 and 10 <= inv_c <= 1e12:
                return float(1.0 / inv_c)
        return 1e-6

    else:
        raise ValueError(f"未知の推定方法: {method}")


def _estimate_intercept(x: np.ndarray, y: np.ndarray) -> float:
    """周波数の離れた点を組にして外挿し、切片の中央値を返す。"""
    half = len(x) // 2
    if half:
        # x が小さい側から大きい側への比を使い、単位や桁の影響を避ける。
        first, last = x[:half], x[-half:]
        distinct = last > first
        if np.any(distinct):
            ratio = first[distinct] / last[distinct]
            intercepts = (y[:half][distinct] - ratio * y[-half:][distinct]) / (1 - ratio)
            return float(np.median(intercepts))
    return float(np.median(y))


def estimate_rlc_parameters(z_c: np.ndarray, f_grid: np.ndarray) -> Tuple[float, float, float]:
    """コンデンサ単体の等価 ESR [Ω]、ESL [H]、C [F] を推定する。"""
    z_np = ensure_numpy(z_c).ravel()
    freq = ensure_numpy(f_grid).ravel()
    valid = (freq > 0) & np.isfinite(freq) & np.isfinite(z_np)
    if not np.any(valid):
        return 0.0, 0.5e-9, 1e-6

    freq, z_np = freq[valid], z_np[valid]
    order = np.argsort(freq)
    omega = 2 * np.pi * freq[order].astype(np.float64, copy=False)
    reactance = z_np.imag[order].astype(np.float64, copy=False)
    # ESR は自己共振（|Z|最小）点の実部
    esr = float(z_np.real[order][np.argmin(np.abs(z_np[order]))])
    tail_size = max(2, len(omega) // 5)

    capacitive = np.flatnonzero(reactance < 0)
    inductive = np.flatnonzero(reactance > 0)
    # 片側の領域が不足する場合も、帯域端の複数点で直列RLCを外挿する。
    low = capacitive[:tail_size] if capacitive.size >= 2 else slice(0, tail_size)
    high = inductive[-tail_size:][::-1] if inductive.size >= 2 else slice(-1, -tail_size - 1, -1)

    # -ω Im(Z) = 1/C - ESL·ω²、Im(Z)/ω = ESL - (1/C)/ω²
    low_omega, high_omega = omega[low], omega[high]
    inv_c = _estimate_intercept((low_omega / low_omega[-1]) ** 2, -low_omega * reactance[low])
    inductance_samples = reactance[high] / high_omega
    esl = _estimate_intercept((high_omega[-1] / high_omega) ** 2, inductance_samples)

    # ESLがゼロのモデルでは外挿の丸め誤差を除く。
    if np.issubdtype(z_np.real.dtype, np.floating):
        precision = np.finfo(z_np.real.dtype).eps
        if abs(esl) <= precision * np.median(np.abs(inductance_samples)):
            esl = 0.0

    if not np.isfinite(esl) or esl < 0:
        esl = 0.5e-9
    capacitance = 1.0 / inv_c if np.isfinite(inv_c) and inv_c > 0 else 1e-6
    return esr, esl, capacitance


def calculate_single_capacitor_impedance(cap_config: Dict, f_grid: np.ndarray,
                                        model_path: Path, dc_bias: float,
                                        xp: Any = np, L_mntN: float = 0.2e-10,
                                        dtype: Optional[Any] = None) -> Tuple[np.ndarray, float]:
    """
    単一コンデンサの部品単体インピーダンスを計算
    
    Args:
        cap_config: コンデンサ設定
        f_grid: 周波数グリッド
        model_path: SPICEモデルのベースパス
        dc_bias: DCバイアス
        xp: バックエンドモジュール
        L_mntN: 呼び出し互換性のための引数（計算には使用しない）
        dtype: 戻り値の配列データ型
    
    Returns:
        (z_c, capacitance): インピーダンスと容量値
    """
    name = cap_config['name']
    
    z_c = None
    capacitance = to_float(cap_config.get('C'))
    
    # SPICEモデルが指定されている場合
    if 'path' in cap_config and cap_config['path']:
        spice_path = Path(cap_config['path'])
        
        # 相対パスの場合はmodel_pathを基準にする
        if not spice_path.is_absolute():
            spice_path = model_path / spice_path

        if not spice_path.exists():
            candidates = []
            if spice_path.suffix == '':
                candidates.extend([
                    spice_path.with_suffix('.mod'),
                    spice_path.with_suffix('.MOD'),
                    spice_path.with_suffix('.ckt'),
                    spice_path.with_suffix('.CKT')
                ])
            elif spice_path.suffix.lower() not in ['.mod', '.ckt']:
                candidates.extend([
                    spice_path.with_suffix('.mod'),
                    spice_path.with_suffix('.CKT')
                ])

            for candidate in candidates:
                if candidate.exists():
                    spice_path = candidate
                    break

        if spice_path.exists():
            # PySpiceでACサンプリング
            f_host = ensure_numpy(f_grid)
            z_samples = simulate_ac_impedance(spice_path, f_host, dc_bias)
            
            if z_samples is not None:
                # VectorFitting適用
                z_fitted = apply_vector_fitting(z_samples, f_host)
                if z_fitted is not None:
                    z_c = z_fitted
                    logger.info(f"{name}: SPICEモデル + VectorFitting成功")
                else:
                    z_c = z_samples
                    logger.info(f"{name}: SPICEモデル使用（VectorFittingスキップ）")
                
                # 設定に容量が無い場合は低域から推定する。
                if capacitance is None:
                    capacitance = estimate_capacitance_from_impedance(
                        z_c, f_host, method='low_freq'
                    )
                    if capacitance >= 1e-6:
                        logger.info(f"{name}: 推定容量 = {capacitance*1e6:.3f}μF")
                    else:
                        logger.info(f"{name}: 推定容量 = {capacitance*1e9:.3f}nF")
        else:
            logger.warning(f"SPICEモデルが見つかりません: {spice_path}")
    
    # RLCフォールバック
    if z_c is None:
        z_c = calculate_rlc_impedance(cap_config, f_grid, xp)
        logger.info(f"{name}: RLCモデル使用")
        
        # 容量値確認
        if capacitance is None:
            capacitance = to_float(cap_config.get('C'), 1e-6)
    
    # GPU転送（必要な場合）
    z_c = transfer_to_device(z_c, xp, dtype)

    return z_c, capacitance


def calculate_all_capacitor_impedances(config: UserConfig, f_grid: np.ndarray,
                                      xp: Any = np,
                                      gui_callback: Optional[Callable] = None,
                                      dtype: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """
    全コンデンサの部品単体インピーダンスを計算（マウント寄生はPDN側で加算）
    
    Args:
        config: ユーザー設定
        f_grid: 周波数グリッド
        xp: バックエンドモジュール
        gui_callback: GUI更新コールバック
        dtype: 戻り値の配列データ型
    
    Returns:
        {name: z_c} の辞書
    """
    capacitor_impedances = {}
    capacitances = {}
    
    model_path = Path(config.model_path)
    gui_frequency = transfer_to_host(f_grid) if gui_callback else None
    
    progress = get_progress_bar(
        config.capacitors,
        desc="コンデンサZ_c計算",
        total=len(config.capacitors)
    )
    
    for cap_config in progress:
        raw_name = cap_config.get('name')
        name = str(raw_name).strip() if raw_name is not None else ""
        if not name:
            path_value = cap_config.get('path')
            inferred = Path(str(path_value)).stem if path_value else ""
            if not inferred:
                raise ValueError("コンデンサ設定に名前が見つかりません (path も指定されていません)")
            name = inferred
            cap_config['name'] = name
        
        # インピーダンス計算
        z_c, capacitance = calculate_single_capacitor_impedance(
            cap_config,
            f_grid,
            model_path,
            config.dc_bias,
            xp,
            dtype=dtype,
        )
        
        capacitor_impedances[name] = z_c
        capacitances[name] = capacitance
        
        # GUI更新
        if gui_callback:
            gui_callback({
                'type': 'capacitor_update',
                'name': name,
                'z_c': transfer_to_host(z_c),
                'capacitance': capacitance,
                'frequency': gui_frequency
            })
    
    logger.info(f"各コンデンサZ_c計算完了：{len(capacitor_impedances)}種類")
    
    # 容量でソート情報を出力
    sorted_names = sorted(capacitances.keys(), key=lambda k: capacitances[k])
    for name in sorted_names:
        c_value = capacitances[name]
        c_str = f"{c_value*1e6:.3f}μF" if c_value >= 1e-6 else f"{c_value*1e9:.3f}nF"
        logger.info(f"{name}: {c_str}")
    
    return capacitor_impedances
