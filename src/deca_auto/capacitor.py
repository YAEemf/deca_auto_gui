"""
コンデンサインピーダンス計算モジュール
SPICEモデル読み込み、PySpice ACサンプリング、VectorFitting、RLCフォールバック
"""

import re
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable
import numpy as np

# 絶対パスでインポート
from deca_auto.config import UserConfig
from deca_auto.utils import (
    logger, Timer, get_progress_bar, transfer_to_device,
    log_interpolate, validate_result, ensure_numpy, to_float
)

# 条件付きインポート
try:
    import PySpice
    import PySpice.Logging.Logging as Logging
    from PySpice.Spice.Netlist import Circuit, SubCircuit
    from PySpice.Unit import *
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    PYSPICE_AVAILABLE = True
    # PySpiceのログレベルを調整
    logger_pyspice = Logging.setup_logging(logging_level='WARNING')
except ImportError:
    PYSPICE_AVAILABLE = False
    logger.warning("PySpiceが利用できません。RLCモデルのみ使用します")

try:
    import skrf as rf
    from skrf.vi.vf import VectorFitting
    SKRF_AVAILABLE = True
except ImportError:
    SKRF_AVAILABLE = False
    logger.warning("scikit-rfが利用できません。VectorFittingは使用できません")


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
    PySpiceでACインピーダンスをシミュレーション
    
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
        # サブサーキット解析
        subckt_name, pins = parse_spice_model(model_path)
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
            circuit.gnd,  # negative
            n1,  # positive
            # dc_offset=dc_bias@u_A,
            # amplitude=1@u_A,
            frequency=100@u_kHz,
            delay=0@u_s,
            damping_factor=0,
            ac_magnitude=1@u_A
        )
        
        # DUTをサブサーキットとして接続
        circuit.X('DUT', subckt_name, n1, circuit.gnd)
        
        # DCバイアス電圧源（必要な場合）
        if dc_bias > 0:
            circuit.V('bias', 'vbias', circuit.gnd, dc_bias@u_V)
            circuit.R('bias', 'vbias', n1, 1@u_MOhm)  # 高抵抗でDCバイアスを印加
        
        # シミュレータ作成
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        simulator.save(["all", f"v({n1})"])
        
        # AC解析
        # 周波数点を対数スケールで設定
        f_min = float(f_grid[0])
        f_max = float(f_grid[-1])
        points_per_decade = len(f_grid) / np.log10(f_max / f_min)
        
        analysis = simulator.ac(
            start_frequency=f_min@u_Hz,
            stop_frequency=f_max@u_Hz,
            number_of_points=len(f_grid),
            variation='dec'  # 対数スケール
        )
        
        # 結果取得
        freq = np.array([float(f) for f in analysis.frequency])
        # V(n1) / I = Z (I=1Aなので V=Z)
        voltage = analysis[n1].as_ndarray()
        z_complex = voltage  # 1Aの電流源なので電圧=インピーダンス
        
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
        # NetworkオブジェクトをS11形式で作成
        # Z -> S11変換: S11 = (Z - Z0) / (Z + Z0), Z0=50Ω
        z0 = 50.0
        s11 = (z_samples - z0) / (z_samples + z0)
        
        # Networkオブジェクト作成（1ポート）
        network = rf.Network()
        network.frequency = rf.Frequency.from_f(f_grid, unit='Hz')
        network.s = s11.reshape(-1, 1, 1)  # (F, N, N)形状
        network.z0 = z0
        
        # VectorFitting実行
        vf = VectorFitting(network)
        vf.vector_fit(
            n_poles=n_poles,
            poles_init='linlogspaced',
            parameter_type='z',  # インピーダンスとしてフィット
            n_iterations=20
        )
        
        # フィット結果を取得
        z_fitted = vf.get_model_response(network.frequency.f)
        
        # 1次元配列に戻す
        if hasattr(z_fitted, 'squeeze'):
            z_fitted = z_fitted.squeeze()
        else:
            z_fitted = np.squeeze(z_fitted)
        
        # 検証
        if not validate_result(z_fitted, "VectorFitting結果"):
            return None
        
        return z_fitted
        
    except Exception as e:
        logger.warning(f"VectorFittingエラー: {e}")
        traceback.print_exc()
        return None


def calculate_rlc_impedance(cap_config: Dict, f_grid: np.ndarray,
                           xp: Any = np, L_mntN: float = 0.5e-9) -> np.ndarray:
    """
    RLCモデルでインピーダンスを計算（フォールバック）
    
    Args:
        cap_config: コンデンサ設定
        f_grid: 周波数グリッド
        xp: バックエンドモジュール
        L_mntN: デフォルトマウントインダクタンス
    
    Returns:
        複素インピーダンス
    """
    # パラメータ取得（デフォルト値使用）
    C = to_float(cap_config.get('C'), 1e-6)
    ESR = to_float(cap_config.get('ESR'), 10e-3)
    ESL = to_float(cap_config.get('ESL'), 0.5e-9)
    
    # L_mntの処理：Noneの場合はデフォルト値を使用
    L_mnt = to_float(cap_config.get('L_mnt'), L_mntN)
    
    # 角周波数
    f_backend = xp.asarray(f_grid)
    omega = 2 * xp.pi * f_backend
    omega_safe = xp.where(omega == 0, xp.finfo(f_backend.dtype).tiny, omega)

    # 直列RLC+L_mnt: Z = ESR + j*ω*ESL - j/(ω*C) + j*ω*L_mnt
    z_c = (
        ESR
        + 1j * omega * ESL
        - 1j / (omega_safe * max(C, 1e-24))
        + 1j * omega * L_mnt
    )

    return z_c


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
        # 最小二乗法でRLC推定
        y = np.imag(z_np)
        X = np.vstack([omega, -1.0 / np.maximum(omega, 1e-300)]).T
        try:
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            _, invC = theta
            if invC > 1e-6:  # 妥当性チェック
                c_est = 1.0 / invC
                if 1e-12 <= c_est <= 0.1:
                    return float(c_est)
        except Exception:
            pass
        return 1e-6

    else:
        raise ValueError(f"未知の推定方法: {method}")


def estimate_rlc_parameters(z_c: np.ndarray, f_grid: np.ndarray) -> Tuple[float, float, float]:
    """
    直列RLCパラメータを推定（ESR, ESL, C）

    Args:
        z_c: 複素インピーダンス配列
        f_grid: 周波数グリッド [Hz]

    Returns:
        (ESR, ESL, C): 推定されたESR [Ω], ESL [H], 容量 [F]

    Notes:
        - ESR: 実部の中央値
        - ESL, C: 虚部から最小二乗法で推定
    """
    z_np = ensure_numpy(z_c)
    f_np = ensure_numpy(f_grid)
    omega = 2 * np.pi * f_np

    # ESR推定: 実部の中央値
    esr = float(np.median(np.real(z_np)))

    # LとCの推定: Im(Z) = ωL - 1/(ωC)
    y = np.imag(z_np)
    X = np.vstack([omega, -1.0 / np.maximum(omega, 1e-300)]).T

    try:
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        L_est, invC_est = theta

        # 容量の妥当性チェック
        if invC_est <= 1e-6:
            # 異常値の場合は別の方法で推定
            C_est = estimate_capacitance_from_impedance(z_c, f_grid, method='low_freq')
        else:
            C_est = 1.0 / invC_est
            if not (1e-12 <= C_est <= 0.1):
                C_est = estimate_capacitance_from_impedance(z_c, f_grid, method='low_freq')

        # インダクタンスの範囲チェック
        if L_est < 0 or L_est > 100e-6:
            L_est = 0.5e-9  # デフォルト

        return float(esr), float(L_est), float(C_est)

    except Exception as e:
        logger.warning(f"RLCパラメータ推定エラー: {e}")
        # フォールバック
        C_est = estimate_capacitance_from_impedance(z_c, f_grid, method='low_freq')
        return esr, 0.5e-9, C_est


def calculate_single_capacitor_impedance(cap_config: Dict, f_grid: np.ndarray,
                                        model_path: Path, dc_bias: float,
                                        xp: Any = np, L_mntN: float = 0.5e-9,
                                        dtype: Optional[Any] = None) -> Tuple[np.ndarray, float]:
    """
    単一コンデンサのインピーダンスを計算
    
    Args:
        cap_config: コンデンサ設定
        f_grid: 周波数グリッド
        model_path: SPICEモデルのベースパス
        dc_bias: DCバイアス
        xp: バックエンドモジュール
        L_mntN: デフォルトマウントインダクタンス
    
    Returns:
        (z_c, capacitance): インピーダンスと容量値
    """
    name = cap_config['name']
    logger.info(f"コンデンサ {name} のインピーダンス計算中...")
    
    z_c = None
    capacitance_raw = cap_config.get('C', None)
    capacitance = float(capacitance_raw) if capacitance_raw is not None else None
    
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
                    spice_path.with_suffix('.MOD')
                ])
            elif spice_path.suffix.lower() != '.mod':
                candidates.append(spice_path.with_suffix('.mod'))

            for candidate in candidates:
                if candidate.exists():
                    logger.info(f"{name}: 拡張子を補完してモデル {candidate.name} を使用します")
                    spice_path = candidate
                    break

        if spice_path.exists():
            # PySpiceでACサンプリング
            z_samples = simulate_ac_impedance(spice_path, f_grid, dc_bias)
            
            if z_samples is not None:
                # VectorFitting適用
                z_fitted = apply_vector_fitting(z_samples, f_grid)
                if z_fitted is not None:
                    z_c = z_fitted
                    try:
                        cap_config['ESR'] = 0.0
                        cap_config['ESL'] = 0.0
                        logger.debug(f"{name}: SPICE AC成功のためESR/ESL=0を適用（デフォルト不使用）")
                    except Exception as _:
                        pass
                    logger.info(f"{name}: SPICEモデル + VectorFitting成功")
                else:
                    z_c = z_samples
                    logger.info(f"{name}: SPICEモデル使用（VectorFittingスキップ）")
                
                # 容量推定（未定義の場合）- utils.pyの統合版を使用
                if capacitance is None:
                    capacitance = estimate_capacitance_from_impedance(
                        ensure_numpy(z_c),
                        ensure_numpy(f_grid),
                        method='low_freq'  # 低周波数領域からロバストに推定
                    )
                    if capacitance >= 1e-6:
                        logger.info(f"{name}: 推定容量 = {capacitance*1e6:.3f}μF")
                    else:
                        logger.info(f"{name}: 推定容量 = {capacitance*1e9:.3f}nF")
        else:
            logger.warning(f"SPICEモデルが見つかりません: {spice_path}")
    
    # RLCフォールバック
    if z_c is None:
        z_c = calculate_rlc_impedance(cap_config, f_grid, xp, L_mntN)
        logger.info(f"{name}: RLCモデル使用")
        
        # 容量値確認
        if capacitance is None:
            capacitance = to_float(cap_config.get('C'), 1e-6)
    
    # GPU転送（必要な場合）
    z_c = transfer_to_device(z_c, xp)
    if dtype is not None:
        z_c = z_c.astype(dtype, copy=False)

    return z_c, capacitance


def calculate_all_capacitor_impedances(config: UserConfig, f_grid: np.ndarray,
                                      xp: Any = np,
                                      gui_callback: Optional[Callable] = None,
                                      dtype: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """
    全コンデンサのインピーダンスを計算
    
    Args:
        config: ユーザー設定
        f_grid: 周波数グリッド
        xp: バックエンドモジュール
        gui_callback: GUI更新コールバック
    
    Returns:
        {name: z_c} の辞書
    """
    capacitor_impedances = {}
    capacitances = {}
    
    model_path = Path(config.model_path)
    
    progress = get_progress_bar(
        config.capacitors,
        desc="コンデンサZ_c計算",
        total=len(config.capacitors)
    )
    
    for cap_config in progress:
        name = cap_config['name']
        
        # インピーダンス計算
        z_c, capacitance = calculate_single_capacitor_impedance(
            cap_config,
            f_grid,
            model_path,
            config.dc_bias,
            xp,
            config.L_mntN,
            dtype,
        )
        
        capacitor_impedances[name] = z_c
        capacitances[name] = capacitance
        
        # GUI更新
        if gui_callback:
            gui_callback({
                'type': 'capacitor_update',
                'name': name,
                'z_c': transfer_to_device(z_c, np),
                'capacitance': capacitance,
                'frequency': transfer_to_device(f_grid, np)
            })
    
    logger.info(f"全{len(capacitor_impedances)}個のコンデンサのZ_c計算完了")
    
    # 容量でソート情報を出力
    sorted_names = sorted(capacitances.keys(), key=lambda k: capacitances[k])
    logger.info("容量順序（小→大）:")
    for name in sorted_names:
        c_value = capacitances[name]
        c_str = f"{c_value*1e6:.3f}μF" if c_value >= 1e-6 else f"{c_value*1e9:.3f}nF"
        logger.info(f"  {name}: {c_str}")
    
    return capacitor_impedances
