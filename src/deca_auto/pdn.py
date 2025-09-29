"""
PDN合成モジュール
ラダー回路の組み立てとGPU最適化されたインピーダンス並列計算
"""

import traceback
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# 絶対パスでインポート
from deca_auto.config import UserConfig
from deca_auto.utils import logger, validate_result, safe_divide


def _build_config_lookup(config: UserConfig) -> Dict[str, Dict[str, Any]]:
    """コンデンサ名→設定辞書のルックアップを作成"""
    return {cap['name']: cap for cap in config.capacitors}


def _prepare_capacitor_arrays(
    capacitor_impedances: Dict[str, np.ndarray],
    config: UserConfig,
    parasitic_elements: Dict[str, np.ndarray],
    xp: Any,
    omega: Optional[np.ndarray] = None,
    dtype: Optional[Any] = None,
) -> Tuple[Tuple[str, ...], Any, Any, Any]:
    """ラダー計算用にコンデンサ関連配列を整形"""

    cap_names: Tuple[str, ...] = tuple(capacitor_impedances.keys())
    if not cap_names:
        raise ValueError("コンデンサインピーダンス辞書が空です")

    z_cap_array = xp.stack([capacitor_impedances[name] for name in cap_names])
    if dtype is not None:
        z_cap_array = z_cap_array.astype(dtype, copy=False)

    # マウントインダクタンス配列を構築（個別指定があれば優先）
    z_mnt_default = parasitic_elements['z_mntN']
    if omega is None:
        omega = parasitic_elements.get('omega')
    if omega is None:
        raise ValueError("omega が取得できませんでした")

    config_map = _build_config_lookup(config)
    z_mnt_array = xp.empty_like(z_cap_array)
    for idx, name in enumerate(cap_names):
        cap_cfg = config_map.get(name)
        if cap_cfg and cap_cfg.get('L_mnt') is not None:
            z_mnt_array[idx] = xp.asarray(1j * omega * cap_cfg['L_mnt'], dtype=z_cap_array.dtype)
        else:
            z_mnt_array[idx] = xp.asarray(z_mnt_default, dtype=z_cap_array.dtype)

    return cap_names, z_cap_array, z_mnt_array, omega


def calculate_pdn_parasitic_elements(f_grid: np.ndarray, config: UserConfig,
                                    xp: Any = np,
                                    dtype: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """
    PDN寄生成分のインピーダンス/アドミタンスを計算
    
    Args:
        f_grid: 周波数グリッド
        config: ユーザー設定
        xp: バックエンドモジュール
    
    Returns:
        寄生成分の辞書（'omega'も含む）
    """
    omega = 2 * xp.pi * f_grid
    
    # VRM (Voltage Regulator Module)
    # Y_vrm = 1/(R_vrm + jωL_vrm)
    z_vrm = config.R_vrm + 1j * omega * config.L_vrm
    y_vrm = 1.0 / z_vrm
    
    # Via (BGAビア)
    # Z_v = R_v + jωL_v
    z_v = config.R_v + 1j * omega * config.L_v
    
    # Spreading (拡散抵抗・インダクタンス)
    # Z_s = R_s + jωL_s
    z_s = config.R_s + 1j * omega * config.L_s
    
    # Spreading per decap (デカップリングコンデンサ用)
    # Z_sN = R_sN + jωL_sN
    z_sN = config.R_sN + 1j * omega * config.L_sN
    
    # Planar (プレーナ容量と損失)
    # 損失正接を考慮: Y_p = ωC_p(tanδ + j) / (1 + R_p*ωC_p(tanδ + j))
    tan_delta = config.tan_delta_p
    planar_admittance = omega * config.C_p * (tan_delta + 1j)
    denominator = 1 + config.R_p * planar_admittance
    y_p = safe_divide(planar_admittance, denominator, fill_value=1e-12, xp=xp)
    
    # Mounting inductance (マウントインダクタンス)
    # デフォルト値
    z_mntN = 1j * omega * config.L_mntN
    
    if dtype is not None:
        target = np.dtype(dtype)
        y_vrm = y_vrm.astype(target, copy=False)
        z_v = z_v.astype(target, copy=False)
        z_s = z_s.astype(target, copy=False)
        z_sN = z_sN.astype(target, copy=False)
        y_p = y_p.astype(target, copy=False)
        z_mntN = z_mntN.astype(target, copy=False)

    return {
        'y_vrm': y_vrm,
        'z_v': z_v,
        'z_s': z_s,
        'z_sN': z_sN,
        'y_p': y_p,
        'z_mntN': z_mntN,
        'omega': omega  # omegaも保存
    }


def _pdn_impedance_core(
    count_vectors: Any,
    z_cap_array: Any,
    z_mnt_array: Any,
    parasitic_elements: Dict[str, Any],
    xp: Any,
) -> Any:
    """ラダー回路のPDNインピーダンスをバッチ計算"""

    count_vectors = xp.asarray(count_vectors)
    if count_vectors.ndim == 1:
        count_vectors = count_vectors[xp.newaxis, :]

    n_batch = count_vectors.shape[0]
    z_cap_array = xp.asarray(z_cap_array)
    z_mnt_array = xp.asarray(z_mnt_array)

    if z_cap_array.ndim == 2:
        z_cap_array = xp.broadcast_to(z_cap_array, (n_batch,) + z_cap_array.shape)
    if z_mnt_array.ndim == 2:
        z_mnt_array = xp.broadcast_to(z_mnt_array, (n_batch,) + z_mnt_array.shape)

    if z_cap_array.shape[0] != n_batch or z_mnt_array.shape[0] != n_batch:
        raise ValueError("コンデンサ配列とカウントベクトルのバッチ数が一致していません")

    z_with_mount = z_cap_array + z_mnt_array
    n_caps = z_with_mount.shape[1]

    y_vrm = parasitic_elements['y_vrm']
    y_total = xp.tile(y_vrm, (n_batch, 1))

    for cap_idx in range(n_caps - 1, -1, -1):
        idx = xp.where(count_vectors[:, cap_idx] > 0)[0]
        if idx.size == 0:
            continue

        counts = count_vectors[idx, cap_idx].astype(z_with_mount.dtype, copy=False)
        inv_z_cm = safe_divide(1.0, z_with_mount[idx, cap_idx, :], fill_value=0.0, xp=xp)
        y_cm = counts[:, None] * inv_z_cm

        z_series = parasitic_elements['z_sN'] + safe_divide(1.0, y_cm, fill_value=0.0, xp=xp)
        y_total[idx] = y_total[idx] + safe_divide(1.0, z_series, fill_value=0.0, xp=xp)

    y_with_planar = y_total + parasitic_elements['y_p']
    z_after_spreading = parasitic_elements['z_s'] + safe_divide(1.0, y_with_planar, fill_value=0.0, xp=xp)
    z_pdn = parasitic_elements['z_v'] + z_after_spreading

    return z_pdn


def assemble_pdn_ladder(count_vector: np.ndarray,
                       capacitor_impedances: Dict[str, np.ndarray],
                       capacitor_indices: np.ndarray,
                       parasitic_elements: Dict[str, np.ndarray],
                       config: UserConfig,
                       xp: Any = np) -> np.ndarray:
    """PDNラダー回路を組み立ててZ_pdnを計算（単一の組み合わせ）"""

    try:
        omega = parasitic_elements.get('omega')
        sample_dtype = next(iter(capacitor_impedances.values())).dtype
        cap_names, z_cap_array, z_mnt_array, _ = _prepare_capacitor_arrays(
            capacitor_impedances,
            config,
            parasitic_elements,
            xp,
            omega,
            sample_dtype,
        )
        _ = cap_names  # 順序保持（測定ノード側が小容量）の確認用
        z_pdn_batch = _pdn_impedance_core(
            count_vector,
            z_cap_array,
            z_mnt_array,
            parasitic_elements,
            xp,
        )
        return z_pdn_batch[0]
    except Exception as exc:
        logger.error(f"PDNラダー計算エラー: {exc}")
        traceback.print_exc()
        raise


def calculate_pdn_impedance_batch(count_vectors: np.ndarray,
                                 capacitor_impedances: Dict[str, np.ndarray],
                                 capacitor_indices: np.ndarray,
                                 f_grid: np.ndarray,
                                 config: UserConfig,
                                 xp: Any = np) -> np.ndarray:
    """
    PDNインピーダンスをバッチ計算（GPU最適化）
    
    Args:
        count_vectors: カウントベクトル配列 (N_batch, N_capacitors)
        capacitor_impedances: コンデンサインピーダンス辞書
        capacitor_indices: コンデンサインデックス
        f_grid: 周波数グリッド
        config: ユーザー設定
        xp: バックエンドモジュール
    
    Returns:
        Z_pdn配列 (N_batch, N_freq)
    """
    target_dtype = np.dtype(getattr(config, 'dtype_c', 'complex64'))
    parasitic_elements = calculate_pdn_parasitic_elements(f_grid, config, xp, target_dtype)
    cap_names, z_cap_array, z_mnt_array, _ = _prepare_capacitor_arrays(
        capacitor_impedances,
        config,
        parasitic_elements,
        xp,
        parasitic_elements.get('omega', 2 * xp.pi * f_grid),
        target_dtype,
    )
    _ = cap_names  # 順序維持の意図を明示

    z_pdn_batch = _pdn_impedance_core(
        count_vectors,
        z_cap_array,
        z_mnt_array,
        parasitic_elements,
        xp,
    ).astype(z_cap_array.dtype, copy=False)

    if not validate_result(z_pdn_batch, "PDNインピーダンス"):
        logger.error("PDNインピーダンス計算で異常値を検出")

    return z_pdn_batch


def calculate_pdn_impedance_monte_carlo(count_vector: np.ndarray,
                                       capacitor_impedances: Dict[str, np.ndarray],
                                       capacitor_indices: np.ndarray,
                                       f_grid: np.ndarray,
                                       config: UserConfig,
                                       n_samples: int,
                                       xp: Any = np) -> np.ndarray:
    """
    Monte Carlo法でPDNインピーダンスを計算（ロバスト性評価）
    
    Args:
        count_vector: カウントベクトル（単一）
        capacitor_impedances: コンデンサインピーダンス辞書
        capacitor_indices: コンデンサインデックス
        f_grid: 周波数グリッド
        config: ユーザー設定
        n_samples: MCサンプル数
        xp: バックエンドモジュール
    
    Returns:
        Z_pdn_mc配列 (n_samples, n_freq)
    """
    n_caps = len(capacitor_impedances)

    if n_samples <= 0:
        return xp.zeros((0, len(f_grid)), dtype=xp.asarray(f_grid).dtype)

    if xp is np:
        rng = np.random.default_rng(config.seed)
        normal = rng.normal
    else:
        rng = xp.random.RandomState(config.seed)
        normal = lambda loc, scale, size: rng.normal(loc, scale, size=size, dtype=xp.float32)

    target_dtype = np.dtype(getattr(config, 'dtype_c', 'complex64'))
    parasitic_elements = calculate_pdn_parasitic_elements(f_grid, config, xp, target_dtype)
    _, z_cap_base, z_mnt_array, _ = _prepare_capacitor_arrays(
        capacitor_impedances,
        config,
        parasitic_elements,
        xp,
        parasitic_elements.get('omega'),
        target_dtype,
    )

    esr_orig = xp.mean(xp.real(z_cap_base), axis=1)
    z_imag_base = xp.imag(z_cap_base)

    shape = (n_samples, n_caps)
    float_dtype = z_cap_base.real.dtype
    c_variation = xp.clip(normal(1.0, config.tol_C / 3, shape), 1 - config.tol_C, 1 + config.tol_C).astype(float_dtype, copy=False)
    esr_variation = xp.clip(normal(1.0, config.tol_ESR / 3, shape), 1 - config.tol_ESR, 1 + config.tol_ESR).astype(float_dtype, copy=False)
    esl_variation = xp.clip(normal(1.0, config.tol_ESL / 3, shape), 1 - config.tol_ESL, 1 + config.tol_ESL).astype(float_dtype, copy=False)

    c_variation = c_variation * (1 - config.mlcc_derating)

    esr_new = esr_orig[None, :] * esr_variation
    imag_scale = xp.sqrt(c_variation * esl_variation)
    z_imag_new = imag_scale[:, :, None] * z_imag_base[None, :, :]
    z_cap_varied = esr_new[:, :, None] + 1j * z_imag_new

    count_vector = xp.asarray(count_vector)
    count_vectors_mc = xp.broadcast_to(count_vector, (n_samples, count_vector.size))

    z_pdn_mc = _pdn_impedance_core(
        count_vectors_mc,
        z_cap_varied,
        z_mnt_array,
        parasitic_elements,
        xp,
    )

    return z_pdn_mc.astype(z_cap_base.dtype, copy=False)
