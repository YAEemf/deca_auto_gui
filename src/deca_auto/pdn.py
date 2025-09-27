"""
PDN合成モジュール
ラダー回路の組み立てとGPU最適化されたインピーダンス並列計算
"""

import traceback
from typing import Dict, List, Optional, Any
import numpy as np

# 絶対パスでインポート
from deca_auto.config import UserConfig
from deca_auto.utils import logger, validate_result, safe_divide


def calculate_pdn_parasitic_elements(f_grid: np.ndarray, config: UserConfig,
                                    xp: Any = np) -> Dict[str, np.ndarray]:
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
    
    return {
        'y_vrm': y_vrm,
        'z_v': z_v,
        'z_s': z_s,
        'z_sN': z_sN,
        'y_p': y_p,
        'z_mntN': z_mntN,
        'omega': omega  # omegaも保存
    }


def assemble_pdn_ladder(count_vector: np.ndarray,
                       capacitor_impedances: Dict[str, np.ndarray],
                       capacitor_indices: np.ndarray,
                       parasitic_elements: Dict[str, np.ndarray],
                       config: UserConfig,
                       xp: Any = np) -> np.ndarray:
    """
    PDNラダー回路を組み立ててZ_pdnを計算（単一の組み合わせ）
    
    測定ノード(Load) -> Z_v -> Z_s -> Y_p -> Σ(Z_sN, Y_cmN) -> Y_vrm
    
    Args:
        count_vector: コンデンサ個数ベクトル
        capacitor_impedances: コンデンサインピーダンス辞書
        capacitor_indices: コンデンサインデックス
        parasitic_elements: 寄生成分辞書
        config: ユーザー設定
        xp: バックエンドモジュール
    
    Returns:
        Z_pdn: PDNインピーダンス
    """
    
    # 寄生成分取得
    y_vrm = parasitic_elements['y_vrm']
    z_v = parasitic_elements['z_v']
    z_s = parasitic_elements['z_s']
    z_sN = parasitic_elements['z_sN']
    y_p = parasitic_elements['y_p']
    z_mntN = parasitic_elements['z_mntN']
    
    # VRMから開始（アドミタンス）
    y_total = y_vrm
    
    # 周波数点数から推定（parasitic_elementsはすべて同じ長さ）
    n_freq = len(y_vrm)
    
    # コンデンサラダーを逆順で構築（VRM側から測定ノード側へ）
    # 容量の大きい順（VRM側）から小さい順（測定ノード側）へ
    cap_names = list(capacitor_impedances.keys())
    
    for i in reversed(range(len(cap_names))):
        n_caps = count_vector[i]
        
        if n_caps > 0:
            # このコンデンサのインピーダンス
            z_c = capacitor_impedances[cap_names[i]]
            
            # マウントインダクタンスを追加
            # 各コンデンサ固有のL_mntがある場合はそれを使用
            cap_config = next((c for c in config.capacitors if c['name'] == cap_names[i]), None)
            if cap_config and cap_config.get('L_mnt') is not None:
                # omegaを取得
                omega = parasitic_elements.get('omega')
                if omega is not None:
                    z_mnt_custom = 1j * omega * cap_config['L_mnt']
                    z_cm = z_c + z_mnt_custom
                else:
                    # フォールバック：z_mntNから逆算
                    omega_L_mntN = xp.imag(z_mntN)
                    omega = omega_L_mntN / config.L_mntN if config.L_mntN > 0 else xp.zeros_like(z_c)
                    z_mnt_custom = 1j * omega * cap_config['L_mnt']
                    z_cm = z_c + z_mnt_custom
            else:
                z_cm = z_c + z_mntN
            
            # コンデンサのアドミタンス（並列接続）
            y_cm = n_caps / z_cm
            
            # spreading抵抗・インダクタンスを直列追加してから並列合成
            # Z -> Y変換: Y_total = Y_total + 1/(Z_sN + 1/Y_cm)
            z_series = z_sN + 1.0 / y_cm
            y_total = y_total + 1.0 / z_series
    
    # プレーナ容量を並列追加
    y_with_planar = y_total + y_p
    
    # spreading抵抗・インダクタンスを直列追加
    z_after_spreading = z_s + 1.0 / y_with_planar
    
    # via抵抗・インダクタンスを直列追加
    z_pdn = z_v + z_after_spreading
    
    return z_pdn


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
    n_batch = len(count_vectors)
    n_freq = len(f_grid)
    n_caps = len(capacitor_impedances)
    
    # 寄生成分を事前計算
    parasitic_elements = calculate_pdn_parasitic_elements(f_grid, config, xp)
    
    # 結果配列
    z_pdn_batch = xp.zeros((n_batch, n_freq), dtype=xp.complex64)
    
    # バッチ処理
    # GPU最適化: ベクトル化可能な部分を分離
    
    # コンデンサインピーダンスを配列に変換
    cap_names = list(capacitor_impedances.keys())
    z_cap_array = xp.stack([capacitor_impedances[name] for name in cap_names])  # (N_caps, N_freq)
    
    # マウントインダクタンス配列
    omega = parasitic_elements.get('omega', 2 * xp.pi * f_grid)
    z_mnt_array = xp.zeros_like(z_cap_array)
    for i, name in enumerate(cap_names):
        cap_config = next((c for c in config.capacitors if c['name'] == name), None)
        if cap_config and cap_config.get('L_mnt') is not None:
            z_mnt_array[i] = 1j * omega * cap_config['L_mnt']
        else:
            z_mnt_array[i] = parasitic_elements['z_mntN']
    
    # 各バッチで計算
    for batch_idx in range(n_batch):
        count_vec = count_vectors[batch_idx]
        
        # VRMアドミタンスから開始
        y_total = parasitic_elements['y_vrm'].copy()
        
        # コンデンサラダーの構築（ベクトル化）
        for cap_idx in reversed(range(n_caps)):
            n_caps_i = count_vec[cap_idx]
            
            if n_caps_i > 0:
                # インピーダンス取得
                z_c = z_cap_array[cap_idx]
                z_mnt = z_mnt_array[cap_idx]
                z_cm = z_c + z_mnt
                
                # 並列アドミタンス
                y_cm = n_caps_i / z_cm
                
                # spreading込みで並列合成
                z_series = parasitic_elements['z_sN'] + 1.0 / y_cm
                y_total = y_total + 1.0 / z_series
        
        # プレーナ容量
        y_with_planar = y_total + parasitic_elements['y_p']
        
        # spreading
        z_after_spreading = parasitic_elements['z_s'] + 1.0 / y_with_planar
        
        # via
        z_pdn = parasitic_elements['z_v'] + z_after_spreading
        
        z_pdn_batch[batch_idx] = z_pdn
    
    # 検証
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
    n_freq = len(f_grid)
    n_caps = len(capacitor_impedances)
    
    # 乱数生成器（CuPyとNumPyで分岐）
    if xp is np:
        rng = np.random.default_rng(config.seed)
    else:
        # CuPyの場合
        rng = xp.random.RandomState(config.seed)
    
    # 結果配列
    z_pdn_mc = xp.zeros((n_samples, n_freq), dtype=xp.complex64)
    
    # コンデンサインピーダンスを配列化
    cap_names = list(capacitor_impedances.keys())
    z_cap_base = xp.stack([capacitor_impedances[name] for name in cap_names])
    
    # 寄生成分（固定）
    parasitic_elements = calculate_pdn_parasitic_elements(f_grid, config, xp)
    omega = parasitic_elements.get('omega', 2 * xp.pi * f_grid)
    
    for sample_idx in range(n_samples):
        # コンデンサパラメータのばらつき生成
        # 容量: ±tol_C * (1 - mlcc_derating)
        # ESR: ±tol_ESR
        # ESL: ±tol_ESL
        
        # ばらつき係数（正規分布、3σで公差範囲）
        if xp is np:
            c_variation = rng.normal(1.0, config.tol_C/3, n_caps)
            esr_variation = rng.normal(1.0, config.tol_ESR/3, n_caps)
            esl_variation = rng.normal(1.0, config.tol_ESL/3, n_caps)
        else:
            # CuPyの場合
            c_variation = rng.normal(1.0, config.tol_C/3, n_caps, dtype=xp.float32)
            esr_variation = rng.normal(1.0, config.tol_ESR/3, n_caps, dtype=xp.float32)
            esl_variation = rng.normal(1.0, config.tol_ESL/3, n_caps, dtype=xp.float32)
        
        c_variation = xp.clip(c_variation, 1-config.tol_C, 1+config.tol_C)
        esr_variation = xp.clip(esr_variation, 1-config.tol_ESR, 1+config.tol_ESR)
        esl_variation = xp.clip(esl_variation, 1-config.tol_ESL, 1+config.tol_ESL)
        
        # MLCCディレーティングを適用
        c_variation = c_variation * (1 - config.mlcc_derating)
        
        # インピーダンスにばらつきを適用
        # Z = ESR + jωL - j/(ωC) の関係から
        omega = 2 * xp.pi * f_grid
        z_cap_varied = xp.zeros_like(z_cap_base)
        
        for cap_idx in range(n_caps):
            # 元のインピーダンスから各成分を推定（簡易的）
            z_orig = z_cap_base[cap_idx]
            
            # RLC成分の分離（近似）
            esr_orig = xp.real(z_orig).mean()  # 実部の平均をESRと仮定
            
            # 虚部から L と C を推定
            z_imag = xp.imag(z_orig)
            # 低周波で容量性、高周波で誘導性と仮定
            mid_idx = len(f_grid) // 2
            
            # ばらつきを適用（簡易モデル）
            esr_new = esr_orig * esr_variation[cap_idx]
            
            # 虚部にもばらつきを適用
            z_imag_new = z_imag * xp.sqrt(c_variation[cap_idx] * esl_variation[cap_idx])
            
            z_cap_varied[cap_idx] = esr_new + 1j * z_imag_new
        
        # PDN計算（ばらつきを含む）
        z_pdn_sample = assemble_pdn_ladder(
            count_vector,
            {name: z_cap_varied[i] for i, name in enumerate(cap_names)},
            capacitor_indices,
            parasitic_elements,
            config,
            xp
        )
        
        z_pdn_mc[sample_idx] = z_pdn_sample
    
    return z_pdn_mc