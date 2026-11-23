"""
PDNインピーダンス計算（ラダー合成、Monte Carlo）

回路（images/PDN.jpg）を測定ノード（Load側）から見た順:

    Load ─ Z_v ─ Z_s ─┬─ Z_s1 ─┬─ Z_s2 ─┬─ … ─ Z_sN ─┬─ VRM
                      Y_p      Y_cm1    Y_cm2          Y_cmN // Y_vrm

- Z_v = R_v + jωL_v（BGAビア）、Z_s = R_s + jωL_s（Load直前のspreading）
- Y_p = ωC_p(tanδ + j) / (1 + R_p·ωC_p(tanδ + j))（損失を持つプレーナ容量）
- Z_sk = R_sN + jωL_sN（コンデンサ1個ごとのspreading、直列）
- Y_cmk = 1 / (Z_ck + jωL_mnt,k)（コンデンサ1個 + マウントインダクタンス、並列）
- Y_vrm = 1/(R_vrm + jωL_vrm) + Y_p（理想電源は短絡、寄生分のみアドミタンスとして残す）

コンデンサは1個ごとに1段（Z_sk, Y_cmk）を占め、Load側から容量の小さい順に並ぶ。
同じ種類をn個使う組み合わせは同一の段がn段連続する。
測定ノードに1Aを注入したときのポート電圧が Z_pdn となる。
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from deca_auto.config import UserConfig
from deca_auto.utils import transfer_to_device, transfer_to_host


def _as_complex_dtype(dtype: Any) -> np.dtype:
    target = np.dtype(dtype)
    if target.kind != 'c':
        raise ValueError(f"複素数dtypeが必要です: {target}")
    return target


def calculate_pdn_parasitic_elements(
    f_grid: Any,
    config: UserConfig,
    xp: Any = np,
    dtype: Any = np.complex64,
) -> Dict[str, Any]:
    """
    PDN寄生成分（周波数のみに依存する要素）を計算する

    Returns:
        'z_v', 'z_s', 'z_sN', 'y_p', 'y_vrm'（複素 (F,)）と 'omega'（実数 (F,)）
    """
    target = _as_complex_dtype(dtype)
    # 寄生成分は float64 で計算してから目標精度へ落とす（小さいR/Lでの丸め誤差を避ける）
    omega = 2.0 * np.pi * xp.asarray(f_grid, dtype=np.float64)
    jw = 1j * omega

    z_v = config.R_v + jw * config.L_v
    z_s = config.R_s + jw * config.L_s
    z_sN = config.R_sN + jw * config.L_sN

    planar = omega * config.C_p * (config.tan_delta_p + 1j)
    y_p = planar / (1.0 + config.R_p * planar)

    z_vrm = config.R_vrm + jw * config.L_vrm
    # R_vrm = L_vrm = 0（理想電源直結）の場合でも有限値に留める
    z_vrm = xp.where(xp.abs(z_vrm) > 0, z_vrm, 1e-30)
    y_vrm = 1.0 / z_vrm + y_p

    return {
        'z_v': z_v.astype(target),
        'z_s': z_s.astype(target),
        'z_sN': z_sN.astype(target),
        'y_p': y_p.astype(target),
        'y_vrm': y_vrm.astype(target),
        'omega': omega.astype(np.finfo(target).dtype),
    }


def prepare_pdn_components(
    capacitor_impedances: Dict[str, Any],
    config: UserConfig,
    f_grid: Any,
    xp: Any = np,
    dtype: Any = None,
    rlc_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    PDN計算で全チャンク共通に再利用する配列を事前計算する

    Args:
        capacitor_impedances: {name: Z_c}（容量の小さい順。Z_cはマウント寄生を含まない部品単体）
        config: ユーザー設定（コンデンサ別 L_mnt、既定 L_mntN、寄生成分）
        f_grid: 周波数グリッド（xp配列）
        xp: バックエンド
        dtype: 複素dtype（既定は config.dtype_c）
        rlc_params: {name: {'C': 容量, 'ESL': 等価ESL}}（Monte Carloの公差適用に使用）

    Returns:
        'cap_names', 'z_c'(T,F), 'z_mnt'(T,F), 'y_cm'(T,F), 'parasitic', 'omega',
        'dtype', 'C_nom'(T,), 'ESL_nom'(T,)
    """
    target = _as_complex_dtype(dtype if dtype is not None else config.dtype_c)
    cap_names = tuple(capacitor_impedances.keys())
    if not cap_names:
        raise ValueError("コンデンサインピーダンス辞書が空です")

    parasitic = calculate_pdn_parasitic_elements(f_grid, config, xp, target)
    omega = parasitic['omega']

    z_c = xp.stack([xp.asarray(capacitor_impedances[name]) for name in cap_names]).astype(target)

    cap_cfg = {cap.get('name'): cap for cap in config.capacitors}
    default_l_mnt = float(config.L_mntN)
    l_mnt = []
    for name in cap_names:
        value = cap_cfg.get(name, {}).get('L_mnt')
        l_mnt.append(default_l_mnt if value is None else float(value))
    l_mnt_arr = xp.asarray(l_mnt, dtype=omega.dtype)
    z_mnt = (1j * omega[None, :] * l_mnt_arr[:, None]).astype(target)

    y_cm = (1.0 / (z_c + z_mnt)).astype(target)

    rlc_params = rlc_params or {}
    c_nom = [float(rlc_params.get(name, {}).get('C', 0.0) or 0.0) for name in cap_names]
    esl_nom = [float(rlc_params.get(name, {}).get('ESL', 0.0) or 0.0) for name in cap_names]

    return {
        'cap_names': cap_names,
        'z_c': z_c,
        'z_mnt': z_mnt,
        'y_cm': y_cm,
        'parasitic': parasitic,
        'omega': omega,
        'dtype': target,
        'C_nom': xp.asarray(c_nom, dtype=omega.dtype),
        'ESL_nom': xp.asarray(esl_nom, dtype=omega.dtype),
    }


def _ladder_impedance(
    count_vectors: Any,
    stage_admittance: Callable[[int, Any], Any],
    parasitic: Dict[str, Any],
    xp: Any,
) -> Any:
    """
    ラダー回路をVRM側から測定ノード側へ畳み込んで Z_pdn を求める

    Args:
        count_vectors: (B, T) 各種類の使用数（列は容量の小さい順）
        stage_admittance: (k, types_k) -> (B, F) のY_cm。k はVRM側から数えた段番号、
            types_k は各行の k 段目の種類インデックス（未使用段は -1 ではなく 0 にクリップ済み）
        parasitic: calculate_pdn_parasitic_elements の結果
        xp: バックエンド

    Returns:
        Z_pdn (B, F)
    """
    counts = xp.asarray(count_vectors).astype(np.int32, copy=False)
    if counts.ndim == 1:
        counts = counts[None, :]
    n_batch = counts.shape[0]

    # suffix[:, t] = 種類t以降（容量の大きい側）の総数。VRM側から k 段目の種類は suffix[:, t] > k を満たす最大の t
    suffix = xp.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
    n_stages = suffix[:, 0]
    max_stages = int(n_stages.max()) if n_batch > 0 else 0

    y_vrm = parasitic['y_vrm']
    z_sN = parasitic['z_sN']
    y = xp.empty((n_batch, y_vrm.shape[0]), dtype=y_vrm.dtype)
    y[...] = y_vrm

    for k in range(max_stages):
        types_k = (suffix > k).sum(axis=1) - 1
        active = (n_stages > k)[:, None]
        types_k = xp.maximum(types_k, 0)
        # 段k: ノードに Y_cm を並列接続し、Load側へ Z_sN を直列に挟む
        y_stage = 1.0 / (z_sN + 1.0 / (y + stage_admittance(k, types_k)))
        y = xp.where(active, y_stage, y)

    z = parasitic['z_s'] + 1.0 / (y + parasitic['y_p'])
    return parasitic['z_v'] + z


_LADDER_KERNEL_SOURCE = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void pdn_ladder_{name}(
    const int* counts, const long long n_rows, const int n_types, const int n_freq,
    const complex<{real}>* y_cm, const complex<{real}>* z_sN, const complex<{real}>* y_vrm,
    const complex<{real}>* y_p, const complex<{real}>* z_s, const complex<{real}>* z_v,
    complex<{real}>* out)
{{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * n_freq) return;
    const long long b = idx / n_freq;
    const int f = (int)(idx - b * n_freq);
    const int* row = counts + b * n_types;
    const complex<{real}> one(1, 0);
    const complex<{real}> zsn = z_sN[f];
    complex<{real}> y = y_vrm[f];
    // VRM側（容量の大きい種類）から1個ずつ段を畳み込む
    for (int t = n_types - 1; t >= 0; --t) {{
        const int n = row[t];
        if (n <= 0) continue;
        const complex<{real}> ycm = y_cm[(long long)t * n_freq + f];
        for (int j = 0; j < n; ++j) {{
            y = one / (zsn + one / (y + ycm));
        }}
    }}
    out[idx] = z_v[f] + z_s[f] + one / (y + y_p[f]);
}}
"""
_LADDER_KERNELS: Dict[str, Any] = {}


def _ladder_kernel(dtype: np.dtype) -> Any:
    """CuPy用の融合ラダーカーネル（dtypeごとにコンパイルしてキャッシュ）"""
    import cupy as cp

    name = 'c64' if np.dtype(dtype) == np.complex64 else 'c128'
    if name not in _LADDER_KERNELS:
        real = 'float' if name == 'c64' else 'double'
        source = _LADDER_KERNEL_SOURCE.format(name=name, real=real)
        _LADDER_KERNELS[name] = cp.RawKernel(source, f'pdn_ladder_{name}')
    return _LADDER_KERNELS[name]


def calculate_pdn_impedance_batch(
    count_vectors: Any,
    prepared: Dict[str, Any],
    xp: Any = np,
) -> Any:
    """
    組み合わせバッチの Z_pdn を計算する

    CuPyでは1スレッドが（組み合わせ, 周波数）1点のラダー全体を計算する融合カーネルを使い、
    中間配列の生成とメモリ往復を省く。NumPyでは段ごとのベクトル演算（_ladder_impedance）。

    Args:
        count_vectors: (B, T) カウントベクトル（xp配列）
        prepared: prepare_pdn_components の結果
        xp: バックエンド

    Returns:
        Z_pdn (B, F)
    """
    y_cm = prepared['y_cm']
    if xp is np:
        return _ladder_impedance(
            count_vectors,
            lambda _k, types_k: y_cm[types_k],
            prepared['parasitic'],
            xp,
        )

    parasitic = prepared['parasitic']
    dtype = prepared['dtype']
    counts = xp.ascontiguousarray(xp.asarray(count_vectors, dtype=np.int32))
    if counts.ndim == 1:
        counts = counts[None, :]
    n_rows, n_types = counts.shape
    n_freq = int(y_cm.shape[1])
    out = xp.empty((n_rows, n_freq), dtype=dtype)
    total = n_rows * n_freq
    if total == 0:
        return out
    threads = 256
    args = [xp.ascontiguousarray(parasitic[key], dtype=dtype)
            for key in ('z_sN', 'y_vrm', 'y_p', 'z_s', 'z_v')]
    _ladder_kernel(dtype)(
        ((total + threads - 1) // threads,), (threads,),
        (counts, np.int64(n_rows), np.int32(n_types), np.int32(n_freq),
         xp.ascontiguousarray(y_cm, dtype=dtype), *args, out),
    )
    return out


def _tolerance_factors(counts_host: np.ndarray, n_samples: int, config: UserConfig,
                       seed: int) -> np.ndarray:
    """
    公差倍率 (3, max_stages, K*n_samples) = (k_C, k_R, k_L) をホストで生成する

    乱数は組み合わせごとに (seed, カウントベクトル) から初期化するため、バックエンド
    （NumPy/CuPy）、チャンク分割、評価順のシャッフルに依らず同じ組み合わせは同じ結果になる。
    各倍率は平均1、3σ=公差の正規分布を ±公差 でクリップしたもの。k_C にはディレーティングを含む。
    """
    tolerances = (float(config.tol_C), float(config.tol_ESR), float(config.tol_ESL))
    n_combo = counts_host.shape[0]
    n_stages = counts_host.sum(axis=1)
    max_stages = int(n_stages.max()) if n_combo else 0
    factors = np.ones((3, max_stages, n_combo * n_samples), dtype=np.float64)
    for i, row in enumerate(counts_host):
        rng = np.random.default_rng([int(seed), *(int(v) for v in row)])
        cols = slice(i * n_samples, (i + 1) * n_samples)
        for j, tol in enumerate(tolerances):
            if tol > 0:
                values = rng.normal(1.0, tol / 3.0, size=(int(n_stages[i]), n_samples))
                factors[j, :n_stages[i], cols] = np.clip(values, 1.0 - tol, 1.0 + tol)
    factors[0] *= 1.0 - float(config.mlcc_derating)
    return factors


def calculate_pdn_impedance_monte_carlo(
    count_vectors: Any,
    prepared: Dict[str, Any],
    config: UserConfig,
    n_samples: int,
    xp: Any = np,
    seed: Optional[int] = None,
) -> Any:
    """
    部品公差を考慮した Z_pdn を Monte Carlo で計算する

    各コンデンサ（段）ごとに独立に C/ESR/ESL の倍率をサンプリングし、
    Z_c' = k_R·Re(Z_c) + j·(Im(Z_c) + ωESL(k_L − 1) − (1/(ωC))·(1/(k_C(1 − derating)) − 1))
    として部品単体のインピーダンスを摂動させる（ESL, C は prepared の等価値）。

    Args:
        count_vectors: (K, T) 評価する組み合わせ
        prepared: prepare_pdn_components の結果
        config: ユーザー設定（tol_C, tol_ESR, tol_ESL, mlcc_derating, seed）
        n_samples: 1組み合わせあたりのサンプル数
        xp: バックエンド
        seed: 乱数シード（既定は config.seed）

    Returns:
        Z_pdn (K, n_samples, F)
    """
    counts = xp.asarray(count_vectors).astype(np.int32, copy=False)
    if counts.ndim == 1:
        counts = counts[None, :]
    n_combo = counts.shape[0]
    n_freq = prepared['omega'].shape[0]
    if n_samples <= 0 or n_combo == 0:
        return xp.zeros((n_combo, max(n_samples, 0), n_freq), dtype=prepared['dtype'])

    omega = prepared['omega']
    real_dtype = omega.dtype
    z_c = prepared['z_c']
    z_mnt = prepared['z_mnt']
    esr_part = z_c.real
    x_part = z_c.imag
    inductive = omega[None, :] * prepared['ESL_nom'][:, None]                       # ωESL (T,F)
    c_nom = prepared['C_nom']
    capacitive = xp.where(
        c_nom[:, None] > 0,
        1.0 / (omega[None, :] * xp.where(c_nom > 0, c_nom, 1.0)[:, None]),
        0.0,
    )                                                                                  # 1/(ωC) (T,F)

    counts_host = transfer_to_host(counts)
    counts_rep = xp.repeat(counts, n_samples, axis=0)
    factors = _tolerance_factors(counts_host, n_samples, config, config.seed if seed is None else seed)
    k_c, k_r, k_l = transfer_to_device(factors.astype(real_dtype), xp)

    def stage_admittance(k: int, types_k: Any) -> Any:
        kc = k_c[k][:, None]
        kr = k_r[k][:, None]
        kl = k_l[k][:, None]
        x = (
            x_part[types_k]
            + inductive[types_k] * (kl - 1.0)
            - capacitive[types_k] * (1.0 / kc - 1.0)
        )
        z = kr * esr_part[types_k] + 1j * x + z_mnt[types_k]
        return 1.0 / z

    z_pdn = _ladder_impedance(counts_rep, stage_admittance, prepared['parasitic'], xp)
    return z_pdn.astype(prepared['dtype'], copy=False).reshape(n_combo, n_samples, n_freq)
