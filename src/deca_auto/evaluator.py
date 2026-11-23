"""
スコア評価（評価帯域内の |Z_pdn| と目標マスクの比較）、top_k抽出、Monte Carloロバスト評価

全成分は「小さいほど良い」非負値で、重み 1.0 が同程度の影響になるよう正規化している。
目標比は e = log10(|Z_pdn| / Z_target)（decade）で扱い、単位 U = log10(2)（≒6.02dB）で割る。

  max        : 帯域内の s の最大値。s は目標の1/2以下で0、目標ちょうどで1。目標より下は (e + U)/U、
               目標超過側は 1 + VIOLATION_GAIN·e/U（超過は同じdB数のマージン改善より VIOLATION_GAIN 倍重い）
  area       : 目標超過量 max(e, 0)/U の帯域平均（対数周波数で重み付け）
  mean       : s の帯域平均（目標の1/2以下に下げても改善しないため、過剰設計を評価しない）
  anti       : 反共振ピークの突出量（プロミネンス: 左右それぞれ自分より高い点までの最小値のうち
               高い方からの高さ）/U の合計。VRM-バルク間のような広い反共振も検出する
  resonance  : 直列共振の谷の深さ（同じ定義を上下反転）/U の合計
  flat       : e の加重標準偏差 / U
  parts      : 部品総数 / PARTS_REFERENCE（10個で1）
  num_types  : 種類数 / TYPES_REFERENCE（5種類で1）
               ※ max_total_parts や候補リストの長さで正規化すると、上限を大きくしただけで
                 1個あたりのコストが下がり部品を使い過ぎるため、固定の基準値で正規化する
  mc_worst   : Monte Carlo最悪サンプルにおける「インピーダンス成分の重み付き和」の公称値からの悪化量

総合スコア = Σ weight_<成分> × 成分
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deca_auto.config import UserConfig, WEIGHT_FIELDS
from deca_auto.utils import cp, transfer_to_host
from deca_auto.pdn import calculate_pdn_impedance_monte_carlo

# 成分名（WEIGHT_FIELDS と同順）
SCORE_COMPONENTS: Tuple[str, ...] = tuple(name.removeprefix("weight_") for name in WEIGHT_FIELDS)
# |Z_pdn| から決まる成分（Monte Carloで再評価する対象）
IMPEDANCE_COMPONENTS: Tuple[str, ...] = ("max", "area", "mean", "anti", "resonance", "flat")

UNIT_DECADE = float(np.log10(2.0))  # 正規化単位（6.02dB）
VIOLATION_GAIN = 4.0               # 目標超過側の s の傾き（目標割れは仕様違反なので強く罰する）
PEAK_MIN_DECADE = 0.5 / 20.0        # 0.5dB未満の突出は数値リップルとして無視
PARTS_REFERENCE = 10.0              # parts成分 = 部品総数 / 10
TYPES_REFERENCE = 5.0               # num_types成分 = 種類数 / 5
MC_BATCH_ELEMENTS = 1 << 24         # select_top_k でMCを一括評価するZ配列の要素数上限


_SCORE_KERNEL_SOURCE = r"""
#include <cupy/complex.cuh>

using Real = SCORE_REAL;
using Count = SCORE_COUNT;
constexpr int threads = 128;
constexpr int warps = threads / 32;

// Match the array operations' NaN propagation.
__device__ Real larger(Real a, Real b) { return a >= b || isnan(a) ? a : b; }
__device__ Real smaller(Real a, Real b) { return a <= b || isnan(a) ? a : b; }

__device__ Real warp_sum(Real value) {
    for (int step = 16; step > 0; step /= 2)
        value += __shfl_down_sync(0xffffffff, value, step);
    return value;
}

__device__ Real warp_max(Real value) {
    for (int step = 16; step > 0; step /= 2)
        value = larger(value, __shfl_down_sync(0xffffffff, value, step));
    return value;
}

// Topographic prominence: scan each side until a point beyond the extremum (or the band edge).
template<bool peak>
__device__ Real excursion(const Real* e, int i, int n, bool ignore_safe, Real threshold) {
    const Real center = e[i];
    Real left = e[i - 1];
    Real right = e[i + 1];
    for (int j = i - 2; j >= 0 && (peak ? !(e[j] > center) : !(e[j] < center)); --j)
        left = peak ? smaller(left, e[j]) : larger(left, e[j]);
    for (int j = i + 2; j < n && (peak ? !(e[j] > center) : !(e[j] < center)); ++j)
        right = peak ? smaller(right, e[j]) : larger(right, e[j]);
    const Real shoulder = peak ? larger(left, right) : smaller(left, right);
    const Real amount = peak ? center - shoulder : shoulder - center;
    return amount > threshold && (peak || !ignore_safe || shoulder > 0) ? amount : 0;
}

extern "C" __global__ void score_components(
    const complex<Real>* z, long long z_row_stride, long long z_col_stride,
    const long long* indices, const Real* target, const Real* weights,
    int n, int ignore_safe, Real unit, Real threshold, Real violation_gain,
    const Count* counts, int n_types, Real parts_ref, Real types_ref, Real* out)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    extern __shared__ Real e[];
    __shared__ Real partial[5][warps];
    __shared__ Real e_mean;
    z += row * z_row_stride;
    for (int i = tid; i < n; i += threads) {
        const Real magnitude = abs(z[indices[i] * z_col_stride]);
        e[i] = log10(larger(magnitude, Real(1e-30))) - target[i];
    }
    __syncthreads();

    Real maximum = 0, mean = 0, area = 0, center_sum = 0, anti = 0, resonance = 0;
    for (int i = tid; i < n; i += threads) {
        const Real center = e[i];
        const Real s = center > Real(0) ? Real(1) + violation_gain * center / unit
                                        : (larger(center, -unit) + unit) / unit;
        const Real w = weights[i];
        maximum = larger(maximum, s);
        mean += s * w;
        area += larger(center, Real(0)) * w;
        center_sum += center * w;
        if (i == 0 || i == n - 1) continue;
        if (center >= e[i - 1] && center > e[i + 1] && (!ignore_safe || center > 0))
            anti += excursion<true>(e, i, n, ignore_safe, threshold);
        if (center <= e[i - 1] && center < e[i + 1])
            resonance += excursion<false>(e, i, n, ignore_safe, threshold);
    }
    maximum = warp_max(maximum);
    mean = warp_sum(mean);
    area = warp_sum(area);
    center_sum = warp_sum(center_sum);
    anti = warp_sum(anti);
    resonance = warp_sum(resonance);
    __shared__ Real center_partial[warps];
    if (lane == 0) {
        partial[0][warp] = maximum;
        partial[1][warp] = mean;
        partial[2][warp] = area;
        partial[3][warp] = anti;
        partial[4][warp] = resonance;
        center_partial[warp] = center_sum;
    }
    __syncthreads();
    if (tid == 0) {
        e_mean = 0;
        for (int j = 0; j < warps; ++j) e_mean += center_partial[j];
    }
    __syncthreads();

    // Accumulate centered squares directly to retain accuracy for flat curves.
    Real variance = 0;
    for (int i = tid; i < n; i += threads) {
        const Real delta = e[i] - e_mean;
        variance += (delta * delta) * weights[i];
    }
    variance = warp_sum(variance);
    if (lane == 0) center_partial[warp] = variance;
    __syncthreads();

    if (tid == 0) {
        maximum = 0;
        mean = area = anti = resonance = variance = 0;
        for (int j = 0; j < warps; ++j) {
            maximum = larger(maximum, partial[0][j]);
            mean += partial[1][j];
            area += partial[2][j];
            anti += partial[3][j];
            resonance += partial[4][j];
            variance += center_partial[j];
        }
        long long parts = 0;
        int types = 0;
        for (int t = 0; t < n_types; ++t) {
            const Count count = counts[(long long)row * n_types + t];
            parts += count;
            types += count > 0;
        }
        // Component-major output gives each returned component a contiguous view.
        out[row] = Real(parts) / parts_ref;
        out[gridDim.x + row] = Real(types) / types_ref;
        out[2LL * gridDim.x + row] = maximum;
        out[3LL * gridDim.x + row] = mean;
        out[4LL * gridDim.x + row] = area / unit;
        out[5LL * gridDim.x + row] = sqrt(larger(variance, Real(0))) / unit;
        out[6LL * gridDim.x + row] = anti / unit;
        out[7LL * gridDim.x + row] = resonance / unit;
    }
}
"""
_SCORE_KERNELS: Dict[Tuple[np.dtype, np.dtype], Any] = {}
_GPU_SCORE_COMPONENTS = ('parts', 'num_types', 'max', 'mean', 'area', 'flat', 'anti', 'resonance')


def _gpu_score_components(z: Any, counts: Any, metadata: Dict[str, Any],
                          ignore_safe: bool) -> Optional[Dict[str, Any]]:
    """同じ実数精度の評価データを行ごとに融合評価する。共有メモリに収まらなければ従来経路を使う。"""
    n_eval = int(metadata['n_eval'])
    dtype = z.real.dtype
    shared_mem = n_eval * dtype.itemsize
    target, weights = metadata['log_target'], metadata['weights']
    if (z.dtype not in (np.complex64, np.complex128) or counts.dtype.kind != 'i'
            or counts.shape[0] != z.shape[0] or z.ndim != 2 or counts.ndim != 2
            or target.dtype != dtype or weights.dtype != dtype
            or shared_mem + 256 > 48 * 1024):
        return None

    count_dtype = np.dtype(np.int32 if counts.dtype.itemsize <= 4 else np.int64)
    key = (dtype, count_dtype)
    if key not in _SCORE_KERNELS:
        real = 'float' if dtype == np.float32 else 'double'
        count = 'int' if count_dtype == np.int32 else 'long long'
        source = _SCORE_KERNEL_SOURCE.replace('SCORE_REAL', real).replace('SCORE_COUNT', count)
        _SCORE_KERNELS[key] = cp.RawKernel(source, 'score_components')

    batch_size = z.shape[0]
    out = cp.empty((len(_GPU_SCORE_COMPONENTS), batch_size), dtype=dtype)
    if batch_size:
        _SCORE_KERNELS[key](
            (batch_size,), (128,),
            (z, *(np.int64(stride // z.itemsize) for stride in z.strides),
             cp.asarray(metadata['indices'], dtype=np.int64, order='C'),
             cp.asarray(target, order='C'), cp.asarray(weights, order='C'),
             np.int32(n_eval),
             np.int32(ignore_safe), dtype.type(UNIT_DECADE), dtype.type(PEAK_MIN_DECADE),
             dtype.type(VIOLATION_GAIN),
             cp.ascontiguousarray(counts, dtype=count_dtype), np.int32(counts.shape[1]),
             dtype.type(PARTS_REFERENCE), dtype.type(TYPES_REFERENCE), out),
            shared_mem=shared_mem,
        )
    return {name: out[i] for i, name in enumerate(_GPU_SCORE_COMPONENTS)}


def build_eval_metadata(f_grid: Any, eval_mask: Any, target_mask: Any, xp: Any = np) -> Dict[str, Any]:
    """
    評価帯域に関する事前計算（全チャンク共通）

    Returns:
        'indices'      : 評価帯域の周波数インデックス (n,)
        'log_target'   : log10(Z_target) (n,)
        'weights'      : 対数周波数の台形重み（合計1） (n,)
    """
    indices = xp.where(xp.asarray(eval_mask, dtype=bool))[0]
    n_eval = int(indices.size)
    f_eval = xp.asarray(f_grid)[indices].astype(np.float64)
    log_f = xp.log10(f_eval)
    if n_eval >= 2:
        dlog = xp.diff(log_f)
        weights = xp.zeros(n_eval, dtype=np.float64)
        weights[:-1] += 0.5 * dlog
        weights[1:] += 0.5 * dlog
    else:
        weights = xp.ones(max(n_eval, 1), dtype=np.float64)[:n_eval]
    weights = weights / max(float(weights.sum()), 1e-300)
    real_dtype = xp.asarray(f_grid).dtype
    return {
        'indices': indices,
        'n_eval': n_eval,
        'log_target': xp.log10(xp.maximum(xp.asarray(target_mask)[indices], 1e-30)).astype(real_dtype),
        'weights': weights.astype(real_dtype),
    }


def _extremum_excursion(e: Any, mask: Any, peak: bool, xp: Any) -> Any:
    """
    局所極値の突出量（地形学的プロミネンス、scipy.signal.peak_prominences と同じ定義）

    ピークは左右それぞれ「自分より高い点」または帯域端に当たるまでの最小値を求め、
    高い方の最小値（基準）からの高さを返す。谷は符号を反転して同様に深さを返す。
    極値の位置だけを左右に1点ずつ走査するため、計算量は 極値数 × 走査長。

    Returns:
        (B, n) 各極値位置の突出量（極値以外は0）
    """
    sign = 1.0 if peak else -1.0
    rows, cols = xp.nonzero(mask)
    out = xp.zeros(e.shape, dtype=e.dtype)
    if int(rows.size) == 0:
        return out
    n = e.shape[1]
    se = sign * e
    center = se[rows, cols]
    shoulders = []
    for step in (-1, 1):
        # 走査中の極値だけを残しながら1点ずつ進める（計算量は走査長の総和）
        base = xp.full(center.shape, np.inf, dtype=e.dtype)
        active = xp.arange(center.size)
        j = cols + step
        while int(active.size):
            inside = (j >= 0) & (j < n)
            active, j = active[inside], j[inside]
            value = se[rows[active], j]
            keep = ~(value > center[active])
            active, j, value = active[keep], j[keep], value[keep]
            base[active] = xp.minimum(base[active], value)
            j = j + step
        shoulders.append(base)
    out[rows, cols] = center - xp.maximum(shoulders[0], shoulders[1])
    return out
    n = e.shape[1]
    se = sign * e
    center = se[rows, cols]
    bases = []
    for step in (-1, 1):
        base = xp.full(center.shape, np.inf, dtype=e.dtype)
        alive = xp.ones(center.shape, dtype=bool)
        for d in range(1, n):
            j = cols + step * d
            alive &= (j >= 0) & (j < n)
            if not bool(alive.any()):
                break
            value = se[rows, xp.clip(j, 0, n - 1)]
            alive &= ~(value > center)
            base = xp.where(alive, xp.minimum(base, value), base)
        bases.append(base)
    out[rows, cols] = center - xp.maximum(bases[0], bases[1])
    return out


def calculate_score_components_batch(
    z_pdn_batch: Any,
    count_vectors: Any,
    eval_metadata: Dict[str, Any],
    config: UserConfig,
    xp: Any = np,
) -> Dict[str, Any]:
    """
    バッチのスコア成分を計算する（mc_worst を除く）。
    CuPyでは対応する精度と帯域幅を融合カーネルで評価し、それ以外は配列演算で評価する。

    Args:
        z_pdn_batch: (B, F) 複素 Z_pdn
        count_vectors: (B, T) カウントベクトル
        eval_metadata: build_eval_metadata の結果
        config: ユーザー設定
        xp: バックエンド

    Returns:
        {成分名: (B,) 実数配列}
    """
    z = xp.asarray(z_pdn_batch)
    if z.ndim == 1:
        z = z[None, :]
    counts = xp.asarray(count_vectors)
    if counts.ndim == 1:
        counts = counts[None, :]
    real_dtype = z.real.dtype
    n_batch = z.shape[0]
    n_eval = int(eval_metadata['n_eval'])

    if xp is cp and n_eval:
        fused = _gpu_score_components(z, counts, eval_metadata, bool(config.ignore_safe_anti_resonance))
        if fused is not None:
            return fused

    comps: Dict[str, Any] = {}

    total_parts = counts.sum(axis=1).astype(real_dtype)
    comps['parts'] = total_parts / PARTS_REFERENCE
    n_types = xp.count_nonzero(counts > 0, axis=1).astype(real_dtype)
    comps['num_types'] = n_types / TYPES_REFERENCE

    if n_eval == 0:
        for name in IMPEDANCE_COMPONENTS:
            comps[name] = xp.zeros(n_batch, dtype=real_dtype)
        return comps

    w = eval_metadata['weights']
    e = xp.log10(xp.maximum(xp.abs(z[:, eval_metadata['indices']]), 1e-30)) - eval_metadata['log_target']
    e = e.astype(real_dtype, copy=False)

    s = xp.where(e > 0, 1.0 + VIOLATION_GAIN * e / UNIT_DECADE,
                 (xp.maximum(e, -UNIT_DECADE) + UNIT_DECADE) / UNIT_DECADE)
    comps['max'] = s.max(axis=1)
    comps['mean'] = s @ w
    comps['area'] = xp.maximum(e, 0.0) @ w / UNIT_DECADE

    e_mean = e @ w
    comps['flat'] = xp.sqrt(xp.maximum(((e - e_mean[:, None]) ** 2) @ w, 0.0)) / UNIT_DECADE

    if n_eval >= 3:
        ignore_safe = bool(config.ignore_safe_anti_resonance)
        edge = xp.zeros((n_batch, 1), dtype=bool)
        center = e[:, 1:-1]

        # 反共振ピーク: 局所最大点のプロミネンス
        is_peak = xp.concatenate([edge, (center >= e[:, :-2]) & (center > e[:, 2:]), edge], axis=1)
        prominence = _extremum_excursion(e, is_peak, True, xp)
        valid = is_peak & (prominence > PEAK_MIN_DECADE)
        if ignore_safe:
            valid &= e > 0.0
        comps['anti'] = xp.where(valid, prominence, 0.0).sum(axis=1) / UNIT_DECADE

        # 共振の谷: 局所最小点の深さ（肩 = 谷の基準 = e + 深さ）
        is_dip = xp.concatenate([edge, (center <= e[:, :-2]) & (center < e[:, 2:]), edge], axis=1)
        depth = _extremum_excursion(e, is_dip, False, xp)
        valid = is_dip & (depth > PEAK_MIN_DECADE)
        if ignore_safe:
            valid &= (e + depth) > 0.0
        comps['resonance'] = xp.where(valid, depth, 0.0).sum(axis=1) / UNIT_DECADE
    else:
        comps['anti'] = xp.zeros(n_batch, dtype=real_dtype)
        comps['resonance'] = xp.zeros(n_batch, dtype=real_dtype)

    return {name: value.astype(real_dtype, copy=False) for name, value in comps.items()}


def weighted_sum(components: Dict[str, Any], config: UserConfig,
                 names: Tuple[str, ...] = SCORE_COMPONENTS) -> Any:
    """成分の重み付き和（存在しない成分は0扱い）"""
    total = None
    for name in names:
        if name not in components:
            continue
        term = float(getattr(config, f"weight_{name}")) * components[name]
        total = term if total is None else total + term
    return total


def evaluate_combinations(
    z_pdn_batch: Any,
    count_vectors: Any,
    eval_metadata: Dict[str, Any],
    config: UserConfig,
    xp: Any = np,
) -> Tuple[Any, Dict[str, Any]]:
    """
    バッチの公称スコアを計算する（小さいほど良い）

    Returns:
        (scores (B,), components {成分名: (B,)})
    """
    comps = calculate_score_components_batch(z_pdn_batch, count_vectors, eval_metadata, config, xp)
    return weighted_sum(comps, config), comps


def select_top_k(
    scores: Any,
    components: Dict[str, Any],
    count_vectors: Any,
    k: int,
    threshold: Optional[float],
    prepared: Dict[str, Any],
    eval_metadata: Dict[str, Any],
    config: UserConfig,
    xp: Any = np,
) -> Tuple[np.ndarray, Any, Any]:
    """
    総合スコア（公称スコア + weight_mc_worst × mc_worst）の上位k件を厳密に選ぶ

    mc_worst ≥ 0 なので総合スコア ≥ 公称スコア。公称スコアの昇順にMonte Carloを評価し、
    次の候補の公称スコアが「現時点のk位の総合スコア」以上になった時点で打ち切る
    （それ以降の候補はk位に入り得ない）。公称スコアの上位k件だけにMCを適用すると、
    MCで大きく悪化した候補がk位を占め、公称k+1位以下のより良い候補を取りこぼす。

    Args:
        scores: (B,) 公称スコア
        components: 公称成分 {成分名: (B,)}
        count_vectors: (B, T)
        threshold: グローバルtop_kのk位の総合スコア（これ以上は除外。満杯でなければNone）
    Returns:
        (インデックス (K,) ホストint64, 総合スコア (K,) xp, mc_worst (K,) xp)。総合スコアの昇順
    """
    n = int(scores.shape[0])
    k = min(int(k), n)
    bound = math.inf if threshold is None else float(threshold)
    use_mc = bool(config.mc_enable) and int(config.mc_samples) > 0
    n_valid = n if math.isinf(bound) else int((scores < bound).sum())
    if k <= 0 or n_valid == 0:
        return np.zeros(0, dtype=np.int64), scores[:0], scores[:0]

    order = xp.argsort(scores)
    if use_mc:
        # MCのZ配列 (候補 × サンプル × 周波数) が MC_BATCH_ELEMENTS を超えない範囲でまとめて評価
        per_combo = int(config.mc_samples) * int(prepared['omega'].shape[0])
        batch = max(k, MC_BATCH_ELEMENTS // max(per_combo, 1))
    else:
        batch = k
        n_valid = min(n_valid, k)
    w_mc = float(config.weight_mc_worst)

    sel_idx, sel_total, sel_mc = order[:0], scores[:0], scores[:0]
    for start in range(0, n_valid, batch):
        idx = order[start:min(start + batch, n_valid)]
        nominal = scores[idx]
        if math.isfinite(bound):
            count = int((nominal < bound).sum())   # 昇順なので先頭 count 件だけが候補
            if count == 0:
                break
            idx, nominal = idx[:count], nominal[:count]
        if use_mc:
            mc = monte_carlo_worst(count_vectors[idx], {name: components[name][idx] for name in IMPEDANCE_COMPONENTS},
                                   prepared, eval_metadata, config, xp)
        else:
            mc = xp.zeros_like(nominal)
        sel_idx = xp.concatenate([sel_idx, idx])
        sel_total = xp.concatenate([sel_total, nominal + w_mc * mc])
        sel_mc = xp.concatenate([sel_mc, mc])
        keep = xp.argsort(sel_total)[:k]
        sel_idx, sel_total, sel_mc = sel_idx[keep], sel_total[keep], sel_mc[keep]
        if int(sel_idx.shape[0]) == k:
            bound = min(bound, float(sel_total[-1]))

    if threshold is not None:
        count = int((sel_total < float(threshold)).sum())
        sel_idx, sel_total, sel_mc = sel_idx[:count], sel_total[:count], sel_mc[:count]
    return transfer_to_host(sel_idx).astype(np.int64), sel_total, sel_mc


def monte_carlo_worst(
    count_vectors: Any,
    nominal_components: Dict[str, Any],
    prepared: Dict[str, Any],
    eval_metadata: Dict[str, Any],
    config: UserConfig,
    xp: Any = np,
) -> Any:
    """
    Monte Carlo最悪悪化量（mc_worst成分）を計算する

    各候補について mc_samples 個の公差サンプルの Z_pdn を求め、インピーダンス成分の
    重み付き和の最大値と公称値の差（0未満は0）を返す。

    Args:
        count_vectors: (K, T) 候補
        nominal_components: 候補の公称成分 {成分名: (K,)}
        prepared: pdn.prepare_pdn_components の結果
    Returns:
        (K,) 実数配列
    """
    counts = xp.asarray(count_vectors)
    n_combo = int(counts.shape[0])
    n_samples = int(config.mc_samples)
    if not config.mc_enable or n_combo == 0 or n_samples <= 0:
        return xp.zeros(n_combo, dtype=prepared['omega'].dtype)

    z_mc = calculate_pdn_impedance_monte_carlo(counts, prepared, config, n_samples, xp)
    flat_counts = xp.repeat(counts, n_samples, axis=0)
    comps = calculate_score_components_batch(
        z_mc.reshape(n_combo * n_samples, -1), flat_counts, eval_metadata, config, xp
    )
    mc_imp = weighted_sum(comps, config, IMPEDANCE_COMPONENTS).reshape(n_combo, n_samples)
    nominal_imp = weighted_sum(
        {name: xp.asarray(nominal_components[name]) for name in IMPEDANCE_COMPONENTS},
        config,
        IMPEDANCE_COMPONENTS,
    )
    return xp.maximum(mc_imp.max(axis=1) - nominal_imp, 0.0)


def format_combination_name(count_vector: Any, capacitor_names: List[str]) -> str:
    """
    組み合わせを文字列表現にフォーマット

    Examples:
        >>> format_combination_name(np.array([2, 0, 3, 1]), ["C1", "C2", "C3", "C4"])
        '(C1)x2 + (C3)x3 + (C4)'
        >>> format_combination_name(np.array([0, 0]), ["C1", "C2"])
        'Empty'
    """
    parts = []
    for name, count in zip(capacitor_names, np.asarray(count_vector).tolist()):
        count = int(count)
        if count == 1:
            parts.append(f"({name})")
        elif count > 1:
            parts.append(f"({name})x{count}")
    return " + ".join(parts) if parts else "Empty"
