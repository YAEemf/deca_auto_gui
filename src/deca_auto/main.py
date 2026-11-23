import math
import traceback
from typing import Optional, Dict, List, Any, Callable, Iterable
import numpy as np
from dataclasses import dataclass, field

from deca_auto.config import UserConfig, validate_config, resolve_capacitor_usage_bounds
from deca_auto.utils import (
    logger, get_backend, get_gpu_info, get_vram_budget,
    generate_frequency_grid, create_evaluation_mask, create_target_mask,
    Timer, memory_cleanup, get_progress_bar, transfer_to_device,
    transfer_to_host, get_dtype, ensure_numpy, to_float,
    get_custom_mask_freq_range,
    profile_block, increment_metric, update_metric_max, get_memory_snapshot,
    log_metrics,
)
from deca_auto.capacitor import (
    calculate_all_capacitor_impedances,
    estimate_capacitance_from_impedance,
    estimate_rlc_parameters,
)
from deca_auto.pdn import calculate_pdn_impedance_batch, prepare_pdn_components
from deca_auto.evaluator import (
    SCORE_COMPONENTS, build_eval_metadata, evaluate_combinations,
    extract_top_k, monte_carlo_worst,
)
from deca_auto.excel_out import export_to_excel

@dataclass
class OptimizationContext:
    """最適化実行コンテキスト（全チャンク共通の事前計算とGUI連携）"""
    config: UserConfig
    xp: Any  # NumPy or CuPy
    is_gpu: bool
    pdn_assets: Dict[str, Any]
    eval_metadata: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    stop_flag: bool = False
    stop_event: Optional[Any] = None

    def should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            self.stop_flag = True
        return self.stop_flag


class CombinationSpace:
    """
    使用数の組み合わせ（カウントベクトル）の全体集合

    各種類 t の使用数 c_t は [min_counts[t], max_counts[t]]、総数は [min_total, max_total]。
    DP表 ways[t][a]（種類 t 以降を埋めて総数が範囲に入る方法の数、a は種類 t 未満の累計）を
    用いて、順位 0..total-1 を辞書順の組み合わせへ直接変換（アンランク）する。
    これにより組み合わせをバックエンド（GPU可）上でチャンク単位に生成でき、
    全組み合わせを保持するバッファが不要になる。
    """

    def __init__(self, min_counts: np.ndarray, max_counts: np.ndarray,
                 min_total: int, max_total: int) -> None:
        self.min_counts = np.asarray(min_counts, dtype=np.int64)
        self.max_counts = np.asarray(max_counts, dtype=np.int64)
        self.min_total = int(min_total)
        self.max_total = int(max_total)
        n_types = self.min_counts.size

        ways = [[0] * (self.max_total + 1) for _ in range(n_types + 1)]
        for acc in range(self.max_total + 1):
            ways[n_types][acc] = int(self.min_total <= acc <= self.max_total)
        for t in range(n_types - 1, -1, -1):
            lo, hi = int(self.min_counts[t]), int(self.max_counts[t])
            for acc in range(self.max_total + 1):
                ways[t][acc] = sum(ways[t + 1][acc + v] for v in range(lo, min(hi, self.max_total - acc) + 1))
        self.total = ways[0][0] if n_types > 0 else 0
        if self.total >= 2 ** 62:
            raise ValueError(f"組み合わせ数が多すぎます: {self.total}")
        self._ways = np.asarray(ways, dtype=np.int64)

    def unrank(self, ranks: Any, xp: Any) -> Any:
        """有効な順位 (B,) を辞書順のカウントベクトル (B, T) int32 に変換する。"""
        if xp is not np:
            from deca_auto.utils import cp

            if xp is cp:
                r = cp.asarray(ranks, dtype=np.int64, order='C')
                n_types = self.min_counts.size
                out = cp.empty((r.shape[0], n_types), dtype=np.int32)
                if not out.size:
                    return out

                kernel = getattr(CombinationSpace.unrank, '_gpu_kernel', None)
                if kernel is None:
                    kernel = cp.RawKernel(r"""
extern "C" __global__ void unrank(
    const long long* ranks, long long n_rows, int n_types, long long max_total,
    const long long* ways, const long long* lower, const long long* upper, int* out)
{
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long rank = ranks[row];
    long long acc = 0;
    const long long stride = max_total + 1;
    out += row * n_types;
    if (rank < 0 || rank >= ways[0]) {
        for (int t = 0; t < n_types; ++t) out[t] = -1;
        return;
    }
    for (int t = 0; t < n_types; ++t) {
        ways += stride;
        const long long limit = min(upper[t], max_total - acc);
        long long chosen = -1;
        // Skip each lexicographic group using the number of possible suffixes.
        for (long long v = lower[t]; v <= limit; ++v) {
            const long long n = ways[acc + v];
            if (rank < n) {
                chosen = v;
                break;
            }
            rank -= n;
        }
        out[t] = (int)chosen;
        acc += chosen;
    }
}
""", 'unrank')
                    CombinationSpace.unrank._gpu_kernel = kernel

                threads = 128
                kernel(
                    ((r.shape[0] + threads - 1) // threads,), (threads,),
                    (r, np.int64(r.shape[0]), np.int32(n_types), np.int64(self.max_total),
                     cp.asarray(self._ways), cp.asarray(self.min_counts), cp.asarray(self.max_counts), out),
                )
                return out

        ways = xp.asarray(self._ways)
        r = xp.asarray(ranks, dtype=np.int64).copy()
        n_types = self.min_counts.size
        out = xp.empty((r.shape[0], n_types), dtype=np.int32)
        acc = xp.zeros(r.shape[0], dtype=np.int64)
        for t in range(n_types):
            chosen = xp.full(r.shape[0], -1, dtype=np.int64)
            for v in range(int(self.min_counts[t]), int(self.max_counts[t]) + 1):
                nxt = acc + v
                n = xp.where(nxt <= self.max_total, ways[t + 1][xp.minimum(nxt, self.max_total)], 0)
                pending = chosen < 0
                chosen = xp.where(pending & (r < n), v, chosen)
                r = xp.where(pending & (chosen < 0), r - n, r)
            out[:, t] = chosen
            acc += chosen
        return out

    def iter_chunks(self, chunk_size: int, xp: Any, shuffle: bool = False,
                    seed: Optional[int] = None) -> Iterable[Any]:
        """
        チャンクごとのカウントベクトル (B, T) を返す

        shuffle=True では順位を全単射 r → (a·r + b) mod total で並べ替え、
        組み合わせ全体から満遍なく評価する（a は total と互いに素、O(1)メモリ）。
        """
        total = self.total
        chunk_size = max(1, int(chunk_size))
        mul, add = 1, 0
        if shuffle and total > 2:
            rng = np.random.default_rng(seed)
            # a·(total−1) が int64 に収まる範囲で、total の黄金比付近の値を使う
            limit = max(2, min(total - 1, (2 ** 63 - 1) // total))
            mul = max(2, min(limit, int(total * 0.6180339887) + int(rng.integers(0, 1024))))
            while math.gcd(mul, total) != 1:
                mul -= 1 if mul > 2 else -1
            add = int(rng.integers(0, total))
        for start in range(0, total, chunk_size):
            ranks = xp.arange(start, min(start + chunk_size, total), dtype=np.int64)
            if mul != 1 or add != 0:
                ranks = (ranks * mul + add) % total
            yield self.unrank(ranks, xp)


def build_combination_space(
    num_capacitors: int,
    max_total: int,
    min_total: int,
    min_counts: Optional[np.ndarray] = None,
    max_counts: Optional[np.ndarray] = None,
) -> CombinationSpace:
    """
    使用数の制約から組み合わせ空間を作る

    Args:
        num_capacitors: コンデンサ種類数
        max_total / min_total: 総数の上限/下限
        min_counts / max_counts: 各種類の使用数の下限/上限（既定 0 / max_total）
    """
    if max_total < 0 or min_total < 0:
        raise ValueError("max_totalおよびmin_totalは非負である必要があります")
    min_counts = np.zeros(num_capacitors, dtype=np.int64) if min_counts is None else np.asarray(min_counts)
    max_counts = np.full(num_capacitors, max_total) if max_counts is None else np.asarray(max_counts)
    if min_counts.shape[0] != num_capacitors or max_counts.shape[0] != num_capacitors:
        raise ValueError("min_counts と max_counts の長さは num_capacitors に一致する必要があります")
    min_counts = np.maximum(min_counts, 0)
    max_counts = np.minimum(max_counts, max_total)

    lo = max(min_total, int(min_counts.sum()))
    hi = min(max_total, int(max_counts.sum()))
    if lo > hi or np.any(min_counts > max_counts):
        logger.warning("指定された使用数の範囲では有効な組み合わせが存在しません")
    logger.info(
        f"組み合わせ生成: {num_capacitors}種類, 総数{lo}～{hi}個, "
        f"個別範囲min={min_counts.tolist()}, max={max_counts.tolist()}"
    )
    return CombinationSpace(min_counts, np.maximum(max_counts, min_counts), lo, max(hi, 0))


def process_chunk(ctx: OptimizationContext,
                  count_vectors_chunk: np.ndarray,
                  threshold: Optional[float]) -> List[Dict]:
    """
    チャンクの処理（PDN計算 → 公称スコア → チャンク内top_k → Monte Carlo）

    Args:
        ctx: 最適化コンテキスト
        count_vectors_chunk: (B, T) カウントベクトル（xp配列）
        threshold: 現在のグローバルtop_kの最下位スコア（未満の候補のみ残す。満杯でなければNone）

    Returns:
        候補のリスト（ホスト配列）。各要素は count_vector, z_pdn, total_score, components
    """
    xp = ctx.xp
    config = ctx.config
    counts = count_vectors_chunk

    increment_metric(ctx.metrics, "chunk_received", 1)
    update_metric_max(ctx.metrics, "chunk_size_max", len(count_vectors_chunk))

    with profile_block(ctx.metrics, "pdn_impedance_batch"):
        z_pdn = calculate_pdn_impedance_batch(counts, ctx.pdn_assets, xp)

    with profile_block(ctx.metrics, "combination_evaluation"):
        scores, comps = evaluate_combinations(z_pdn, counts, ctx.eval_metadata, config, xp)

    with profile_block(ctx.metrics, "topk_extraction"):
        # 総合スコア = 公称スコア + w_mc × (MC悪化量 ≥ 0) なので、公称スコアで足切りできる
        top_idx = extract_top_k(scores, config.top_k, xp, threshold)
    if top_idx.size == 0:
        return []

    idx = transfer_to_device(top_idx, xp) if ctx.is_gpu else top_idx
    top_counts = counts[idx]
    top_comps = {name: value[idx] for name, value in comps.items()}
    totals = scores[idx]

    if config.mc_enable and config.mc_samples > 0:
        with profile_block(ctx.metrics, "monte_carlo_evaluation"):
            mc = monte_carlo_worst(top_counts, top_comps, ctx.pdn_assets, ctx.eval_metadata, config, xp)
        top_comps['mc_worst'] = mc
        totals = totals + float(config.weight_mc_worst) * mc
    else:
        top_comps['mc_worst'] = xp.zeros_like(totals)

    # ホストへ一括転送
    counts_host = transfer_to_host(top_counts)
    z_host = transfer_to_host(z_pdn[idx])
    totals_host = transfer_to_host(totals)
    comps_host = {name: transfer_to_host(top_comps[name]) for name in SCORE_COMPONENTS}

    if ctx.is_gpu:
        for name, value in get_memory_snapshot(xp).items():
            update_metric_max(ctx.metrics, f"gpu_{name}_max", value)
    increment_metric(ctx.metrics, "chunk_processed", 1)

    return [
        {
            'count_vector': counts_host[i],
            'z_pdn': z_host[i],
            'total_score': float(totals_host[i]),
            'components': {name: float(comps_host[name][i]) for name in SCORE_COMPONENTS},
        }
        for i in range(len(top_idx))
    ]


def update_global_top_k(global_top_k: List[Dict], candidates: List[Dict], k: int) -> List[Dict]:
    """グローバルtop_kに候補を統合し、順位を再採番する"""
    merged = sorted(global_top_k + candidates, key=lambda r: r['total_score'])[:k]
    for rank, result in enumerate(merged, start=1):
        result['rank'] = rank
    return merged


def _capacitor_rlc_params(config: UserConfig, names: List[str],
                          impedances: Dict[str, Any], f_grid_host: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    各コンデンサの等価 C / ESL（容量ソートとMonte Carlo公差の基準）

    RLCモデル（SPICEパス無し）は設定値、SPICE/VFモデルは Z_c から推定する。
    """
    cap_cfg = {cap.get('name'): cap for cap in config.capacitors}
    params: Dict[str, Dict[str, float]] = {}
    for name in names:
        cfg = cap_cfg.get(name, {})
        z_host = ensure_numpy(impedances[name])
        if not cfg.get('path'):
            c_val = to_float(cfg.get('C'), None)
            esl_val = to_float(cfg.get('ESL'), None)
            if c_val is None or esl_val is None:
                _, esl_est, c_est = estimate_rlc_parameters(z_host, f_grid_host)
                c_val = c_est if c_val is None else c_val
                esl_val = esl_est if esl_val is None else esl_val
        else:
            _, esl_val, c_val = estimate_rlc_parameters(z_host, f_grid_host)
            if to_float(cfg.get('C'), None):
                c_val = to_float(cfg.get('C'), c_val)
        if not c_val or c_val <= 0:
            c_val = estimate_capacitance_from_impedance(z_host, f_grid_host, method='median')
        params[name] = {'C': float(c_val), 'ESL': float(max(esl_val or 0.0, 0.0))}
    return params


def run_optimization(config: UserConfig,
                     gui_callback: Optional[Callable] = None,
                     stop_event: Optional[Any] = None) -> Dict:
    """
    最適化処理のメイン実行関数

    Args:
        config: ユーザー設定
        gui_callback: GUI更新用コールバック関数
        stop_event: 停止要求（threading.Event）

    Returns:
        最適化結果（ホスト配列のみ）
    """
    if not validate_config(config):
        raise ValueError("設定が無効です")

    xp, backend_name, is_gpu = get_backend(force_numpy=config.force_numpy, cuda_device=config.cuda)
    gpu_info = get_gpu_info(config.cuda) if is_gpu else None
    if gpu_info:
        logger.info(f"選択中GPU: {gpu_info['name']} \n"
                    f"Total: {gpu_info['total_memory_gb']:.2f}GB, "
                    f"Free: {gpu_info['free_memory_gb']:.2f}GB")

    dtype_c = np.dtype(get_dtype(config.dtype_c))
    dtype_r = np.dtype(get_dtype(config.dtype_r))

    vram_budget, _ = get_vram_budget(config.max_vram_ratio_limit, config.cuda)
    logger.info(f"メモリバジェット: {vram_budget / 1024**3:.2f}GB")

    with Timer("周波数グリッド生成"):
        f_grid = generate_frequency_grid(config.f_start, config.f_stop,
                                         config.num_points_per_decade, xp, dtype_r)
        logger.info(f"周波数点数: {len(f_grid)}")
    f_grid_host = transfer_to_host(f_grid)

    # 評価帯域（カスタムマスクモードではマスクの定義帯域を使用）
    eval_f_L, eval_f_H = config.f_L, config.f_H
    if config.target_impedance_mode == "custom" and config.z_custom_mask:
        custom_f_L, custom_f_H = get_custom_mask_freq_range(config.z_custom_mask)
        if custom_f_L is not None and custom_f_H is not None:
            eval_f_L, eval_f_H = custom_f_L, custom_f_H
            logger.info(f"カスタムマスクモードによる評価帯域: {eval_f_L:.2e} - {eval_f_H:.2e} Hz")
    eval_mask = create_evaluation_mask(f_grid, eval_f_L, eval_f_H, xp)

    target_mask = create_target_mask(f_grid, config.z_target, config.z_custom_mask, xp,
                                     mode=config.target_impedance_mode, config=config)
    target_mask_host = transfer_to_host(target_mask)
    eval_metadata = build_eval_metadata(f_grid, eval_mask, target_mask, xp)
    logger.info(f"評価帯域内の周波数点数: {eval_metadata['n_eval']}")

    with Timer("コンデンサインピーダンス計算"):
        cap_impedances = calculate_all_capacitor_impedances(config, f_grid, xp, gui_callback, dtype_c)

    # 容量の小さい順に並べる（測定ノード側が小容量）
    rlc_params = _capacitor_rlc_params(config, list(cap_impedances.keys()), cap_impedances, f_grid_host)
    sorted_cap_names = sorted(cap_impedances.keys(), key=lambda n: rlc_params[n]['C'])
    sorted_cap_impedances = {name: cap_impedances[name] for name in sorted_cap_names}
    pdn_assets = prepare_pdn_components(sorted_cap_impedances, config, f_grid, xp, dtype_c, rlc_params)
    logger.info(f"コンデンサ候補（容量小→大）: {sorted_cap_names}")

    ctx = OptimizationContext(
        config=config,
        xp=xp,
        is_gpu=is_gpu,
        pdn_assets=pdn_assets,
        eval_metadata=eval_metadata,
        stop_event=stop_event,
    )
    num_capacitors = len(sorted_cap_names)

    # デカップリングコンデンサ無しのPDN特性（参照表示用）
    z_without_decap = transfer_to_host(
        calculate_pdn_impedance_batch(xp.zeros((1, num_capacitors), dtype=np.int32), pdn_assets, xp)[0]
    )
    if gui_callback:
        gui_callback({
            'type': 'grid_update',
            'frequency_grid': f_grid_host,
            'target_mask': target_mask_host,
            'z_without_decap': z_without_decap,
        })

    min_counts_arr, max_counts_arr = resolve_capacitor_usage_bounds(config, sorted_cap_names)
    max_total_parts = int(config.max_total_parts)
    min_total = max(1, int(max_total_parts * config.min_total_parts_ratio), int(np.sum(min_counts_arr)))

    with Timer("組み合わせ準備"):
        combination_space = build_combination_space(
            num_capacitors, max_total_parts, min_total, min_counts_arr, max_counts_arr,
        )
    total_combinations = combination_space.total
    logger.info(f"組み合わせ総数: {total_combinations}")

    global_top_k: List[Dict] = []
    processed = 0

    if total_combinations > 0:
        chunk_size = _decide_chunk_size(total_combinations, len(f_grid), eval_metadata['n_eval'],
                                        num_capacitors, dtype_c, dtype_r, vram_budget, is_gpu)
        logger.info(f"チャンクサイズ: {chunk_size} Combi/Chunk")
        combination_iterator = combination_space.iter_chunks(
            chunk_size, xp, shuffle=config.shuffle_evaluation, seed=config.seed,
        )

        progress = get_progress_bar(range(total_combinations), desc="探索処理", total=total_combinations)
        progress_update = getattr(progress, "update", None)
        progress_close = getattr(progress, "close", None)

        try:
            with memory_cleanup(xp):
                for count_vectors_chunk in combination_iterator:
                    if ctx.should_stop():
                        logger.info("探索を停止しました")
                        break
                    if count_vectors_chunk is None or len(count_vectors_chunk) == 0:
                        continue

                    threshold = (global_top_k[-1]['total_score']
                                 if len(global_top_k) >= config.top_k else None)
                    candidates = process_chunk(ctx, count_vectors_chunk, threshold)
                    if ctx.should_stop():
                        logger.info("停止リクエストを検知したため探索を終了します")
                        break

                    global_top_k = update_global_top_k(global_top_k, candidates, config.top_k)
                    processed += len(count_vectors_chunk)
                    if progress_update:
                        progress_update(len(count_vectors_chunk))

                    if gui_callback:
                        gui_callback({
                            'type': 'top_k_update',
                            'top_k': global_top_k,
                            'progress': min(processed / max(total_combinations, 1), 1.0),
                            'capacitor_names': sorted_cap_names,
                            'frequency_grid': f_grid_host,
                            'target_mask': target_mask_host,
                        })
        except KeyboardInterrupt:
            logger.info("探索が中断されました")
        except Exception as e:
            logger.error(f"探索エラー: {e}")
            traceback.print_exc()
        finally:
            if progress_close:
                progress_close()
    else:
        logger.warning("有効な組み合わせが存在しないため探索をスキップします")

    host_cap_impedances = {name: transfer_to_host(z) for name, z in sorted_cap_impedances.items()}
    host_eval_metadata = {
        key: transfer_to_host(value) if hasattr(value, 'shape') else value
        for key, value in eval_metadata.items()
    }

    if is_gpu:
        sorted_cap_impedances.clear()
        pdn_assets.clear()
        ctx.pdn_assets = {}
        try:
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    results = {
        'top_k_results': global_top_k,
        'capacitor_names': sorted_cap_names,
        'frequency_grid': f_grid_host,
        'target_mask': target_mask_host,
        'z_pdn_without_decap': z_without_decap,
        'config': config,
        'backend': backend_name,
        'gpu_info': gpu_info,
        'capacitor_impedances': host_cap_impedances,
        'eval_metadata': host_eval_metadata,
        'stopped': bool(ctx.stop_flag),
    }

    if global_top_k and not ctx.stop_flag:
        with Timer("Excel出力"):
            export_to_excel(results, config)

    logger.info("最適化処理が完了しました")
    log_metrics(ctx.metrics)
    return results


def _decide_chunk_size(total: int, n_freq: int, n_eval: int, n_caps: int,
                       dtype_c: np.dtype, dtype_r: np.dtype, budget: int, is_gpu: bool) -> int:
    """
    1チャンクの組み合わせ数を決める

    ラダー計算で (B, F) の複素配列を約6本、スコア計算で (B, n_eval) の実数配列を約14本
    同時に保持するため、その合計がメモリバジェットの半分に収まるようにする。
    CPUではキャッシュ効率のため上限を小さくする。
    """
    per_combo = n_freq * dtype_c.itemsize * 6 + n_eval * dtype_r.itemsize * 14 + n_caps * 4
    by_memory = max(1, int(budget * 0.5) // max(per_combo, 1))
    cap = 65536 if is_gpu else 4096
    return int(max(1, min(total, by_memory, cap)))
