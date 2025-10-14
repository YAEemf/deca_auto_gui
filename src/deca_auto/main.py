import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable, Iterable
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
    estimate_capacitance_from_impedance
)
from deca_auto.pdn import calculate_pdn_impedance_batch, prepare_pdn_components
from deca_auto.evaluator import evaluate_combinations, extract_top_k, monte_carlo_evaluation
from deca_auto.excel_out import export_to_excel

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class OptimizationContext:
    """最適化実行コンテキスト"""
    config: UserConfig
    xp: Any  # NumPy or CuPy
    backend_name: str
    is_gpu: bool
    dtype_c: np.dtype
    dtype_r: np.dtype
    f_grid: np.ndarray
    f_grid_host: np.ndarray
    eval_mask: np.ndarray
    target_mask: np.ndarray
    target_mask_host: np.ndarray
    vram_budget: int
    chunk_size_limit: int
    capacitor_impedances: Dict[str, np.ndarray]
    capacitor_indices: np.ndarray
    pdn_assets: Dict[str, Any]
    top_k_results: List[Dict]
    eval_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stop_flag: bool = False
    gui_callback: Optional[Callable] = None
    stop_event: Optional[Any] = None
    z_without_decap: Optional[np.ndarray] = None


MAX_COMBINATION_BUFFER_BYTES = 256 * 1024 * 1024  # 256MB上限


class CombinationGenerator:
    """メモリ効率の良い組み合わせ列挙器"""

    def __init__(
        self,
        num_capacitors: int,
        min_counts: np.ndarray,
        max_counts: np.ndarray,
        min_total: int,
        max_total: int,
        progress_desc: str = "組み合わせ生成",
        total_desc: str = "組み合わせ総数計算",
    ) -> None:
        self.num_capacitors = int(num_capacitors)
        self.min_counts = np.asarray(min_counts, dtype=np.int32)
        self.max_counts = np.asarray(max_counts, dtype=np.int32)
        self.min_total = int(min_total)
        self.max_total = int(max_total)
        self.progress_desc = progress_desc
        self.total_desc = total_desc
        self._total_cache: Optional[int] = None

        valid_lengths = (
            self.min_counts.size == self.num_capacitors
            and self.max_counts.size == self.num_capacitors
        )
        self.valid = (
            self.num_capacitors >= 0
            and valid_lengths
            and self.min_total <= self.max_total
        )

        if self.valid and self.num_capacitors > 0:
            self.suffix_min = np.cumsum(self.min_counts[::-1])[::-1]
            self.suffix_max = np.cumsum(self.max_counts[::-1])[::-1]
        else:
            self.suffix_min = np.zeros(max(self.num_capacitors, 1), dtype=np.int32)
            self.suffix_max = np.zeros(max(self.num_capacitors, 1), dtype=np.int32)

    def _bounds(self, index: int, accumulated: int) -> Tuple[int, int]:
        remaining_min = self.suffix_min[index + 1] if (index + 1) < self.num_capacitors else 0
        remaining_max = self.suffix_max[index + 1] if (index + 1) < self.num_capacitors else 0
        lower = max(self.min_counts[index], self.min_total - accumulated - remaining_max)
        upper = min(self.max_counts[index], self.max_total - accumulated - remaining_min)
        return int(lower), int(upper)

    def total_count(self) -> int:
        if not self.valid:
            self._total_cache = 0
            return 0

        if self._total_cache is not None:
            return self._total_cache

        if self.num_capacitors == 0:
            total = int(self.min_total <= 0 <= self.max_total)
            self._total_cache = total
            return total

        max_limit = self.max_total
        dp = [0] * (max_limit + 1)
        dp[0] = 1

        progress = get_progress_bar(
            range(self.num_capacitors),
            desc=self.total_desc,
            total=self.num_capacitors
        )
        progress_update = getattr(progress, "update", None)
        progress_close = getattr(progress, "close", None)

        try:
            for idx in range(self.num_capacitors):
                lower = int(self.min_counts[idx])
                upper = int(self.max_counts[idx])
                if lower > upper:
                    dp = [0] * (max_limit + 1)
                    break

                next_dp = [0] * (max_limit + 1)
                for total in range(max_limit + 1):
                    base = dp[total]
                    if base == 0:
                        continue
                    if lower > max_limit - total:
                        continue
                    max_add = min(upper, max_limit - total)
                    for add in range(lower, max_add + 1):
                        next_dp[total + add] += base
                dp = next_dp
                if progress_update:
                    progress_update(1)
        finally:
            if progress_close:
                progress_close()

        total = sum(dp[self.min_total:self.max_total + 1]) if dp else 0
        self._total_cache = total
        return total

    def iter_chunks(
        self,
        chunk_size: int,
        shuffle: bool = False,
        buffer_limit: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterable[np.ndarray]:
        chunk_size = max(1, int(chunk_size))
        total = self.total_count()

        per_combo_bytes = max(self.num_capacitors, 1) * np.dtype(np.int32).itemsize
        max_buffer_by_mem = max(1, MAX_COMBINATION_BUFFER_BYTES // per_combo_bytes)

        if buffer_limit is None:
            buffer_limit = chunk_size
        buffer_limit = max(1, int(buffer_limit))
        buffer_limit = max(buffer_limit, chunk_size)
        buffer_limit = min(buffer_limit, max_buffer_by_mem)

        rng = np.random.default_rng(seed) if shuffle else None

        progress = get_progress_bar(
            range(total),
            desc=self.progress_desc,
            total=total
        )
        progress_update = getattr(progress, "update", None)
        progress_close = getattr(progress, "close", None)

        def generator():
            buffer_width = self.num_capacitors
            buffer_array = np.zeros((buffer_limit, buffer_width), dtype=np.int32)
            buffer_size = 0
            current = np.zeros(buffer_width, dtype=np.int32)

            def flush_buffer():
                nonlocal buffer_size
                if buffer_size == 0:
                    return
                view = buffer_array[:buffer_size].copy()
                if rng is not None:
                    rng.shuffle(view, axis=0)
                total_rows = view.shape[0]
                start = 0
                while start < total_rows:
                    end = min(start + chunk_size, total_rows)
                    yield view[start:end]
                    start = end
                buffer_size = 0

            def backtrack(index: int, accumulated: int):
                nonlocal buffer_size
                if self.num_capacitors == 0:
                    return
                if index == self.num_capacitors:
                    if self.min_total <= accumulated <= self.max_total:
                        if progress_update:
                            progress_update(1)
                        if buffer_size >= buffer_limit:
                            yield from flush_buffer()
                        buffer_array[buffer_size, :buffer_width] = current
                        buffer_size += 1
                    return

                lower, upper = self._bounds(index, accumulated)
                if upper < lower:
                    return

                for value in range(lower, upper + 1):
                    current[index] = value
                    yield from backtrack(index + 1, accumulated + value)
                current[index] = 0

            try:
                if self.num_capacitors == 0:
                    if self.min_total <= 0 <= self.max_total:
                        arr = np.zeros((1, 0), dtype=np.int32)
                        if progress_update:
                            progress_update(1)
                        yield arr
                    return

                if total == 0:
                    return

                yield from backtrack(0, 0)
                yield from flush_buffer()
            finally:
                if progress_close:
                    progress_close()

        return generator()


def generate_count_vectors(
    num_capacitors: int,
    max_total: int,
    min_total: int,
    min_counts: Optional[np.ndarray] = None,
    max_counts: Optional[np.ndarray] = None,
) -> CombinationGenerator:
    """
    コンデンサの組み合わせベクトルを生成

    Args:
        num_capacitors: コンデンサ種類数
        max_total: 最大総数
        min_total: 最小総数
        min_counts: 各コンデンサの最小使用数
        max_counts: 各コンデンサの最大使用数

    Returns:
        カウントベクトルの配列 (N_combinations, num_capacitors)
    """
    if max_total < 0 or min_total < 0:
        raise ValueError("max_totalおよびmin_totalは非負である必要があります")

    if min_total > max_total:
        logger.warning("最小総数が最大総数を上回っています。組み合わせは生成されません")
        return CombinationGenerator(
            num_capacitors,
            np.zeros(num_capacitors, dtype=np.int32),
            np.zeros(num_capacitors, dtype=np.int32),
            1,
            0,
        )

    if min_counts is None:
        min_counts = np.zeros(num_capacitors, dtype=np.int32)
    else:
        min_counts = np.asarray(min_counts, dtype=np.int32)

    if max_counts is None:
        max_counts = np.full(num_capacitors, max_total, dtype=np.int32)
    else:
        max_counts = np.asarray(max_counts, dtype=np.int32)

    if min_counts.shape[0] != num_capacitors or max_counts.shape[0] != num_capacitors:
        raise ValueError("min_counts と max_counts の長さは num_capacitors に一致する必要があります")

    max_counts = np.minimum(max_counts, max_total)
    min_counts = np.maximum(min_counts, 0)

    total_min_possible = int(np.sum(min_counts))
    total_max_possible = int(np.sum(max_counts))

    if total_min_possible > max_total:
        logger.error("各コンデンサの最小使用数の総和が最大総数を超えています。組み合わせが存在しません")
        return CombinationGenerator(
            num_capacitors,
            min_counts,
            max_counts,
            1,
            0,
        )

    adjusted_min_total = max(min_total, total_min_possible)
    adjusted_max_total = min(max_total, total_max_possible)

    if adjusted_min_total > adjusted_max_total:
        logger.warning("指定された範囲では有効な組み合わせが存在しません")
        return CombinationGenerator(
            num_capacitors,
            min_counts,
            max_counts,
            1,
            0,
        )

    logger.info(
        f"組み合わせ生成: {num_capacitors}種類, 総数{adjusted_min_total}～{adjusted_max_total}個, "
        f"個別範囲min={min_counts.tolist()}, max={max_counts.tolist()}"
    )

    return CombinationGenerator(
        num_capacitors,
        min_counts,
        max_counts,
        adjusted_min_total,
        adjusted_max_total,
    )


def process_chunk(ctx: OptimizationContext, 
                 count_vectors_chunk: np.ndarray,
                 chunk_id: int) -> Dict:
    """
    チャンクの処理（PDN計算と評価）
    
    Args:
        ctx: 最適化コンテキスト
        count_vectors_chunk: カウントベクトルのチャンク
        chunk_id: チャンクID
    
    Returns:
        チャンクの処理結果
    """
    xp = ctx.xp

    if ctx.stop_flag or (ctx.stop_event and ctx.stop_event.is_set()):
        ctx.stop_flag = True
        return {'chunk_id': chunk_id, 'top_k': [], 'num_processed': 0}
    
    # カウントベクトルをGPUに転送
    if ctx.is_gpu:
        count_vectors_gpu = transfer_to_device(count_vectors_chunk, xp)
    else:
        count_vectors_gpu = count_vectors_chunk

    increment_metric(ctx.metrics, "chunk_received", 1)
    update_metric_max(ctx.metrics, "chunk_size_max", len(count_vectors_chunk))
    increment_metric(ctx.metrics, "combinations_received", len(count_vectors_chunk))

    # PDNインピーダンス計算（バッチ処理）
    with profile_block(ctx.metrics, "pdn_impedance_batch"):
        z_pdn_batch = calculate_pdn_impedance_batch(
            count_vectors_gpu,
            ctx.capacitor_impedances,
            ctx.capacitor_indices,
            ctx.f_grid,
            ctx.config,
            xp,
            prepared=ctx.pdn_assets,
        )

    if ctx.stop_event and ctx.stop_event.is_set():
        ctx.stop_flag = True
        return {'chunk_id': chunk_id, 'top_k': [], 'num_processed': len(count_vectors_chunk)}

    # スコア評価
    with profile_block(ctx.metrics, "combination_evaluation"):
        scores = evaluate_combinations(
            z_pdn_batch,
            ctx.target_mask,
            ctx.eval_mask,
            count_vectors_gpu,
            ctx.config,
            xp,
            ctx.f_grid,
            eval_metadata=ctx.eval_metadata,
            z_without_decap=ctx.z_without_decap,
        )

    if ctx.stop_event and ctx.stop_event.is_set():
        ctx.stop_flag = True
        return {'chunk_id': chunk_id, 'top_k': [], 'num_processed': len(count_vectors_chunk)}

    # チャンク内のtop_k抽出
    with profile_block(ctx.metrics, "topk_extraction"):
        chunk_top_k = extract_top_k(
            z_pdn_batch,
            scores,
            count_vectors_gpu,
            ctx.config.top_k,
            xp
        )

    # Monte Carlo評価（有効な場合）
    if ctx.config.mc_enable and len(chunk_top_k) > 0:
        with profile_block(ctx.metrics, "monte_carlo_evaluation"):
            mc_scores = monte_carlo_evaluation(
                chunk_top_k,
                ctx.capacitor_impedances,
                ctx.capacitor_indices,
                ctx.f_grid,
                ctx.eval_mask,
                ctx.target_mask,
                ctx.config,
                xp,
                ctx.pdn_assets,
                eval_metadata=ctx.eval_metadata,
                z_without_decap=ctx.z_without_decap,
            )
        
        # MC最悪値をスコアに反映
        mc_scores_host = transfer_to_host(mc_scores) if ctx.is_gpu else mc_scores
        for i, result in enumerate(chunk_top_k):
            mc_value = float(mc_scores_host[i])
            result['mc_worst_score'] = mc_value
            result['total_score'] += ctx.config.weight_mc_worst * mc_value

    # CPUに転送（必要な場合）
    if ctx.is_gpu:
        for result in chunk_top_k:
            result['z_pdn'] = transfer_to_host(result['z_pdn'])
            result['count_vector'] = transfer_to_host(result['count_vector'])

        snapshot = get_memory_snapshot(ctx.xp)
        for name, value in snapshot.items():
            update_metric_max(ctx.metrics, f"gpu_{name}_max", value)

    increment_metric(ctx.metrics, "chunk_processed", 1)
    increment_metric(ctx.metrics, "topk_candidates_collected", len(chunk_top_k))
    
    return {
        'chunk_id': chunk_id,
        'top_k': chunk_top_k,
        'num_processed': len(count_vectors_chunk)
    }


def update_global_top_k(global_top_k: List[Dict], 
                       chunk_results: Dict,
                       k: int) -> List[Dict]:
    """
    グローバルのtop_kを更新
    
    Args:
        global_top_k: 現在のグローバルtop_k
        chunk_results: チャンクの処理結果
        k: 保持する上位数
    
    Returns:
        更新されたグローバルtop_k
    """
    # チャンク結果をグローバルに追加
    all_results = global_top_k + chunk_results['top_k']

    # スコアでソート
    all_results.sort(key=lambda x: x['total_score'])

    # 上位k個を保持し、順位を再採番
    top_k = all_results[:k]
    for idx, result in enumerate(top_k, start=1):
        result['rank'] = idx
    return top_k


def run_optimization(config: UserConfig,
                    gui_callback: Optional[Callable] = None,
                    stop_event: Optional[Any] = None) -> Dict:
    """
    最適化処理のメイン実行関数
    
    Args:
        config: ユーザー設定
        gui_callback: GUI更新用コールバック関数
    
    Returns:
        最適化結果
    """
    
    # 設定検証
    if not validate_config(config):
        raise ValueError("設定が無効です")
    
    # バックエンド決定
    xp, backend_name, is_gpu = get_backend(
        force_numpy=config.force_numpy,
        cuda_device=config.cuda
    )
    
    # GPU情報取得
    if is_gpu:
        gpu_info = get_gpu_info(config.cuda)
        if gpu_info:
            logger.info(f"選択中GPU: {gpu_info['name']} \n"
                       f"Total: {gpu_info['total_memory_gb']:.2f}GB, "
                       f"Free: {gpu_info['free_memory_gb']:.2f}GB")
    
    # データ型設定
    dtype_c = np.dtype(get_dtype(config.dtype_c))
    dtype_r = np.dtype(get_dtype(config.dtype_r))
    
    # VRAMバジェット決定
    vram_budget, chunk_size_limit = get_vram_budget(
        config.max_vram_ratio_limit,
        config.cuda
    )
    logger.info(f"VRAMバジェット: {vram_budget / 1024**3:.2f}GB, "
               f"チャンクサイズ上限: {chunk_size_limit / 1024**2:.2f}MB")
    
    # 周波数グリッド生成
    with Timer("周波数グリッド生成"):
        f_grid = generate_frequency_grid(
            config.f_start,
            config.f_stop,
            config.num_points_per_decade,
            xp,
            dtype_r,
        )
        logger.info(f"周波数点数: {len(f_grid)}")
    
    # 評価帯域マスク生成
    eval_f_L, eval_f_H = config.f_L, config.f_H

    # カスタムマスク使用時は評価帯域を更新
    if config.z_custom_mask:
        custom_f_L, custom_f_H = get_custom_mask_freq_range(config.z_custom_mask)
        if custom_f_L is not None and custom_f_H is not None:
            eval_f_L, eval_f_H = custom_f_L, custom_f_H
            logger.info(f"カスタムマスクによる評価帯域: {eval_f_L:.2e} - {eval_f_H:.2e} Hz")

    eval_mask = create_evaluation_mask(f_grid, eval_f_L, eval_f_H, xp)
    logger.info(f"評価帯域内の周波数点数: {xp.sum(eval_mask)}")
    eval_indices = xp.where(eval_mask)[0]
    eval_metadata: Dict[str, Any] = {
        'indices': eval_indices,
        'n_eval': int(eval_indices.size),
    }
    if int(eval_indices.size) >= 2:
        f_eval = f_grid[eval_indices]
        df_log = xp.log10(f_eval[1:] / f_eval[:-1])
        df_log = xp.where(xp.isfinite(df_log) & (df_log > 0), df_log, 0.0)
        eval_metadata['df_log'] = df_log
    
    # 目標マスク生成（モード対応）
    target_mask = create_target_mask(
        f_grid,
        config.z_target,
        config.z_custom_mask,
        xp,
        mode=config.target_impedance_mode,
        config=config
    )
    
    # コンデンサインピーダンス計算
    with Timer("コンデンサインピーダンス計算"):
        cap_impedances = calculate_all_capacitor_impedances(
            config,
            f_grid,
            xp,
            gui_callback,
            dtype_c,
        )
    
    # 容量でソート（小→大）
    capacitor_names = list(cap_impedances.keys())
    capacitances = {}

    for name in capacitor_names:
        cap_config = next((c for c in config.capacitors if c['name'] == name), None)
        if cap_config and 'C' in cap_config and cap_config['C'] is not None:
            capacitances[name] = to_float(cap_config['C'], 1e-6)
        else:
            # インピーダンスから容量推定（utils.pyの統合版を使用）
            z_c_np = ensure_numpy(cap_impedances[name])
            f_grid_np = ensure_numpy(f_grid)
            capacitances[name] = estimate_capacitance_from_impedance(
                z_c_np, f_grid_np, method='median'  # 中央値でロバストに推定
            )

    # ソート
    sorted_cap_names = sorted(capacitor_names, key=lambda n: capacitances[n])
    sorted_cap_impedances = {name: cap_impedances[name] for name in sorted_cap_names}
    pdn_assets = prepare_pdn_components(sorted_cap_impedances, config, f_grid, xp, dtype_c)
    if pdn_assets.get('cap_names') != tuple(sorted_cap_names):
        pdn_assets['cap_names'] = tuple(sorted_cap_names)
    
    logger.info(f"コンデンサ候補（容量小→大）: {sorted_cap_names}")
    
    f_grid_host = transfer_to_host(f_grid)
    target_mask_host = transfer_to_host(target_mask)

    # 最適化コンテキスト作成
    ctx = OptimizationContext(
        config=config,
        xp=xp,
        backend_name=backend_name,
        is_gpu=is_gpu,
        dtype_c=dtype_c,
        dtype_r=dtype_r,
        f_grid=f_grid,
        f_grid_host=f_grid_host,
        eval_mask=eval_mask,
        target_mask=target_mask,
        target_mask_host=target_mask_host,
        vram_budget=vram_budget,
        chunk_size_limit=chunk_size_limit,
        capacitor_impedances=sorted_cap_impedances,
        capacitor_indices=xp.arange(len(sorted_cap_names)),
        pdn_assets=pdn_assets,
        top_k_results=[],
        eval_metadata=eval_metadata,
        gui_callback=gui_callback,
        stop_event=stop_event,
    )
    
    # カウントベクトル生成
    num_capacitors = len(sorted_cap_names)

    # PDN特性（デカップリングなし）を計算
    try:
        zero_counts = np.zeros((1, num_capacitors), dtype=np.int32)
        z_without_decap = calculate_pdn_impedance_batch(
            zero_counts,
            ctx.capacitor_impedances,
            ctx.capacitor_indices,
            ctx.f_grid,
            ctx.config,
            xp,
            prepared=ctx.pdn_assets,
        )[0]
        z_without_decap_np = transfer_to_host(z_without_decap)
        ctx.z_without_decap = z_without_decap_np
    except Exception as exc:
        logger.error(f"PDN特性（デカップリングなし）計算に失敗しました: {exc}")
        z_without_decap_np = None

    if gui_callback:
        payload = {
            'type': 'grid_update',
            'frequency_grid': ctx.f_grid_host,
            'target_mask': ctx.target_mask_host,
        }
        if z_without_decap_np is not None:
            payload['z_without_decap'] = z_without_decap_np
        gui_callback(payload)

    max_total_parts = config.max_total_parts
    min_counts_arr, max_counts_arr = resolve_capacitor_usage_bounds(
        config,
        sorted_cap_names,
    )

    min_total = max(1, int(config.max_total_parts * config.min_total_parts_ratio))
    min_total = max(min_total, int(np.sum(min_counts_arr)))
    
    with Timer("組み合わせ準備"):
        combination_space = generate_count_vectors(
            num_capacitors,
            max_total_parts,
            min_total,
            min_counts_arr,
            max_counts_arr,
        )

    total_combinations = combination_space.total_count()
    logger.info(f"組み合わせ総数: {total_combinations}")

    global_top_k: List[Dict] = []
    processed_combinations = 0

    if total_combinations > 0:
        # チャンクサイズ決定
        bytes_per_combination = int(
            num_capacitors * np.dtype(np.int32).itemsize +
            len(f_grid) * dtype_c.itemsize +
            len(f_grid) * dtype_r.itemsize
        )
        bytes_per_combination = max(bytes_per_combination, 1)
        safety_margin = 1.2
        effective_limit = max(1, int(ctx.chunk_size_limit / safety_margin))
        max_chunk_by_mem = max(1, effective_limit // bytes_per_combination)
        raw_chunk = max(1, min(total_combinations, max_chunk_by_mem))
        chunk_size = max(1, raw_chunk)
        estimated_mem = chunk_size * bytes_per_combination / (1024 ** 2)

        logger.info(
            f"チャンクサイズ: {chunk_size} Combi/Chunk (推定使用メモリ: {estimated_mem:.2f} MB)"
        )

        buffer_limit = None
        if config.buffer_limit and config.buffer_limit > 0:
            buffer_limit = int(config.buffer_limit)

        combination_iterator = combination_space.iter_chunks(
            chunk_size=chunk_size,
            shuffle=config.shuffle_evaluation,
            buffer_limit=buffer_limit,
            seed=config.seed,
        )

        progress = get_progress_bar(
            range(total_combinations),
            desc="探索処理",
            total=total_combinations
        )
        progress_update = getattr(progress, "update", None)
        progress_close = getattr(progress, "close", None)

        try:
            with memory_cleanup(xp):
                for chunk_id, count_vectors_chunk in enumerate(combination_iterator):
                    if stop_event and stop_event.is_set():
                        ctx.stop_flag = True
                    if ctx.stop_flag:
                        logger.info("探索を停止しました")
                        break

                    if count_vectors_chunk is None or len(count_vectors_chunk) == 0:
                        continue

                    # チャンク処理
                    chunk_results = process_chunk(ctx, count_vectors_chunk, chunk_id)
                    if ctx.stop_event and ctx.stop_event.is_set():
                        ctx.stop_flag = True

                    if ctx.stop_flag:
                        logger.info("停止リクエストを検知したため探索を終了します")
                        break

                    # グローバルtop_k更新
                    global_top_k = update_global_top_k(
                        global_top_k,
                        chunk_results,
                        config.top_k
                    )
                    ctx.top_k_results = global_top_k

                    processed_combinations += int(chunk_results.get('num_processed', len(count_vectors_chunk)))
                    if progress_update:
                        progress_update(int(chunk_results.get('num_processed', len(count_vectors_chunk))))

                    progress_ratio = min(
                        processed_combinations / max(total_combinations, 1),
                        1.0
                    )

                    # GUI更新コールバック
                    if gui_callback:
                        gui_callback({
                            'type': 'top_k_update',
                            'top_k': global_top_k,
                            'progress': progress_ratio,
                            'capacitor_names': sorted_cap_names,
                            'frequency_grid': ctx.f_grid_host,
                            'target_mask': ctx.target_mask_host
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
        ctx.top_k_results = []

    ctx.top_k_results = global_top_k

    host_cap_impedances = {
        name: transfer_to_host(z)
        for name, z in sorted_cap_impedances.items()
    }
    host_eval_metadata: Dict[str, Any] = {}
    if ctx.eval_metadata:
        for key, value in ctx.eval_metadata.items():
            if hasattr(value, '__cuda_array_interface__') or isinstance(value, np.ndarray):
                host_eval_metadata[key] = transfer_to_host(value)
            else:
                host_eval_metadata[key] = value

    if ctx.is_gpu:
        sorted_cap_impedances.clear()
        ctx.capacitor_impedances = {}
        try:
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # 結果の整理
    results = {
        'top_k_results': global_top_k,
        'capacitor_names': sorted_cap_names,
        'frequency_grid': ctx.f_grid_host,
        'target_mask': ctx.target_mask_host,
        'z_pdn_without_decap': ctx.z_without_decap,
        'config': config,
        'backend': backend_name,
        'gpu_info': get_gpu_info(config.cuda) if is_gpu else None,
        'capacitor_impedances': host_cap_impedances,
        'eval_metadata': host_eval_metadata,
        'stopped': bool(ctx.stop_flag)
    }

    # Excel出力
    if global_top_k and not ctx.stop_flag:
        with Timer("Excel出力"):
            export_to_excel(results, config)
    
    logger.info("最適化処理が完了しました")
    log_metrics(ctx.metrics)
    
    return results


def stop_optimization(ctx: OptimizationContext):
    """最適化処理を停止"""
    ctx.stop_flag = True
    logger.info("停止リクエストを受信しました")
