"""
メイン処理モジュール
全体処理、リソース管理、GPU情報取得、バックエンド決定、
周波数グリッド生成/評価帯域/マスク生成、VRAMバジェットとチャンク上限サイズ決定
"""

import os
import sys
import time
import traceback
import itertools
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass

# 絶対パスでインポート
from deca_auto.config import UserConfig, validate_config
from deca_auto.utils import (
    logger, get_backend, get_gpu_info, get_vram_budget,
    generate_frequency_grid, create_evaluation_mask, create_target_mask,
    Timer, memory_cleanup, get_progress_bar, transfer_to_device,
    get_dtype, ensure_numpy
)
from deca_auto.capacitor import calculate_all_capacitor_impedances
from deca_auto.pdn import calculate_pdn_impedance_batch
from deca_auto.evaluator import evaluate_combinations, extract_top_k, monte_carlo_evaluation
from deca_auto.excel_out import export_to_excel

# CuPy条件付きインポート
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
    eval_mask: np.ndarray
    target_mask: np.ndarray
    vram_budget: int
    chunk_size_limit: int
    capacitor_impedances: Dict[str, np.ndarray]
    capacitor_indices: np.ndarray
    top_k_results: List[Dict]
    stop_flag: bool = False
    gui_callback: Optional[Callable] = None
    stop_event: Optional[Any] = None


def generate_count_vectors(num_capacitors: int, max_total: int, 
                          min_total: int) -> np.ndarray:
    """
    コンデンサの組み合わせカウントベクトルを生成
    
    Args:
        num_capacitors: コンデンサ種類数
        max_total: 最大総数
        min_total: 最小総数
    
    Returns:
        カウントベクトルの配列 (N_combinations, num_capacitors)
    """
    logger.info(f"組み合わせ生成: {num_capacitors}種類, 総数{min_total}～{max_total}個")
    
    out = []

    for total in range(min_total, max_total + 1):
        N = total + num_capacitors - 1
        # バー位置の全組合せ
        for bars in itertools.combinations(range(N), num_capacitors - 1):
            prev = -1
            counts = []
            for b in (*bars, N):
                counts.append(b - prev - 1)
                prev = b
            out.append(tuple(counts))
    
    count_vectors = np.asarray(out, dtype=np.int32)
    logger.info(f"生成された組み合わせ数: {len(count_vectors)}")
    
    return count_vectors


def shuffle_combinations(count_vectors: np.ndarray, 
                       buffer_size: int,
                       seed: int = None) -> np.ndarray:
    """
    組み合わせの順番をバッファサイズごとにシャッフル
    
    Args:
        count_vectors: カウントベクトル
        buffer_size: バッファサイズ
        seed: 乱数シード
    
    Returns:
        シャッフルされたカウントベクトル
    """
    n = len(count_vectors)
    # NumPyのRNGを使用（CPU側でシャッフル）
    rng = np.random.default_rng(seed)
    
    # バッファサイズごとにシャッフル
    for i in range(0, n, buffer_size):
        end = min(i + buffer_size, n)
        indices = np.arange(i, end)
        rng.shuffle(indices)
        count_vectors[i:end] = count_vectors[indices]
    
    return count_vectors


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
    
    # PDNインピーダンス計算（バッチ処理）
    z_pdn_batch = calculate_pdn_impedance_batch(
        count_vectors_gpu,
        ctx.capacitor_impedances,
        ctx.capacitor_indices,
        ctx.f_grid,
        ctx.config,
        xp
    )

    if ctx.stop_event and ctx.stop_event.is_set():
        ctx.stop_flag = True
        return {'chunk_id': chunk_id, 'top_k': [], 'num_processed': len(count_vectors_chunk)}
    
    # スコア評価
    scores = evaluate_combinations(
        z_pdn_batch,
        ctx.target_mask,
        ctx.eval_mask,
        count_vectors_gpu,
        ctx.config,
        xp,
        ctx.f_grid,
    )

    if ctx.stop_event and ctx.stop_event.is_set():
        ctx.stop_flag = True
        return {'chunk_id': chunk_id, 'top_k': [], 'num_processed': len(count_vectors_chunk)}
    
    # チャンク内のtop_k抽出
    chunk_top_k = extract_top_k(
        z_pdn_batch,
        scores,
        count_vectors_gpu,
        ctx.config.top_k,
        xp
    )
    
    # Monte Carlo評価（有効な場合）
    if ctx.config.mc_enable and len(chunk_top_k) > 0:
        mc_scores = monte_carlo_evaluation(
            chunk_top_k,
            ctx.capacitor_impedances,
            ctx.capacitor_indices,
            ctx.f_grid,
            ctx.eval_mask,
            ctx.target_mask,
            ctx.config,
            xp
        )
        
        # MC最悪値をスコアに反映
        for i, result in enumerate(chunk_top_k):
            result['mc_worst_score'] = mc_scores[i]
            result['total_score'] += ctx.config.weight_mc_worst * mc_scores[i]
    
    # CPUに転送（必要な場合）
    if ctx.is_gpu:
        for result in chunk_top_k:
            result['z_pdn'] = transfer_to_device(result['z_pdn'], np)
            result['count_vector'] = transfer_to_device(result['count_vector'], np)
    
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
            logger.info(f"GPU情報: {gpu_info['name']} - "
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
    eval_f_L = config.f_L
    eval_f_H = config.f_H
    
    # カスタムマスク使用時は評価帯域を更新
    if config.z_custom_mask:
        custom_freqs = [f for f, _ in config.z_custom_mask]
        eval_f_L = min(custom_freqs)
        eval_f_H = max(custom_freqs)
        logger.info(f"カスタムマスクによる評価帯域: {eval_f_L:.2e} - {eval_f_H:.2e} Hz")
    
    eval_mask = create_evaluation_mask(f_grid, eval_f_L, eval_f_H, xp)
    logger.info(f"評価帯域内の周波数点数: {xp.sum(eval_mask)}")
    
    # 目標マスク生成
    target_mask = create_target_mask(
        f_grid,
        config.z_target,
        config.z_custom_mask,
        xp
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
        
        # GUIに周波数グリッドとターゲットマスクを送信
        if gui_callback:
            gui_callback({
                'type': 'grid_update',
                'frequency_grid': transfer_to_device(f_grid, np),
                'target_mask': transfer_to_device(target_mask, np)
            })
    
    # 容量でソート（小→大）
    capacitor_names = list(cap_impedances.keys())
    capacitances = []
    for name in capacitor_names:
        # 容量値を取得（設定から or インピーダンスから推定）
        cap_config = next((c for c in config.capacitors if c['name'] == name), None)
        if cap_config and 'C' in cap_config:
            capacitances.append(cap_config['C'])
        else:
            # インピーダンスから容量推定（NumPyで実行）
            z_c = cap_impedances[name]
            z_c_np = ensure_numpy(z_c) if xp is not np else z_c
            f_grid_np = ensure_numpy(f_grid) if xp is not np else f_grid
            omega = 2 * np.pi * f_grid_np
            c_estimated = -1 / (omega * np.imag(z_c_np)).mean()
            capacitances.append(float(c_estimated))
    
    # ソート
    sorted_indices = np.argsort(capacitances)
    sorted_cap_names = [capacitor_names[i] for i in sorted_indices]
    sorted_cap_impedances = {name: cap_impedances[name] for name in sorted_cap_names}
    
    logger.info(f"コンデンサ順序（容量小→大）: {sorted_cap_names}")
    
    # 最適化コンテキスト作成
    ctx = OptimizationContext(
        config=config,
        xp=xp,
        backend_name=backend_name,
        is_gpu=is_gpu,
        dtype_c=dtype_c,
        dtype_r=dtype_r,
        f_grid=f_grid,
        eval_mask=eval_mask,
        target_mask=target_mask,
        vram_budget=vram_budget,
        chunk_size_limit=chunk_size_limit,
        capacitor_impedances=sorted_cap_impedances,
        capacitor_indices=xp.arange(len(sorted_cap_names)),
        top_k_results=[],
        gui_callback=gui_callback,
        stop_event=stop_event,
    )
    
    # カウントベクトル生成
    num_capacitors = len(sorted_cap_names)
    min_total = max(1, int(config.max_total_parts * config.min_total_parts_ratio))
    
    with Timer("組み合わせ生成"):
        count_vectors = generate_count_vectors(
            num_capacitors,
            config.max_total_parts,
            min_total
        )
    
    # シャッフル（有効な場合）
    if config.shuffle_evaluation:
        count_vectors = shuffle_combinations(
            count_vectors,
            int(config.buffer_limit),
            config.seed
        )
    
    # チャンクサイズ決定
    bytes_per_combination = int(
        num_capacitors * np.dtype(np.int32).itemsize +
        len(f_grid) * dtype_c.itemsize +
        len(f_grid) * dtype_r.itemsize
    )
    bytes_per_combination = max(bytes_per_combination, 1)
    safety_margin = 1.2
    effective_limit = int(ctx.chunk_size_limit / safety_margin)
    raw_chunk = max(1, min(len(count_vectors), effective_limit // bytes_per_combination))
    chunk_size = max(1, raw_chunk)
    estimated_mem = chunk_size * bytes_per_combination / (1024 ** 2)

    logger.info(
        f"チャンクサイズ: {chunk_size} 組み合わせ/チャンク (推定使用メモリ: {estimated_mem:.2f} MB)"
    )
    
    # 探索ループ
    num_chunks = (len(count_vectors) + chunk_size - 1) // chunk_size
    global_top_k = []
    
    progress = get_progress_bar(
        range(num_chunks),
        desc="探索処理",
        total=num_chunks
    )
    
    try:
        with memory_cleanup(xp):
            for chunk_id in progress:
                if stop_event and stop_event.is_set():
                    ctx.stop_flag = True
                # 停止フラグチェック
                if ctx.stop_flag:
                    logger.info("探索を停止しました")
                    break

                # チャンク取得
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(count_vectors))
                chunk = count_vectors[start_idx:end_idx]
                
                # チャンク処理
                chunk_results = process_chunk(ctx, chunk, chunk_id)
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

                # GUI更新コールバック
                if gui_callback:
                    gui_callback({
                        'type': 'top_k_update',
                        'top_k': global_top_k,
                        'progress': (chunk_id + 1) / num_chunks,
                        'capacitor_names': sorted_cap_names,
                        'frequency_grid': transfer_to_device(f_grid, np),
                        'target_mask': transfer_to_device(target_mask, np)
                    })
                
                # 進捗ログ
                if (chunk_id + 1) % 10 == 0:
                    best_score = global_top_k[0]['total_score'] if global_top_k else float('inf')
                    logger.info(f"進捗: {(chunk_id + 1) / num_chunks * 100:.1f}%, "
                               f"ベストスコア: {best_score:.6f}")

    except KeyboardInterrupt:
        logger.info("探索が中断されました")
    except Exception as e:
        logger.error(f"探索エラー: {e}")
        traceback.print_exc()

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
        'frequency_grid': transfer_to_device(f_grid, np),
        'target_mask': transfer_to_device(target_mask, np),
        'config': config,
        'backend': backend_name,
        'gpu_info': get_gpu_info(config.cuda) if is_gpu else None,
        'stopped': bool(ctx.stop_flag)
    }

    # Excel出力
    if global_top_k and not ctx.stop_flag:
        with Timer("Excel出力"):
            export_to_excel(results, config)
    
    logger.info("最適化処理が完了しました")
    
    return results


def stop_optimization(ctx: OptimizationContext):
    """最適化処理を停止"""
    ctx.stop_flag = True
    logger.info("停止リクエストを受信しました")
