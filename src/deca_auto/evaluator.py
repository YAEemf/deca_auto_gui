import traceback
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from deca_auto.config import UserConfig
from deca_auto.utils import logger, transfer_to_host, safe_divide
from deca_auto.pdn import calculate_pdn_impedance_monte_carlo, prepare_pdn_components


def calculate_score_components_batch(
    z_pdn_batch: Any,
    target_curve: Any,
    eval_mask: Any,
    count_vectors: Any,
    config: UserConfig,
    xp: Any = np,
    f_grid: Optional[Any] = None,
    eval_metadata: Optional[Dict[str, Any]] = None,
    z_without_decap: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    バッチ処理版スコアコンポーネント計算

    評価項目（全て正の値で、小さいほど良い特性）:
        - max: 最大インピーダンス比
        - area: 目標超過面積（台形積分）
        - mean: 平均インピーダンス比（対数周波数重み付き）
        - anti: アンチレゾナンス（ピークの高さと鋭さを考慮）
        - flat: 平滑さ（対数周波数での重み付き変動係数）
        - under: 目標未達面積（負の重みで良い特性として評価）
        - parts: 部品数（正規化）
        - num_types: 種類数（正規化）
        - resonance: レゾナンス（谷の深さ、鋭さ、周波数重みを考慮）
        - improvement: Without Decapからの悪化面積
        - low_improvement: 低改善度帯域の面積

    改善内容（v8.1）:
        - mean: 対数周波数での重み付き平均に変更
        - flat: 対数周波数での重み付き標準偏差と変動係数に変更
        - anti: ピークの鋭さ（周辺との高さ比）を考慮
        - resonance: 谷の鋭さと低周波数での重要度を考慮
    """

    z_pdn_batch = xp.asarray(z_pdn_batch)
    if z_pdn_batch.ndim == 1:
        z_pdn_batch = z_pdn_batch[xp.newaxis, :]

    count_vectors = xp.asarray(count_vectors)
    if count_vectors.ndim == 1:
        count_vectors = count_vectors[xp.newaxis, :]

    if z_pdn_batch.shape[0] != count_vectors.shape[0]:
        raise ValueError("z_pdn_batch と count_vectors のバッチ次元が一致しません")

    metadata = eval_metadata or {}
    freq_indices = metadata.get('indices')
    df_log = metadata.get('df_log')
    n_eval = int(metadata.get('n_eval', 0)) if freq_indices is not None else -1

    if freq_indices is not None:
        freq_indices = xp.asarray(freq_indices, dtype=getattr(xp, "int64", np.int64))
        n_eval = int(n_eval if n_eval >= 0 else freq_indices.shape[0])
        if df_log is not None:
            df_log = xp.asarray(df_log)
    else:
        eval_mask = xp.asarray(eval_mask, dtype=bool)
        freq_indices = xp.where(eval_mask)[0]
        n_eval = int(freq_indices.size)
        df_log = None

    target_curve = xp.asarray(target_curve)
    float_dtype = z_pdn_batch.real.dtype

    if df_log is None and n_eval >= 2:
        if f_grid is None:
            f_eval = xp.arange(n_eval, dtype=float_dtype) + 1.0
        else:
            f_eval = xp.asarray(f_grid)[freq_indices].astype(float_dtype, copy=False)
        df_log = xp.log10(f_eval[1:] / f_eval[:-1])
        df_log = xp.where(xp.isfinite(df_log) & (df_log > 0), df_log, 0.0)

    total_parts = count_vectors.sum(axis=1)
    score_parts = safe_divide(
        total_parts.astype(float_dtype),
        max(1, config.max_total_parts),
        0.0,
        xp,
    )
    score_parts = xp.minimum(score_parts, 1.0)

    if n_eval == 0:
        penalty = xp.full(z_pdn_batch.shape[0], 1.0, dtype=float_dtype)
        zeros = xp.zeros_like(penalty)
        return {
            'max': penalty.copy(),
            'area': penalty.copy(),
            'mean': penalty.copy(),
            'anti': penalty.copy(),
            'flat': penalty.copy(),
            'under': zeros,
            'parts': score_parts,
        }

    z_eval = xp.abs(z_pdn_batch)[:, freq_indices]
    t_eval = xp.maximum(target_curve[freq_indices], 1e-18)
    ratio = z_eval / t_eval
    ratio_clipped = xp.maximum(ratio, 1e-12)

    # 最大インピーダンス比（目標インピーダンスに対する最大値）
    score_max = xp.minimum(ratio_clipped.max(axis=1), 50.0)

    # 平均インピーダンス比（対数周波数での重み付き平均）
    # 対数周波数スケールでは各周波数点の重要度が異なるため、周波数区間の対数幅で重み付け
    weights_normalized = None  # 初期化（flatスコアでも使用）

    if n_eval >= 2:
        # 各周波数点の重み = その周波数区間の対数幅の割合
        weights = xp.zeros(n_eval, dtype=float_dtype)
        weights[0] = df_log[0] if df_log.size > 0 else 0.0
        weights[1:-1] = 0.5 * (df_log[:-1] + df_log[1:]) if df_log.size > 1 else 0.0
        weights[-1] = df_log[-1] if df_log.size > 0 else 0.0

        total_weight = xp.maximum(weights.sum(), 1e-12)
        weights_normalized = weights / total_weight

        # 重み付き平均
        score_mean_raw = (ratio_clipped * weights_normalized).sum(axis=1)
    else:
        # 点数が少ない場合は単純平均
        score_mean_raw = ratio_clipped.mean(axis=1)

    score_mean = xp.minimum(score_mean_raw, 50.0)

    # 目標超過面積（area）：目標インピーダンスを超えた部分の面積
    # 台形積分で対数周波数スケールの面積を計算
    excess = xp.maximum(ratio_clipped - 1.0, 0.0)
    if n_eval >= 2:
        traps = 0.5 * (excess[:, :-1] + excess[:, 1:]) * df_log
        score_area = traps.sum(axis=1)

        # 目標未達面積（under）：目標インピーダンスを下回った部分の面積
        # これは良い特性なので、負の重み（weight_under < 0）で評価される
        # スコア自体は正の値として計算し、重み付け時に負にすることで総スコアを改善
        under_ratio = xp.maximum(1.0 - ratio_clipped, 0.0)
        under_traps = 0.5 * (under_ratio[:, :-1] + under_ratio[:, 1:]) * df_log
        score_under = under_traps.sum(axis=1)
    else:
        score_area = excess[:, 0]
        score_under = xp.maximum(1.0 - ratio_clipped[:, 0], 0.0)

    score_area = xp.minimum(score_area, 50.0)
    score_under = xp.minimum(score_under, 50.0)

    # アンチレゾナンス評価（ピークの高さと鋭さを考慮）
    if n_eval > 2:
        center = ratio_clipped[:, 1:-1]
        left = ratio_clipped[:, :-2]
        right = ratio_clipped[:, 2:]

        # ローカルピークを検出
        peak_mask = (center > left) & (center > right)
        peak_vals = xp.where(peak_mask, center, 0.0)
        peak_excess = xp.maximum(peak_vals - 1.0, 0.0)

        # ignore_safe_anti_resonanceフラグ：目標以下のピークを無視
        if getattr(config, 'ignore_safe_anti_resonance', False):
            peak_excess = xp.where(peak_vals <= 1.0, 0.0, peak_excess)

        # ピークの鋭さを評価（周辺との高さ比）
        # sharpness = (center - neighbor_mean) / neighbor_mean
        neighbor_mean = 0.5 * (left + right)
        sharpness = safe_divide(
            center - neighbor_mean,
            xp.maximum(neighbor_mean, 1e-12),
            0.0,
            xp
        )
        # 鋭さは正の値のみ考慮（ピークの場合のみ）
        sharpness = xp.where(peak_mask, xp.maximum(sharpness, 0.0), 0.0)

        # スコア計算：ピーク高さに鋭さの重みを加える
        # ピークが鋭いほど悪い特性として評価
        sharpness_weight = 0.5  # 鋭さの重み係数
        weighted_peak_excess = peak_excess * (1.0 + sharpness_weight * sharpness)

        # 総スコア = 重み付きピーク高さの合計
        score_anti = weighted_peak_excess.sum(axis=1)
    else:
        score_anti = xp.zeros(ratio_clipped.shape[0], dtype=float_dtype)

    score_anti = xp.minimum(score_anti, 50.0)

    # 平滑さ評価（対数周波数での重み付き標準偏差と変動係数）
    # 周波数特性の平坦性を評価：変動が大きいほど悪い特性
    if n_eval >= 2 and weights_normalized is not None:
        # 対数周波数での重み付き標準偏差
        weighted_mean = score_mean_raw[:, xp.newaxis]  # ブロードキャスト用に次元追加

        # 各点の偏差を計算
        deviation = ratio_clipped - weighted_mean

        # 重み付き分散
        weighted_variance = (weights_normalized * deviation ** 2).sum(axis=1)

        # 重み付き標準偏差
        weighted_std = xp.sqrt(xp.maximum(weighted_variance, 0.0))

        # 変動係数（CV: Coefficient of Variation）
        score_flat = safe_divide(
            weighted_std,
            xp.maximum(score_mean_raw, 1e-12),
            0.0,
            xp,
        )
    else:
        # 点数が少ない場合は単純な変動係数
        score_flat = safe_divide(
            ratio_clipped.std(axis=1),
            xp.maximum(score_mean_raw, 1e-12),
            0.0,
            xp,
        )

    score_flat = xp.minimum(score_flat, 10.0)

    # コンデンサ種類数（正規化）
    num_types = xp.count_nonzero(count_vectors > 0, axis=1).astype(float_dtype, copy=False)
    score_num_types = safe_divide(
        num_types,
        max(1, count_vectors.shape[1]),
        0.0,
        xp,
    )

    # レゾナンス評価（谷の深さ、鋭さ、周波数位置の重要度を考慮）
    if n_eval > 2:
        z_eval_local = xp.abs(z_pdn_batch)[:, freq_indices]
        left = z_eval_local[:, :-2]
        center_abs = z_eval_local[:, 1:-1]
        right = z_eval_local[:, 2:]

        # ローカルミニマムを検出
        neighbor_mean = 0.5 * (left + right)
        local_min_mask = (center_abs < left) & (center_abs < right)

        # 谷の深さ（周辺と中心の比率）
        depth_ratio = safe_divide(neighbor_mean, xp.maximum(center_abs, 1e-18), 1.0, xp) - 1.0
        depth_ratio = xp.where(local_min_mask, xp.maximum(depth_ratio, 0.0), 0.0)

        # 谷の鋭さ（両隣との高さ差の比率）
        # 鋭い谷ほどQ値が高く、問題となりやすい
        left_drop = safe_divide(left - center_abs, xp.maximum(center_abs, 1e-18), 0.0, xp)
        right_drop = safe_divide(right - center_abs, xp.maximum(center_abs, 1e-18), 0.0, xp)
        sharpness = 0.5 * (left_drop + right_drop)
        sharpness = xp.where(local_min_mask, xp.maximum(sharpness, 0.0), 0.0)

        # 周波数位置による重み付け（低周波数の谷ほど問題になりやすい）
        if f_grid is not None:
            f_eval_local = xp.asarray(f_grid)[freq_indices][1:-1]
            f_L_val = float(metadata.get('f_L', freq_indices[0])) if metadata else float(freq_indices[0])
            # 対数スケールでの周波数重み（低周波数: 1.0 → 高周波数: 0.5）
            freq_weight = 1.0 / (1.0 + 0.5 * xp.log10(xp.maximum(f_eval_local / f_L_val, 1.0)))
        else:
            freq_weight = xp.ones_like(depth_ratio[0, :])

        # 総合スコア = 深さ × (1 + 鋭さの重み) × 周波数重み
        sharpness_weight = 0.3  # 鋭さの重み係数
        weighted_depth = depth_ratio * (1.0 + sharpness_weight * sharpness) * freq_weight

        score_resonance = xp.minimum(weighted_depth.sum(axis=1), 50.0)
    else:
        score_resonance = xp.zeros(ratio_clipped.shape[0], dtype=float_dtype)

    # Improvement評価（デカップリングコンデンサなし（Without Decap）との比較）
    # デカップリングコンデンサの追加によってインピーダンスが悪化した部分を検出
    if z_without_decap is not None:
        z_without_decap = xp.asarray(z_without_decap)
        z_base_eval = xp.abs(z_without_decap)[freq_indices]

        # 悪化比率を計算：z_pdn / z_without_decap
        # 1.0より大きい部分 = デカップリングコンデンサによって悪化した部分
        degradation_ratio = safe_divide(z_eval, xp.maximum(z_base_eval, 1e-18), 1.0, xp)
        degradation_excess = xp.maximum(degradation_ratio - 1.0, 0.0)

        # 悪化部分の面積を台形積分で計算
        if n_eval >= 2:
            degrade_traps = 0.5 * (degradation_excess[:, :-1] + degradation_excess[:, 1:]) * df_log
            score_improvement = degrade_traps.sum(axis=1)
        else:
            score_improvement = degradation_excess[:, 0]

        score_improvement = xp.minimum(score_improvement, 50.0).astype(float_dtype, copy=False)

        # 低改善度帯域の評価
        # improvement_ratio = 1.0 - (z_pdn / z_without_decap) で改善度を定義
        # 改善度が閾値未満の帯域はデカップリングコンデンサの効果が不十分
        improvement_ratio = 1.0 - degradation_ratio
        threshold = float(getattr(config, 'low_improvement_threshold', 0.15))

        # 改善度が閾値未満の部分を検出（0.0 <= improvement_ratio < threshold）
        # 悪化部分（improvement_ratio < 0）は既にimprovementスコアで評価済みなので除外
        low_improvement_mask = (improvement_ratio >= 0.0) & (improvement_ratio < threshold)
        low_improvement_value = xp.where(low_improvement_mask, threshold - improvement_ratio, 0.0)

        # 低改善度帯域の面積を台形積分で計算
        if n_eval >= 2:
            low_imp_traps = 0.5 * (low_improvement_value[:, :-1] + low_improvement_value[:, 1:]) * df_log
            score_low_improvement = low_imp_traps.sum(axis=1)
        else:
            score_low_improvement = low_improvement_value[:, 0]

        score_low_improvement = xp.minimum(score_low_improvement, 50.0).astype(float_dtype, copy=False)
    else:
        # Without Decapデータがない場合は評価なし
        score_improvement = xp.zeros(ratio_clipped.shape[0], dtype=float_dtype)
        score_low_improvement = xp.zeros(ratio_clipped.shape[0], dtype=float_dtype)

    return {
        'max': score_max.astype(float_dtype, copy=False),
        'area': score_area.astype(float_dtype, copy=False),
        'mean': score_mean.astype(float_dtype, copy=False),
        'anti': score_anti.astype(float_dtype, copy=False),
        'flat': score_flat.astype(float_dtype, copy=False),
        'under': score_under.astype(float_dtype, copy=False),
        'parts': score_parts.astype(float_dtype, copy=False),
        'num_types': score_num_types.astype(float_dtype, copy=False),
        'resonance': score_resonance.astype(float_dtype, copy=False),
        'improvement': score_improvement,
        'low_improvement': score_low_improvement,
    }


def calculate_score_components(
    z_pdn: np.ndarray,
    target_curve: np.ndarray,
    eval_mask: np.ndarray,
    count_vector: np.ndarray,
    config: UserConfig,
    xp: Any = np,
    f_grid: Optional[np.ndarray] = None,
    eval_metadata: Optional[Dict[str, Any]] = None,
    z_without_decap: Optional[Any] = None,
) -> Dict[str, float]:
    """単一組み合わせのスコアコンポーネント"""

    batch = calculate_score_components_batch(
        z_pdn,
        target_curve,
        eval_mask,
        count_vector,
        config,
        xp,
        f_grid,
        eval_metadata=eval_metadata,
        z_without_decap=z_without_decap,
    )

    return {
        key: float(transfer_to_host(val)[0])
        for key, val in batch.items()
    }


def evaluate_combinations(
    z_pdn_batch: np.ndarray,
    target_curve: np.ndarray,
    eval_mask: np.ndarray,
    count_vectors: np.ndarray,
    config: UserConfig,
    xp: Any = np,
    f_grid: Optional[np.ndarray] = None,
    eval_metadata: Optional[Dict[str, Any]] = None,
    z_without_decap: Optional[Any] = None,
) -> np.ndarray:
    """
    バッチ評価（小さいほど良い）。
    """
    comps = calculate_score_components_batch(
        z_pdn_batch,
        target_curve,
        eval_mask,
        count_vectors,
        config,
        xp,
        f_grid,
        eval_metadata=eval_metadata,
        z_without_decap=z_without_decap,
    )

    total = (
        config.weight_max   * comps['max'] +
        config.weight_area  * comps['area'] +
        config.weight_mean  * comps['mean'] +
        config.weight_anti  * comps['anti'] +
        config.weight_flat  * comps['flat'] +
        config.weight_parts * comps['parts'] +
        config.weight_under * comps['under'] +
        config.weight_num_types * comps['num_types'] +
        config.weight_resonance * comps['resonance'] +
        config.weight_improvement * comps['improvement'] +
        config.weight_low_improvement * comps['low_improvement']
    )

    return total.astype(xp.float32, copy=False)


def extract_top_k(z_pdn_batch: np.ndarray,
                 scores: np.ndarray,
                 count_vectors: np.ndarray,
                 k: int,
                 xp: Any = np) -> List[Dict]:
    """
    上位k個の組み合わせを抽出
    
    Args:
        z_pdn_batch: PDNインピーダンスバッチ
        scores: スコア配列
        count_vectors: カウントベクトル
        k: 抽出数
        xp: バックエンドモジュール
    
    Returns:
        上位k個の結果リスト
    """
    n_batch = len(scores)
    k = min(k, n_batch)
    
    if k == 0:
        return []
    
    # スコアでソート（昇順）
    if xp is np:
        # NumPyの場合
        if k < n_batch:
            indices = np.argpartition(scores, k - 1)[:k]
            indices = indices[np.argsort(scores[indices])]
        else:
            indices = np.argsort(scores)[:k]
    else:
        # CuPyの場合（フルソート使用）。以降の処理はCPU側で実施
        sorted_indices = xp.argsort(scores)[:k]
        indices = transfer_to_host(sorted_indices)

    indices = np.asarray(indices, dtype=np.int64)
    
    # 結果を構築
    top_k_results = []
    for rank, idx in enumerate(indices, start=1):
        idx_int = int(idx)
        if hasattr(scores, '__cuda_array_interface__'):
            score_value = transfer_to_host(scores[idx_int])
        else:
            score_value = scores[idx_int]

        if np.isscalar(score_value):
            score_float = float(score_value)
        else:
            score_float = float(np.asarray(score_value).item())

        result = {
            'count_vector': count_vectors[idx_int],
            'z_pdn': z_pdn_batch[idx_int],
            'total_score': score_float,
            'rank': rank
        }
        top_k_results.append(result)
    
    return top_k_results


def monte_carlo_evaluation(top_k_results: List[Dict],
                          capacitor_impedances: Dict[str, np.ndarray],
                          capacitor_indices: np.ndarray,
                          f_grid: np.ndarray,
                          eval_mask: np.ndarray,
                          target_mask: np.ndarray,
                          config: UserConfig,
                          xp: Any = np,
                          pdn_assets: Optional[Dict[str, Any]] = None,
                          eval_metadata: Optional[Dict[str, Any]] = None,
                          z_without_decap: Optional[Any] = None) -> np.ndarray:
    """
    Monte Carlo法でロバスト性を評価
    
    Args:
        top_k_results: top_k結果リスト
        capacitor_impedances: コンデンサインピーダンス
        capacitor_indices: コンデンサインデックス
        f_grid: 周波数グリッド
        eval_mask: 評価帯域マスク
        target_mask: 目標マスク
        config: ユーザー設定
        xp: バックエンドモジュール
        pdn_assets: 事前計算済みPDN要素（オプション）
    
    Returns:
        MC最悪値スコア配列
    """
    if not config.mc_enable or len(top_k_results) == 0:
        return xp.zeros(len(top_k_results), dtype=xp.float32 if xp is not np else np.float32)
    
    prepared = pdn_assets if pdn_assets is not None else prepare_pdn_components(
        capacitor_impedances,
        config,
        f_grid,
        xp,
        getattr(config, 'dtype_c', 'complex64'),
    )

    mc_worst_scores: List[float] = []
    
    for result in top_k_results:
        count_vector = result['count_vector']
        count_vector_backend = xp.asarray(count_vector)
        
        # Monte Carloサンプリング
        z_pdn_mc = calculate_pdn_impedance_monte_carlo(
            count_vector,
            capacitor_impedances,
            capacitor_indices,
            f_grid,
            config,
            config.mc_samples,
            xp,
            prepared=prepared,
        )
        
        # 各サンプルのスコア計算
        repeated_counts = xp.broadcast_to(
            count_vector_backend[xp.newaxis, :],
            (config.mc_samples, count_vector_backend.shape[0])
        )
        mc_scores = evaluate_combinations(
            z_pdn_mc,
            target_mask,
            eval_mask,
            repeated_counts,
            config,
            xp,
            f_grid,
            eval_metadata=eval_metadata,
            z_without_decap=z_without_decap,
        )

        # 最悪値（最大スコア）を記録
        worst_score_val = transfer_to_host(xp.max(mc_scores))
        worst_score = float(np.asarray(worst_score_val).item() if not np.isscalar(worst_score_val) else worst_score_val)
        mc_worst_scores.append(worst_score)
        
        # 統計情報をログ
        mean_val = transfer_to_host(xp.mean(mc_scores))
        std_val = transfer_to_host(xp.std(mc_scores))
        mean_float = float(np.asarray(mean_val).item() if not np.isscalar(mean_val) else mean_val)
        std_float = float(np.asarray(std_val).item() if not np.isscalar(std_val) else std_val)
        logger.debug(f"MC評価 - 平均: {mean_float:.6f}, 最悪: {worst_score:.6f}, std: {std_float:.6f}")
    
    return xp.asarray(mc_worst_scores, dtype=xp.float32 if xp is not np else np.float32)


def format_combination_name(count_vector: np.ndarray,
                          capacitor_names: List[str]) -> str:
    """
    組み合わせを文字列表現にフォーマット

    Args:
        count_vector: カウントベクトル（各コンデンサの使用個数）
        capacitor_names: コンデンサ名リスト

    Returns:
        フォーマット済み文字列

    Examples:
        >>> count_vec = np.array([2, 0, 3, 1])
        >>> cap_names = ["C1_0.1uF", "C2_0.22uF", "C3_1uF", "C4_10uF"]
        >>> format_combination_name(count_vec, cap_names)
        '(C1_0.1uF)x2 + (C3_1uF)x3 + (C4_10uF) (Total: 6)'

        >>> count_vec = np.array([1, 0, 0, 0])
        >>> cap_names = ["C1_0.1uF", "C2_0.22uF", "C3_1uF", "C4_10uF"]
        >>> format_combination_name(count_vec, cap_names)
        '(C1_0.1uF) (Total: 1)'

        >>> count_vec = np.array([0, 0, 0, 0])
        >>> cap_names = ["C1_0.1uF", "C2_0.22uF", "C3_1uF", "C4_10uF"]
        >>> format_combination_name(count_vec, cap_names)
        'Empty'
    """
    parts = []
    total = 0

    for i, name in enumerate(capacitor_names):
        count = int(count_vector[i])
        if count > 0:
            if count == 1:
                parts.append(f"({name})")
            else:
                parts.append(f"({name})x{count}")
            total += count

    if parts:
        combo_str = " + ".join(parts)
        # return f"{combo_str} (Total: {total})"
        return combo_str
    else:
        return "Empty"


def summarize_results(top_k_results: List[Dict],
                     capacitor_names: List[str],
                     config: UserConfig) -> List[Dict]:
    """
    結果をサマリー形式に整理
    
    Args:
        top_k_results: top_k結果
        capacitor_names: コンデンサ名リスト
        config: ユーザー設定
    
    Returns:
        整理された結果リスト
    """
    summaries = []
    
    for result in top_k_results:
        count_vector = result['count_vector']
        
        # CPU変換（必要な場合）
        if hasattr(count_vector, '__cuda_array_interface__'):
            count_vector = transfer_to_host(count_vector)
        
        summary = {
            'rank': result['rank'],
            'combination': format_combination_name(count_vector, capacitor_names),
            'total_score': result['total_score'],
            'count_vector': count_vector,
            'total_parts': int(np.sum(count_vector))
        }
        
        # MC最悪値がある場合
        if 'mc_worst_score' in result:
            summary['mc_worst_score'] = result['mc_worst_score']
        
        summaries.append(summary)
    
    return summaries


def compare_combinations(combo1: Dict, combo2: Dict,
                        tolerance: float = 1e-6) -> bool:
    """
    2つの組み合わせが同一かチェック
    
    Args:
        combo1: 組み合わせ1
        combo2: 組み合わせ2
        tolerance: 許容誤差
    
    Returns:
        同一ならTrue
    """
    # カウントベクトルの比較
    cv1 = combo1.get('count_vector')
    cv2 = combo2.get('count_vector')
    
    if cv1 is None or cv2 is None:
        return False
    
    # CPU変換
    cv1 = transfer_to_host(cv1)
    cv2 = transfer_to_host(cv2)
    
    # 配列比較
    if cv1.shape != cv2.shape:
        return False
    
    return np.allclose(cv1, cv2, rtol=tolerance, atol=tolerance)
