"""
評価モジュール
スコア計算、Monte Carloロバスト評価、top_k抽出、結果評価、バッチ処理
"""

import traceback
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# 絶対パスでインポート
from deca_auto.config import UserConfig
from deca_auto.utils import logger, transfer_to_device
from deca_auto.pdn import calculate_pdn_impedance_monte_carlo


def calculate_score_components(z_pdn: np.ndarray,
                              target_mask: np.ndarray,
                              eval_mask: np.ndarray,
                              count_vector: np.ndarray,
                              config: UserConfig,
                              xp: Any = np) -> Dict[str, float]:
    """
    スコアコンポーネントを計算
    
    Args:
        z_pdn: PDNインピーダンス (複素数)
        target_mask: 目標マスク
        eval_mask: 評価帯域マスク
        count_vector: コンデンサカウントベクトル
        config: ユーザー設定
        xp: バックエンドモジュール
    
    Returns:
        スコアコンポーネントの辞書
    """
    # 絶対値
    z_abs = xp.abs(z_pdn)
    
    # 評価帯域内のみ
    z_eval = z_abs[eval_mask]
    target_eval = target_mask[eval_mask]
    
    if len(z_eval) == 0:
        logger.warning("評価帯域内にデータがありません")
        return {
            'max': 1.0,  # ペナルティとして高い値
            'area': 1.0,
            'mean': 1.0,
            'anti': 1.0,
            'flat': 1.0,
            'under': 0.0,
            'parts': 0.0
        }
    
    # 1. 最大値 (正規化: target の倍数)
    score_max = float(xp.max(z_eval) / xp.mean(target_eval))
    
    # 2. 目標マスクに対する超過面積（対数スケールで台形積分）
    excess = xp.maximum(z_eval - target_eval, 0)
    # 対数スケールでの積分のため、周波数の対数差分を重みとする
    freq_eval = xp.arange(len(z_eval))  # 簡易的にインデックスを使用
    if len(freq_eval) > 1:
        log_weights = xp.ones(len(freq_eval))
        # 対数差分の近似（ゼロ除算を回避）
        log_weights[1:] = xp.log10(freq_eval[1:] / xp.maximum(freq_eval[:-1], 0.1) + 1)
        score_area = float(xp.sum(excess * log_weights))
    else:
        score_area = float(excess[0] if len(excess) > 0 else 0)
    
    # 3. 平均値 (正規化)
    score_mean = float(xp.mean(z_eval) / xp.mean(target_eval))
    
    # 4. アンチレゾナンス（ピーク検出）
    # ローカルピークの検出（簡易版）
    if len(z_eval) > 2:
        # ピーク検出（前後より大きい点）
        is_peak = (z_eval[1:-1] > z_eval[:-2]) & (z_eval[1:-1] > z_eval[2:])
        if xp is np:
            # NumPyの場合
            peak_heights = z_eval[1:-1][is_peak]
            # 目標を超えるピークの抽出
            target_mid = target_eval[1:-1][is_peak]
            excess_peaks = peak_heights[peak_heights > target_mid]
        else:
            # CuPyの場合（ブールインデックスが異なる可能性）
            peak_indices = xp.where(is_peak)[0]
            if len(peak_indices) > 0:
                peak_heights = z_eval[1:-1][peak_indices]
                target_mid = target_eval[1:-1][peak_indices]
                excess_mask = peak_heights > target_mid
                excess_peaks = peak_heights[excess_mask]
            else:
                excess_peaks = xp.array([])
        
        if len(excess_peaks) > 0:
            score_anti = float(len(excess_peaks) * xp.mean(excess_peaks) / xp.mean(target_eval))
        else:
            score_anti = 0.0
    else:
        score_anti = 0.0
    
    # 5. フラットネス（標準偏差/平均）
    score_flat = float(xp.std(z_eval) / xp.mean(z_eval))
    
    # 6. 余裕面積（目標を下回る面積、負のスコア）
    under = xp.maximum(target_eval - z_eval, 0)
    score_under = float(xp.sum(under))
    
    # 7. 部品点数ペナルティ
    total_parts = int(xp.sum(count_vector))
    score_parts = float(total_parts / config.max_total_parts)
    
    # 正規化（0-1範囲）
    scores = {
        'max': min(score_max, 10.0) / 10.0,  # 10倍を上限
        'area': min(score_area, 100.0) / 100.0,  # 適切な正規化
        'mean': min(score_mean, 10.0) / 10.0,
        'anti': min(score_anti, 10.0) / 10.0,
        'flat': min(score_flat, 1.0),  # すでに0-1範囲
        'under': min(score_under, 100.0) / 100.0,
        'parts': score_parts  # すでに0-1範囲
    }
    
    return scores


def evaluate_combinations(z_pdn_batch: np.ndarray,
                         target_mask: np.ndarray,
                         eval_mask: np.ndarray,
                         count_vectors: np.ndarray,
                         config: UserConfig,
                         xp: Any = np) -> np.ndarray:
    """
    組み合わせのバッチ評価
    
    Args:
        z_pdn_batch: PDNインピーダンスバッチ (N_batch, N_freq)
        target_mask: 目標マスク
        eval_mask: 評価帯域マスク
        count_vectors: カウントベクトル (N_batch, N_caps)
        config: ユーザー設定
        xp: バックエンドモジュール
    
    Returns:
        スコア配列 (N_batch,)
    """
    n_batch = len(z_pdn_batch)
    scores = xp.zeros(n_batch, dtype=xp.float32)
    
    # バッチ処理
    for i in range(n_batch):
        # 各コンポーネント計算
        components = calculate_score_components(
            z_pdn_batch[i],
            target_mask,
            eval_mask,
            count_vectors[i],
            config,
            xp
        )
        
        # 重み付き線形結合
        total_score = (
            config.weight_max * components['max'] +
            config.weight_area * components['area'] +
            config.weight_mean * components['mean'] +
            config.weight_anti * components['anti'] +
            config.weight_flat * components['flat'] +
            config.weight_under * components['under'] +
            config.weight_parts * components['parts']
        )
        
        scores[i] = total_score
    
    return scores


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
            sorted_indices = np.argpartition(scores, k-1)[:k]
            sorted_indices = sorted_indices[np.argsort(scores[sorted_indices])]
        else:
            sorted_indices = np.argsort(scores)[:k]
    else:
        # CuPyの場合（フルソート使用）
        sorted_indices = xp.argsort(scores)[:k]
    
    # 結果を構築
    top_k_results = []
    for idx in sorted_indices:
        result = {
            'count_vector': count_vectors[idx],
            'z_pdn': z_pdn_batch[idx],
            'total_score': float(scores[idx]),
            'rank': len(top_k_results) + 1
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
                          xp: Any = np) -> np.ndarray:
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
    
    Returns:
        MC最悪値スコア配列
    """
    if not config.mc_enable or len(top_k_results) == 0:
        return xp.zeros(len(top_k_results))
    
    mc_worst_scores = []
    
    for result in top_k_results:
        count_vector = result['count_vector']
        
        # Monte Carloサンプリング
        z_pdn_mc = calculate_pdn_impedance_monte_carlo(
            count_vector,
            capacitor_impedances,
            capacitor_indices,
            f_grid,
            config,
            config.mc_samples,
            xp
        )
        
        # 各サンプルのスコア計算
        mc_scores = evaluate_combinations(
            z_pdn_mc,
            target_mask,
            eval_mask,
            xp.tile(count_vector[xp.newaxis, :], (config.mc_samples, 1)),
            config,
            xp
        )
        
        # 最悪値（最大スコア）を記録
        worst_score = float(xp.max(mc_scores))
        mc_worst_scores.append(worst_score)
        
        # 統計情報をログ
        logger.debug(f"MC評価 - 平均: {xp.mean(mc_scores):.6f}, "
                    f"最悪: {worst_score:.6f}, "
                    f"std: {xp.std(mc_scores):.6f}")
    
    return xp.array(mc_worst_scores)


def format_combination_name(count_vector: np.ndarray,
                          capacitor_names: List[str]) -> str:
    """
    組み合わせを文字列表現にフォーマット
    
    Args:
        count_vector: カウントベクトル
        capacitor_names: コンデンサ名リスト
    
    Returns:
        フォーマット済み文字列
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
        return f"{combo_str} (Total: {total})"
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
            count_vector = transfer_to_device(count_vector, np)
        
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
    cv1 = transfer_to_device(cv1, np)
    cv2 = transfer_to_device(cv2, np)
    
    # 配列比較
    if cv1.shape != cv2.shape:
        return False
    
    return np.allclose(cv1, cv2, rtol=tolerance, atol=tolerance)