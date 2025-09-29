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


def calculate_score_components(
    z_pdn: np.ndarray,
    target_curve: np.ndarray,   # target_mask という名前ですが実体は閾値カーブなので改名
    eval_mask: np.ndarray,
    count_vector: np.ndarray,
    config: UserConfig,
    xp: Any = np,
    f_grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    スコアコンポーネント（比ベース, 対数周波数台形則）を計算
    - 面積: r = |Z|/Zt の超過量 (r-1)+ を log10(f) 上で台形積分
    - dB 指標を使いたい場合は excess_db を有効化
    """
    z_abs = xp.abs(z_pdn)

    # マスク適用
    if eval_mask is None or eval_mask.dtype != bool:
        raise ValueError("eval_mask は bool 配列である必要があります")
    z_eval = z_abs[eval_mask]
    t_eval = target_curve[eval_mask]

    if f_grid is None:
        # 既存互換: 等間隔とみなし Δlogf = 1 の重み（※非推奨）
        f_eval = xp.arange(len(z_eval), dtype=float) + 1.0
        warn_no_f = True
    else:
        f_eval = f_grid[eval_mask].astype(float)
        warn_no_f = False

    if len(z_eval) == 0:
        # 何もない時は重ペナルティ
        return {'max':1.0,'area':1.0,'mean':1.0,'anti':1.0,'flat':1.0,'under':0.0,'parts':1.0}

    # 0 除算対策
    t_eval = xp.maximum(t_eval, 1e-18)

    # 比ベース
    ratio = z_eval / t_eval         # r = |Z|/Zt
    ratio_clipped = xp.maximum(ratio, 1e-12)

    # ---- コンポーネント ----
    # 1) 最大比
    score_max = float(xp.max(ratio_clipped))

    # 2) 面積（比超過）: (r-1)+ を log10(f) 上で台形積分
    #    A = ∑ 0.5 * (e[i] + e[i+1]) * Δlog10(f[i->i+1])
    excess = xp.maximum(ratio_clipped - 1.0, 0.0)
    if len(excess) >= 2:
        # Δlog10(f)
        df_log = xp.log10(f_eval[1:] / f_eval[:-1])
        # 非等間隔/単調性チェック
        df_log = xp.where(xp.isfinite(df_log) & (df_log > 0), df_log, 0.0)
        traps = 0.5 * (excess[:-1] + excess[1:]) * df_log
        area_raw = xp.sum(traps)
    else:
        area_raw = excess[0] if len(excess) == 1 else 0.0
    score_area = float(area_raw)

    # 3) 平均比
    score_mean = float(xp.mean(ratio_clipped))

    # 4) アンチレゾナンスの強さ（超過ピークの数×高さの代表値）
    anti = 0.0
    if len(ratio_clipped) > 2:
        is_peak = (ratio_clipped[1:-1] > ratio_clipped[:-2]) & (ratio_clipped[1:-1] > ratio_clipped[2:])
        if xp is np:
            peak_vals = ratio_clipped[1:-1][is_peak]
        else:
            idx = xp.where(is_peak)[0]
            peak_vals = ratio_clipped[1:-1][idx] if len(idx) > 0 else xp.array([], dtype=ratio_clipped.dtype)
        peak_excess = peak_vals[peak_vals > 1.0]
        if len(peak_excess) > 0:
            anti = float(len(peak_excess) * xp.mean(peak_excess - 1.0))
    score_anti = anti

    # 5) フラットネス（変動/平均）: 比ベースで評価
    mu = xp.mean(ratio_clipped)
    score_flat = float((xp.std(ratio_clipped) / mu) if mu > 0 else 0.0)

    # 6) “under” は報酬（目標以下のマージン）
    under_ratio = xp.maximum(1.0 - ratio_clipped, 0.0)
    if len(under_ratio) >= 2:
        df_log = xp.log10(f_eval[1:] / f_eval[:-1])
        df_log = xp.where(xp.isfinite(df_log) & (df_log > 0), df_log, 0.0)
        traps = 0.5 * (under_ratio[:-1] + under_ratio[1:]) * df_log
        under_area = float(xp.sum(traps))
    else:
        under_area = float(under_ratio[0] if len(under_ratio) == 1 else 0.0)
    # 正規化（スケール設定：小さくし過ぎない）
    score_under = under_area  # ←後で合計スコアから減点する

    # 7) 部品点数ペナルティ
    total_parts = int(xp.sum(count_vector))
    score_parts = float(total_parts / max(1, config.max_total_parts))

    # クリップは必要最小限（過度に潰さない）
    def soft_clip(x, hi):
        return min(float(x), hi)

    return {
        'max'  : soft_clip(score_max,  50.0),    # r の上限クリップ
        'area' : soft_clip(score_area, 50.0),    # 面積の上限クリップ
        'mean' : soft_clip(score_mean, 50.0),
        'anti' : soft_clip(score_anti, 50.0),
        'flat' : soft_clip(score_flat, 10.0),
        'under': soft_clip(score_under, 50.0),
        'parts': soft_clip(score_parts, 1.0),
    }


def evaluate_combinations(
    z_pdn_batch: np.ndarray,
    target_curve: np.ndarray,
    eval_mask: np.ndarray,
    count_vectors: np.ndarray,
    config: UserConfig,
    xp: Any = np,
    f_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    バッチ評価（小さいほど良い）。
    """
    n_batch = len(z_pdn_batch)
    scores = xp.zeros(n_batch, dtype=xp.float32)

    for i in range(n_batch):
        comp = calculate_score_components(
            z_pdn=z_pdn_batch[i],
            target_curve=target_curve,
            eval_mask=eval_mask,
            count_vector=count_vectors[i],
            config=config,
            xp=xp,
            f_grid=f_grid,
        )

        total = (
            config.weight_max  * comp['max']  +
            config.weight_area * comp['area'] +
            config.weight_mean * comp['mean'] +
            config.weight_anti * comp['anti'] +
            config.weight_flat * comp['flat'] +
            config.weight_parts* comp['parts'] +
            config.weight_under* comp['under']
        )

        scores[i] = float(total)

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