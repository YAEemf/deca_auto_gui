import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import xlsxwriter

from deca_auto.config import UserConfig, WEIGHT_FIELDS
from deca_auto.utils import (
    logger, ensure_numpy, create_decimated_indices, get_custom_mask_freq_range,
)
from deca_auto.evaluator import format_combination_name


COMPONENT_KEYS = tuple(field.removeprefix("weight_") for field in WEIGHT_FIELDS)
COMPONENT_HEADERS = [
    "MC Worst" if key == "mc_worst" else key.replace("_", " ").title()
    for key in COMPONENT_KEYS
]


ZPDN_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
TARGET_MASK_COLOR = "#000000"
WITHOUT_DECAP_COLOR = "#555555"


def export_to_excel(results: Dict, config: UserConfig) -> bool:
    """
    結果をExcelファイルに出力
    
    Args:
        results: 最適化結果
        config: ユーザー設定
    
    Returns:
        成功フラグ
    """
    try:
        # 出力ディレクトリ作成
        output_dir = Path(config.excel_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"{config.excel_name}_{timestamp}.xlsx"
        filepath = output_dir / filename
        
        # Excelワークブック作成
        workbook = xlsxwriter.Workbook(str(filepath), {'nan_inf_to_errors': True})
        
        # フォーマット定義
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'num_format': '0.000E+00'
        })
        
        # 1. サマリーシート
        summary_sheet = workbook.add_worksheet('Summary')
        write_summary_sheet(summary_sheet, results, config, header_format, cell_format)
        
        # 2. Top-k詳細シート
        detail_sheet = workbook.add_worksheet('Top-k Details')
        write_detail_sheet(detail_sheet, results, config, header_format,
                          cell_format, number_format)

        # 3. |Z_c|データシート
        zc_data_sheet = workbook.add_worksheet('|Z_c| Data')
        write_zc_impedance_data(zc_data_sheet, results, header_format, number_format)

        # 4. PDNインピーダンスデータシート
        data_sheet = workbook.add_worksheet('Impedance Data')
        write_impedance_data(data_sheet, results, config, header_format, number_format)

        # 5. グラフシート
        chart_sheet = workbook.add_worksheet('Charts')
        create_zc_chart(workbook, chart_sheet, zc_data_sheet, results, config)
        create_impedance_chart(workbook, chart_sheet, data_sheet, results, config)

        # 6. 設定シート
        config_sheet = workbook.add_worksheet('Configuration')
        write_config_sheet(config_sheet, config, header_format, cell_format)
        
        # ワークブック保存
        workbook.close()
        
        logger.info(f"Excel出力完了: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Excel出力エラー: {e}")
        traceback.print_exc()
        return False


def write_summary_sheet(worksheet, results: Dict, config: UserConfig,
                       header_format, cell_format):
    """サマリーシートを作成"""
    
    headers = ["Rank", "Combination", "Total Score", "Parts Count", "Types Count", *COMPONENT_HEADERS]
    last_col = len(headers) - 1

    # タイトル
    worksheet.merge_range(0, 0, 0, last_col, "PDN Impedance Optimization Results", header_format)
    
    # 基本情報
    row = 2
    worksheet.write(row, 0, "Date:", cell_format)
    worksheet.write(row, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cell_format)
    
    row += 1
    worksheet.write(row, 0, "Backend:", cell_format)
    worksheet.write(row, 1, results.get('backend', 'Unknown'), cell_format)
    
    if results.get('gpu_info'):
        row += 1
        worksheet.write(row, 0, "GPU:", cell_format)
        worksheet.write(row, 1, results['gpu_info']['name'], cell_format)
    
    # Top-kサマリー
    row += 2
    worksheet.merge_range(row, 0, row, last_col, "Top-k Results Summary", header_format)
    
    row += 1
    worksheet.write_row(row, 0, headers, header_format)
    
    # Top-k結果
    top_k = results.get('top_k_results', [])
    cap_names = results.get('capacitor_names', [])
    
    for i, result in enumerate(top_k[:config.top_k]):
        row += 1
        count_vec = ensure_numpy(result['count_vector'])
        combo_str = format_combination_name(count_vec, cap_names)
        
        worksheet.write(row, 0, result.get('rank', i+1), cell_format)
        worksheet.write(row, 1, combo_str, cell_format)
        worksheet.write(row, 2, f"{result['total_score']:.6f}", cell_format)
        worksheet.write(row, 3, int(np.sum(count_vec)), cell_format)
        worksheet.write(row, 4, int(np.count_nonzero(count_vec)), cell_format)

        _write_score_components(worksheet, row, 5, result, cell_format)
    
    # 列幅調整
    worksheet.set_column(0, 0, 10)
    worksheet.set_column(1, 1, 60)
    worksheet.set_column(2, last_col, 15)


def _write_score_components(worksheet, row, first_col, result, cell_format):
    """評価済みのスコア内訳を設定の重みと同じ順序で出力する。"""
    components = result.get('components') or {}
    for col, key in enumerate(COMPONENT_KEYS, first_col):
        worksheet.write(row, col, components.get(key, "N/A"), cell_format)


def write_detail_sheet(worksheet, results: Dict, config: UserConfig,
                     header_format, cell_format, number_format):
    """詳細データシートを作成"""
    headers = ["Rank", "Combination", "Total", *COMPONENT_HEADERS]
    last_col = len(headers) - 1
    worksheet.merge_range(0, 0, 0, last_col, "Detailed Score Components", header_format)
    worksheet.write_row(1, 0, headers, header_format)

    top_k = results.get('top_k_results', [])
    cap_names = results.get('capacitor_names', [])
    for i, result in enumerate(top_k[:config.top_k]):
        row = i + 2
        count_vec = ensure_numpy(result['count_vector'])
        combo_str = format_combination_name(count_vec, cap_names)
        worksheet.write(row, 0, result.get('rank', i + 1), cell_format)
        worksheet.write(row, 1, combo_str, cell_format)
        worksheet.write(row, 2, result['total_score'], number_format)
        _write_score_components(worksheet, row, 3, result, number_format)

    worksheet.set_column(0, 0, 10)
    worksheet.set_column(1, 1, 60)
    worksheet.set_column(2, last_col, 15)


def write_zc_impedance_data(worksheet, results: Dict, header_format, number_format):
    """Z_cインピーダンスデータを書き込む"""

    # ヘッダー
    row = 0
    col = 0
    worksheet.write(row, col, "Frequency [Hz]", header_format)

    # コンデンサ名
    cap_impedances = results.get('capacitor_impedances', {})
    z_cap_host = {name: ensure_numpy(z_c) for name, z_c in cap_impedances.items()}

    for cap_name in z_cap_host:
        col += 1
        worksheet.write(row, col, f"{cap_name} |Z| [Ω]", header_format)

    # データ書き込み
    f_grid = ensure_numpy(results.get('frequency_grid', []))

    # データを間引く（1000点を目安とする）
    indices = create_decimated_indices(len(f_grid), 1000)

    # データ行
    for i in indices:
        row += 1
        col = 0

        # 周波数
        worksheet.write(row, col, float(f_grid[i]), number_format)

        # 各コンデンサのインピーダンス
        for z_c in z_cap_host.values():
            col += 1
            if i < len(z_c):
                z_abs = float(np.abs(z_c[i]))
                worksheet.write(row, col, z_abs, number_format)


def write_impedance_data(worksheet, results: Dict, config: UserConfig, header_format, number_format):
    """PDNインピーダンスデータを書き込む"""

    # ヘッダー
    row = 0
    col = 0
    worksheet.write(row, col, "Frequency [Hz]", header_format)

    # Top-kのラベル
    top_k = results.get('top_k_results', [])[:10]  # 最大10個
    z_pdn_host = [ensure_numpy(result['z_pdn']) for result in top_k]

    for i in range(len(top_k)):
        col += 1
        worksheet.write(row, col, f"Top-{i+1} |Z| [Ω]", header_format)

    # 目標マスク
    col += 1
    worksheet.write(row, col, "Target Mask [Ω]", header_format)

    # デカップリングなし
    z_without = results.get('z_pdn_without_decap')
    has_without = z_without is not None
    if has_without:
        col += 1
        worksheet.write(row, col, "Without Decap |Z| [Ω]", header_format)

    # データ書き込み
    f_grid = ensure_numpy(results.get('frequency_grid', []))
    target_mask = ensure_numpy(results.get('target_mask', []))

    # 評価帯域の計算（Target Mask表示制御用）
    eval_f_L, eval_f_H = config.f_L, config.f_H
    # カスタムマスクモードの時のみカスタムマスクの帯域を評価帯域として使用
    if config.target_impedance_mode == "custom" and config.z_custom_mask:
        custom_f_L, custom_f_H = get_custom_mask_freq_range(config.z_custom_mask)
        if custom_f_L is not None and custom_f_H is not None:
            eval_f_L, eval_f_H = custom_f_L, custom_f_H

    # データを間引く（1000点を目安とする）
    indices = create_decimated_indices(len(f_grid), 1000)

    # データ行
    z_without_np = ensure_numpy(z_without) if has_without else None

    for i in indices:
        row += 1
        col = 0

        # 周波数
        freq = float(f_grid[i])
        worksheet.write(row, col, freq, number_format)

        # Top-kインピーダンス
        for z_pdn in z_pdn_host:
            col += 1
            if i < len(z_pdn):
                z_abs = float(np.abs(z_pdn[i]))
                worksheet.write(row, col, z_abs, number_format)

        # 目標マスク（評価帯域内のみ）
        col += 1
        if eval_f_L <= freq <= eval_f_H and i < len(target_mask):
            worksheet.write(row, col, float(target_mask[i]), number_format)
        # 評価帯域外は空セル

        if has_without and z_without_np is not None and i < len(z_without_np):
            col += 1
            worksheet.write(row, col, float(np.abs(z_without_np[i])), number_format)


def create_zc_chart(workbook, chart_sheet, zc_data_sheet,
                   results: Dict, config: UserConfig):
    """Z_cインピーダンスグラフを作成"""

    # グラフ作成
    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

    # データ範囲
    cap_impedances = results.get('capacitor_impedances', {})
    cap_names_list = list(cap_impedances)
    f_grid = ensure_numpy(results.get('frequency_grid', []))
    if not cap_names_list or not f_grid.size:
        return

    # データ点数（間引き後）
    indices = create_decimated_indices(len(f_grid), 1000)
    n_points = len(indices)

    # 各コンデンサの系列を追加
    for i, cap_name in enumerate(cap_names_list):
        chart.add_series({
            'name': cap_name,
            'categories': ['|Z_c| Data', 1, 0, n_points, 0],  # 周波数
            'values': ['|Z_c| Data', 1, i+1, n_points, i+1],  # |Z_c|
            'line': {'width': 1.5},
            'marker': {'type': 'none'}
        })

    # グラフ設定
    chart.set_title({'name': 'Capacitor Impedance |Z_c| vs Frequency'})
    chart.set_x_axis({
        'name': 'Frequency [Hz]',
        'log_base': 10,
        'min': float(f_grid[0]),
        'max': float(f_grid[-1]),
        'major_gridlines': {'visible': True, 'line': {'color': '#D0D0D0'}},
        'minor_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}
    })
    chart.set_y_axis({
        'name': 'Impedance |Z_c| [Ω]',
        'log_base': 10,
        'major_gridlines': {'visible': True, 'line': {'color': '#D0D0D0'}},
        'minor_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}
    })

    # 凡例
    chart.set_legend({'position': 'right'})

    # グラフサイズと位置
    chart.set_size({'width': 800, 'height': 600})

    # チャートをシートに挿入（左側）
    chart_sheet.insert_chart('B2', chart)


def create_impedance_chart(workbook, chart_sheet, data_sheet,
                          results: Dict, config: UserConfig):
    """PDNインピーダンスグラフを作成"""

    # グラフ作成
    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

    # データ範囲
    top_k = results.get('top_k_results', [])
    f_grid = ensure_numpy(results.get('frequency_grid', []))
    if not f_grid.size:
        return

    # データ点数（間引き後）
    indices = create_decimated_indices(len(f_grid), 1000)
    n_points = len(indices)

    # Top-k系列追加
    colors = ZPDN_PALETTE
    top_count = len(top_k[:10])

    for i, result in enumerate(top_k[:10]):
        chart.add_series({
            'name': f"Top-{i+1}",
            'categories': ['Impedance Data', 1, 0, n_points, 0],  # 周波数
            'values': ['Impedance Data', 1, i+1, n_points, i+1],  # |Z|
            'line': {'color': colors[i % len(colors)], 'width': 1.5},
            'marker': {'type': 'none'}
        })

    # 目標マスク
    chart.add_series({
        'name': 'Target Mask',
        'categories': ['Impedance Data', 1, 0, n_points, 0],
        'values': ['Impedance Data', 1, top_count + 1, n_points, top_count + 1],
        'line': {'color': TARGET_MASK_COLOR, 'dash_type': 'dash', 'width': 2},
        'marker': {'type': 'none'}
    })

    z_without = results.get('z_pdn_without_decap')
    if z_without is not None:
        chart.add_series({
            'name': 'Without Decap',
            'categories': ['Impedance Data', 1, 0, n_points, 0],
            'values': ['Impedance Data', 1, top_count + 2, n_points, top_count + 2],
            'line': {'color': WITHOUT_DECAP_COLOR, 'dash_type': 'dot', 'width': 2},
            'marker': {'type': 'none'}
        })

    # グラフ設定
    chart.set_title({'name': 'PDN Impedance |Z| vs Frequency'})
    chart.set_x_axis({
        'name': 'Frequency [Hz]',
        'log_base': 10,
        'min': float(f_grid[0]),
        'max': float(f_grid[-1]),
        'major_gridlines': {'visible': True, 'line': {'color': '#D0D0D0'}},
        'minor_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}
    })
    chart.set_y_axis({
        'name': 'Impedance |Z| [Ω]',
        'log_base': 10,
        'major_gridlines': {'visible': True, 'line': {'color': '#D0D0D0'}},
        'minor_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}
    })

    # 凡例
    chart.set_legend({'position': 'right'})

    # グラフサイズと位置
    chart.set_size({'width': 800, 'height': 600})

    # チャートをシートに挿入（右側、Z_cグラフの隣）
    chart_sheet.insert_chart('N2', chart)


def write_config_sheet(worksheet, config: UserConfig, header_format, cell_format):
    """設定シートを作成"""
    
    # タイトル
    worksheet.write(0, 0, "Configuration Parameters", header_format)
    worksheet.merge_range(0, 0, 0, 2, "Configuration Parameters", header_format)
    
    # 設定項目を書き込み
    row = 2
    sections = {
        "Frequency Grid": ["f_start", "f_stop", "num_points_per_decade"],
        "Evaluation Band": ["f_L", "f_H"],
        "Target": ["z_target"],
        "Search": ["max_total_parts", "min_total_parts_ratio", "top_k"],
        "Weights": [*WEIGHT_FIELDS, "ignore_safe_anti_resonance"],
        "Monte Carlo": ["mc_enable", "mc_samples", "tol_C", "tol_ESR", "tol_ESL"],
        "System": ["force_numpy", "cuda", "seed"]
    }
    
    for section, params in sections.items():
        # セクションヘッダー
        worksheet.write(row, 0, section, header_format)
        worksheet.merge_range(row, 0, row, 2, section, header_format)
        row += 1
        
        # パラメータ
        for param in params:
            if hasattr(config, param):
                value = getattr(config, param)
                worksheet.write(row, 0, param, cell_format)
                
                # 値のフォーマット
                if isinstance(value, bool):
                    worksheet.write(row, 1, str(value), cell_format)
                elif isinstance(value, (int, float)):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        worksheet.write(row, 1, f"{value:.3E}", cell_format)
                    else:
                        worksheet.write(row, 1, f"{value:.6f}", cell_format)
                else:
                    worksheet.write(row, 1, str(value), cell_format)
                row += 1
        
        row += 1  # セクション間の空行
    
    # 列幅調整
    worksheet.set_column(0, 0, 25)
    worksheet.set_column(1, 1, 20)
