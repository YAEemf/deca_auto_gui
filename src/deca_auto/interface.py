"""
Streamlit GUIモジュール
GUIとAltairグラフの処理/更新、ユーザー操作、パラメーターの編集と保存
"""

import os
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# 絶対パスでインポート
from deca_auto.config import (
    UserConfig, load_config, save_config, validate_config,
    get_localized_text, parse_scientific_notation
)
from deca_auto.utils import logger, ensure_numpy
from deca_auto.main import run_optimization
from deca_auto.capacitor import calculate_all_capacitor_impedances
from deca_auto.evaluator import format_combination_name


# Streamlit設定
st.set_page_config(
    page_title="PDN Impedance Optimization Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """セッション状態の初期化"""
    if 'config' not in st.session_state:
        # 環境変数から設定ファイルを取得
        config_files = os.environ.get('DECA_CONFIG_FILES', '').split(',')
        if config_files and config_files[0]:
            st.session_state.config = load_config(config_files[0])
        else:
            st.session_state.config = UserConfig()
    
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    
    if 'optimization_thread' not in st.session_state:
        st.session_state.optimization_thread = None
    
    if 'result_queue' not in st.session_state:
        st.session_state.result_queue = queue.Queue()
    
    if 'capacitor_impedances' not in st.session_state:
        st.session_state.capacitor_impedances = {}
    
    if 'top_k_results' not in st.session_state:
        st.session_state.top_k_results = []
    
    if 'frequency_grid' not in st.session_state:
        st.session_state.frequency_grid = None
    
    if 'target_mask' not in st.session_state:
        st.session_state.target_mask = None
    
    if 'capacitor_names' not in st.session_state:
        st.session_state.capacitor_names = []
    
    if 'progress_value' not in st.session_state:
        st.session_state.progress_value = 0.0
    
    if 'no_search_mode' not in st.session_state:
        st.session_state.no_search_mode = os.environ.get('DECA_NO_SEARCH', '0') == '1'


def create_sidebar():
    """サイドバーの作成"""
    config = st.session_state.config
    
    with st.sidebar:
        st.title(get_localized_text('title', config))
        
        # ファイル操作
        st.header("📁 ファイル操作")
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            get_localized_text('load_config', config),
            type=['toml'],
            help=get_localized_text('drop_config', config)
        )
        
        if uploaded_file is not None:
            try:
                # アップロードされたファイルを一時保存
                temp_path = Path(f"temp_{uploaded_file.name}")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                # 設定読み込み
                new_config = load_config(temp_path)
                st.session_state.config = new_config
                st.success("設定ファイルを読み込みました")
                
                # 一時ファイル削除
                temp_path.unlink()
                st.rerun()  # 画面更新
            except Exception as e:
                st.error(f"読み込みエラー: {e}")
                logger.error(f"設定ファイル読み込みエラー: {e}")
        
        # 保存ボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button(get_localized_text('save', config)):
                save_current_config()
        
        with col2:
            save_as_name = st.text_input("ファイル名", "config.toml")
            if st.button(get_localized_text('save_as', config)):
                save_current_config(save_as_name)
        
        st.divider()
        
        # 制御ボタン
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.optimization_running:
                if st.button(get_localized_text('start_search', config), type="primary"):
                    start_optimization()
            else:
                if st.button(get_localized_text('stop_search', config), type="secondary"):
                    stop_optimization()
        
        with col2:
            if st.button(get_localized_text('calculate_zc_only', config)):
                calculate_zc_only()
        
        st.divider()
        
        # パラメーター設定（エクスパンダー）
        
        # 周波数グリッド
        with st.expander(get_localized_text('frequency_grid', config)):
            f_start = parse_value(
                st.text_input("f_start [Hz]", format_value(config.f_start)),
                config.f_start
            )
            if f_start is not None:
                config.f_start = f_start
                
            f_stop = parse_value(
                st.text_input("f_stop [Hz]", format_value(config.f_stop)),
                config.f_stop
            )
            if f_stop is not None:
                config.f_stop = f_stop
                
            config.num_points_per_decade = st.number_input(
                "Points per decade", 
                value=config.num_points_per_decade,
                min_value=10,
                max_value=10000
            )
        
        # 評価帯域
        with st.expander(get_localized_text('evaluation_band', config)):
            f_L = parse_value(
                st.text_input("f_L [Hz]", format_value(config.f_L)),
                config.f_L
            )
            if f_L is not None:
                config.f_L = f_L
                
            f_H = parse_value(
                st.text_input("f_H [Hz]", format_value(config.f_H)),
                config.f_H
            )
            if f_H is not None:
                config.f_H = f_H
        
        # 探索設定
        with st.expander(get_localized_text('search_settings', config)):
            config.max_total_parts = st.number_input(
                "Max total parts",
                value=config.max_total_parts,
                min_value=1,
                max_value=100
            )
            config.min_total_parts_ratio = st.slider(
                "Min total parts ratio",
                min_value=0.0,
                max_value=1.0,
                value=config.min_total_parts_ratio,
                step=0.1
            )
            config.top_k = st.number_input(
                "Top-k",
                value=config.top_k,
                min_value=1,
                max_value=100
            )
            config.shuffle_evaluation = st.checkbox(
                "Shuffle evaluation",
                value=config.shuffle_evaluation
            )
        
        # Monte Carlo設定
        with st.expander(get_localized_text('monte_carlo', config)):
            config.mc_enable = st.checkbox(
                "Enable Monte Carlo",
                value=config.mc_enable
            )
            if config.mc_enable:
                config.mc_samples = st.number_input(
                    "MC samples",
                    value=config.mc_samples,
                    min_value=1,
                    max_value=1000
                )
                config.tol_C = st.slider(
                    "Capacitance tolerance",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.tol_C,
                    step=0.01
                )
                config.tol_ESR = st.slider(
                    "ESR tolerance",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.tol_ESR,
                    step=0.01
                )
                config.tol_ESL = st.slider(
                    "ESL tolerance",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.tol_ESL,
                    step=0.01
                )
        
        # GPU設定
        with st.expander(get_localized_text('gpu_settings', config)):
            config.force_numpy = st.checkbox(
                "Force NumPy (disable GPU)",
                value=config.force_numpy
            )
            if not config.force_numpy:
                config.cuda = st.number_input(
                    "CUDA device",
                    value=config.cuda,
                    min_value=0,
                    max_value=7
                )
                config.max_vram_ratio_limit = st.slider(
                    "Max VRAM ratio",
                    min_value=0.1,
                    max_value=1.0,
                    value=config.max_vram_ratio_limit,
                    step=0.1
                )


def create_main_content():
    """メインコンテンツの作成"""
    config = st.session_state.config
    
    # タブ作成
    tab1, tab2 = st.tabs([
        f"📝 {get_localized_text('settings', config)}",
        f"📊 {get_localized_text('results', config)}"
    ])
    
    # 設定タブ
    with tab1:
        create_settings_tab()
    
    # 結果タブ
    with tab2:
        create_results_tab()


def create_settings_tab():
    """設定タブの内容"""
    config = st.session_state.config
    
    st.header(get_localized_text('settings', config))
    
    # コンデンサリスト
    st.subheader(get_localized_text('capacitor_list', config))
    
    # データフレーム作成
    cap_data = []
    for cap in config.capacitors:
        cap_data.append({
            'Name': cap.get('name', ''),
            'Path': cap.get('path', ''),
            'C [F]': format_value(cap.get('C', 0)),
            'ESR [Ω]': format_value(cap.get('ESR', 15e-3)),
            'ESL [H]': format_value(cap.get('ESL', 0.5e-9)),
            'L_mnt [H]': format_value(cap.get('L_mnt', config.L_mntN))
        })
    
    df = pd.DataFrame(cap_data)
    
    # 編集可能なデータエディタ
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="capacitor_editor"
    )
    
    # 編集内容を反映
    if st.button("コンデンサリストを更新"):
        new_caps = []
        for _, row in edited_df.iterrows():
            cap = {
                'name': row['Name'],
                'path': row['Path'] if row['Path'] else None,
                'C': parse_value(row['C [F]'], None),
                'ESR': parse_value(row['ESR [Ω]'], 15e-3),
                'ESL': parse_value(row['ESL [H]'], 0.5e-9),
                'L_mnt': parse_value(row['L_mnt [H]'], None)
            }
            # 空のL_mntはNoneのままにする（デフォルト値はcapacitor.pyで処理）
            new_caps.append(cap)
        config.capacitors = new_caps
        st.success("コンデンサリストを更新しました")
    
    st.divider()
    
    # 目標マスク設定
    st.subheader(get_localized_text('target_mask', config))
    
    use_custom = st.checkbox(
        get_localized_text('use_custom_mask', config),
        value=config.z_custom_mask is not None
    )
    
    if not use_custom:
        new_target = parse_value(
            st.text_input("Target impedance [Ω]", format_value(config.z_target)),
            config.z_target
        )
        if new_target is not None:
            config.z_target = new_target
        config.z_custom_mask = None
    else:
        # カスタムマスク編集
        if config.z_custom_mask:
            mask_data = pd.DataFrame(config.z_custom_mask, columns=['Frequency [Hz]', 'Impedance [Ω]'])
        else:
            mask_data = pd.DataFrame(columns=['Frequency [Hz]', 'Impedance [Ω]'])
        
        edited_mask = st.data_editor(
            mask_data,
            num_rows="dynamic",
            use_container_width=True,
            key="mask_editor"
        )
        
        if st.button("マスクを更新"):
            if len(edited_mask) > 0:
                mask_points = []
                for _, row in edited_mask.iterrows():
                    f_val = parse_value(str(row['Frequency [Hz]']), None)
                    z_val = parse_value(str(row['Impedance [Ω]']), None)
                    if f_val is not None and z_val is not None and f_val > 0 and z_val > 0:
                        mask_points.append((f_val, z_val))
                if mask_points:
                    config.z_custom_mask = sorted(mask_points, key=lambda x: x[0])
                    st.success("カスタムマスクを更新しました")
                else:
                    st.error("有効なマスクポイントがありません")


def create_results_tab():
    """結果タブの内容"""
    config = st.session_state.config
    
    st.header(get_localized_text('results', config))
    
    # 進捗表示エリア
    progress_container = st.container()
    
    # 進捗表示
    if st.session_state.optimization_running:
        with progress_container:
            col1, col2 = st.columns([3, 1])
            with col1:
                # 進捗バー（session_stateに保存）
                if 'progress_value' not in st.session_state:
                    st.session_state.progress_value = 0.0
                progress_bar = st.progress(st.session_state.progress_value)
            with col2:
                st.info("🔄 最適化実行中...")
    
    # グラフ1: コンデンサのZ_c特性
    st.subheader("📈 コンデンサインピーダンス特性 |Z_c|")
    zc_chart_container = st.container()
    
    with zc_chart_container:
        if st.session_state.capacitor_impedances and st.session_state.frequency_grid is not None:
            try:
                zc_chart = create_zc_chart()
                st.altair_chart(zc_chart, use_container_width=True)
            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")
                logger.error(f"Z_cグラフエラー: {e}")
        else:
            st.info("コンデンサのインピーダンスを計算してください")
    
    st.divider()
    
    # グラフ2: Top-kのZ_pdn特性
    st.subheader("📊 PDNインピーダンス特性 |Z_pdn| (Top-k)")
    zpdn_chart_container = st.container()
    
    with zpdn_chart_container:
        if st.session_state.top_k_results and st.session_state.frequency_grid is not None:
            try:
                zpdn_chart = create_zpdn_chart()
                st.altair_chart(zpdn_chart, use_container_width=True)
            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")
                logger.error(f"Z_pdnグラフエラー: {e}")
        else:
            st.info("探索を実行してください")
    
    # Top-k結果テーブル
    if st.session_state.top_k_results:
        st.subheader("🏆 Top-k 結果")
        try:
            results_df = create_results_dataframe()
            st.dataframe(results_df, use_container_width=True)
        except Exception as e:
            st.error(f"テーブル作成エラー: {e}")
            logger.error(f"結果テーブルエラー: {e}")
    
    # 結果キューの処理
    process_result_queue()


def create_zc_chart() -> alt.Chart:
    """コンデンサZ_c特性のグラフ作成"""
    config = st.session_state.config
    
    # データ準備
    data_list = []
    for name, z_c in st.session_state.capacitor_impedances.items():
        z_c_np = ensure_numpy(z_c)
        f_grid = st.session_state.frequency_grid
        if f_grid is not None:
            f_grid_np = ensure_numpy(f_grid)
            
            # データフレーム用にデータを整形（間引き）
            step = max(1, len(f_grid_np) // 100)
            for i in range(0, len(f_grid_np), step):
                if i < len(f_grid_np):
                    data_list.append({
                        'Frequency': float(f_grid_np[i]),
                        'Impedance': float(np.abs(z_c_np[i])),
                        'Capacitor': str(name)
                    })
    
    # データが空の場合の処理
    if len(data_list) == 0:
        # 空のチャートを返す
        return alt.Chart(pd.DataFrame()).mark_line()
    
    df = pd.DataFrame(data_list)
    
    # Altairチャート作成
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Frequency:Q', 
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(title='Frequency [Hz]', grid=True)),
        y=alt.Y('Impedance:Q',
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(title='|Z| [Ω]', grid=True)),
        color=alt.Color('Capacitor:N', legend=alt.Legend(title='Capacitor')),
        tooltip=['Capacitor:N', 
                alt.Tooltip('Frequency:Q', format='.3e'),
                alt.Tooltip('Impedance:Q', format='.3e')]
    ).properties(
        width=800,
        height=400,
        title='Capacitor Impedance Characteristics'
    ).configure_axis(
        gridOpacity=0.3
    )
    
    return chart


def create_zpdn_chart() -> alt.Chart:
    """PDN Z_pdn特性のグラフ作成"""
    config = st.session_state.config
    
    # データ準備
    data_list = []
    f_grid = st.session_state.frequency_grid
    target_mask = st.session_state.target_mask
    
    # データが不足している場合
    if f_grid is None or len(st.session_state.top_k_results) == 0:
        # 空のチャートを返す
        return alt.Chart(pd.DataFrame()).mark_line()
    
    f_grid_np = ensure_numpy(f_grid)
    
    # Top-k結果
    for i, result in enumerate(st.session_state.top_k_results[:10]):
        z_pdn = result.get('z_pdn')
        if z_pdn is not None:
            z_pdn_np = ensure_numpy(z_pdn)
            if len(z_pdn_np) > 0:
                # データフレーム用にデータを整形（間引き）
                step = max(1, len(f_grid_np) // 100)
                for j in range(0, len(f_grid_np), step):
                    if j < len(f_grid_np) and j < len(z_pdn_np):
                        data_list.append({
                            'Frequency': float(f_grid_np[j]),
                            'Impedance': float(np.abs(z_pdn_np[j])),
                            'Type': f"Top-{i+1}",
                            'Order': i
                        })
    
    # 目標マスク
    if target_mask is not None:
        target_np = ensure_numpy(target_mask)
        step = max(1, len(f_grid_np) // 100)
        for j in range(0, len(f_grid_np), step):
            if j < len(f_grid_np) and j < len(target_np):
                data_list.append({
                    'Frequency': float(f_grid_np[j]),
                    'Impedance': float(target_np[j]),
                    'Type': 'Target Mask',
                    'Order': 999  # 最後に表示
                })
    
    # データが空の場合の処理
    if len(data_list) == 0:
        return alt.Chart(pd.DataFrame()).mark_line()
    
    df = pd.DataFrame(data_list)
    
    # Altairチャート作成
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Frequency:Q',
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(title='Frequency [Hz]', grid=True)),
        y=alt.Y('Impedance:Q',
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(title='|Z| [Ω]', grid=True)),
        color=alt.Color('Type:N',
                       scale=alt.Scale(scheme='category10'),
                       legend=alt.Legend(title='Configuration')),
        strokeDash=alt.condition(
            alt.datum.Type == 'Target Mask',
            alt.value([5, 5]),  # 破線
            alt.value([0])  # 実線
        ),
        order='Order:O',
        tooltip=['Type:N',
                alt.Tooltip('Frequency:Q', format='.3e'),
                alt.Tooltip('Impedance:Q', format='.3e')]
    ).properties(
        width=800,
        height=400,
        title='PDN Impedance Characteristics (Top-k)'
    ).configure_axis(
        gridOpacity=0.3
    )
    
    return chart


def create_results_dataframe() -> pd.DataFrame:
    """結果テーブルのDataFrame作成"""
    data = []
    cap_names = st.session_state.get('capacitor_names', [])
    
    if not cap_names:
        # capacitor_namesが空の場合、デフォルト名を使用
        num_caps = 0
        for result in st.session_state.top_k_results:
            count_vec = result.get('count_vector', [])
            if hasattr(count_vec, '__len__'):
                num_caps = max(num_caps, len(count_vec))
        cap_names = [f"Cap_{i+1}" for i in range(num_caps)]
    
    for i, result in enumerate(st.session_state.top_k_results):
        try:
            count_vec = result.get('count_vector', [])
            if count_vec is not None:
                count_vec = ensure_numpy(count_vec)
                combo_str = format_combination_name(count_vec, cap_names)
                
                data.append({
                    'Rank': result.get('rank', i+1),
                    'Combination': combo_str,
                    'Total Score': f"{result.get('total_score', 0):.6f}",
                    'Parts Count': int(np.sum(count_vec)) if len(count_vec) > 0 else 0,
                    'MC Worst': f"{result.get('mc_worst_score', 0):.6f}" if 'mc_worst_score' in result else 'N/A'
                })
        except Exception as e:
            logger.error(f"結果行作成エラー: {e}")
            continue
    
    if not data:
        # データが空の場合、空のDataFrameを返す
        return pd.DataFrame(columns=['Rank', 'Combination', 'Total Score', 'Parts Count', 'MC Worst'])
    
    return pd.DataFrame(data)


def format_value(value: Optional[float]) -> str:
    """数値を表示用にフォーマット"""
    if value is None:
        return ""
    if value == 0:
        return "0"
    elif abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.3e}"
    else:
        return f"{value:.6f}"


def parse_value(text: str, default: Optional[float] = None) -> Optional[float]:
    """テキストから数値を解析"""
    if not text or text.strip() == "":
        return default
    try:
        return parse_scientific_notation(text)
    except:
        return default


def save_current_config(filename: Optional[str] = None):
    """現在の設定を保存"""
    config = st.session_state.config
    
    if filename:
        filepath = Path(filename)
    else:
        filepath = Path("user_config.toml")
    
    try:
        if save_config(config, filepath):
            st.success(f"設定を保存しました: {filepath}")
        else:
            st.error("設定の保存に失敗しました")
    except Exception as e:
        st.error(f"保存エラー: {e}")
        logger.error(f"設定保存エラー: {e}")


def optimization_worker(config: UserConfig, result_queue: queue.Queue):
    """最適化処理のワーカースレッド"""
    try:
        def gui_callback(data: Dict):
            """GUI更新用コールバック"""
            result_queue.put(data)
        
        # 最適化実行
        results = run_optimization(config, gui_callback)
        
        # 完了通知（結果を含める）
        result_queue.put({
            'type': 'complete',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"最適化エラー: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'type': 'error',
            'message': str(e)
        })


def start_optimization():
    """最適化を開始"""
    if st.session_state.optimization_running:
        st.warning("すでに実行中です")
        return
    
    # 設定検証
    if not validate_config(st.session_state.config):
        st.error("設定が無効です")
        return
    
    # 実行状態を設定
    st.session_state.optimization_running = True
    st.session_state.progress_value = 0.0
    
    # ワーカースレッド開始
    thread = threading.Thread(
        target=optimization_worker,
        args=(st.session_state.config, st.session_state.result_queue),
        daemon=True
    )
    thread.start()
    st.session_state.optimization_thread = thread
    
    st.success("最適化を開始しました")
    st.rerun()  # すぐに画面更新


def stop_optimization():
    """最適化を停止"""
    st.session_state.optimization_running = False
    st.warning("停止リクエストを送信しました")


def calculate_zc_only():
    """Z_cのみ計算"""
    config = st.session_state.config
    
    try:
        # 周波数グリッド生成
        from deca_auto.utils import generate_frequency_grid, create_target_mask, get_backend
        
        xp, _, _ = get_backend(config.force_numpy, config.cuda)
        f_grid = generate_frequency_grid(
            config.f_start,
            config.f_stop,
            config.num_points_per_decade,
            xp
        )
        
        # NumPyに変換して保存
        st.session_state.frequency_grid = ensure_numpy(f_grid)
        
        # 目標マスクも生成
        target_mask = create_target_mask(
            f_grid,
            config.z_target,
            config.z_custom_mask,
            xp
        )
        st.session_state.target_mask = ensure_numpy(target_mask)
        
        # Z_c計算
        capacitor_impedances = {}
        
        def zc_callback(data):
            if data['type'] == 'capacitor_update':
                capacitor_impedances[data['name']] = data['z_c']
                st.session_state.capacitor_impedances[data['name']] = data['z_c']
        
        cap_impedances = calculate_all_capacitor_impedances(
            config, f_grid, xp, zc_callback
        )
        
        # 全て更新
        st.session_state.capacitor_impedances = cap_impedances
        
        st.success("Z_c計算が完了しました")
        st.rerun()
        
    except Exception as e:
        st.error(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()


def process_result_queue():
    """結果キューを処理"""
    try:
        needs_rerun = False
        
        while not st.session_state.result_queue.empty():
            data = st.session_state.result_queue.get_nowait()
            
            if data['type'] == 'capacitor_update':
                # コンデンサインピーダンス更新
                st.session_state.capacitor_impedances[data['name']] = data['z_c']
                
                # 周波数グリッドも更新（初回のみ）
                if 'frequency' in data and st.session_state.frequency_grid is None:
                    st.session_state.frequency_grid = data['frequency']
                
                needs_rerun = True
                
            elif data['type'] == 'top_k_update':
                # Top-k更新
                st.session_state.top_k_results = data['top_k']
                st.session_state.capacitor_names = data.get('capacitor_names', [])
                
                # 周波数グリッドと目標マスクの更新
                if 'frequency_grid' in data:
                    st.session_state.frequency_grid = data['frequency_grid']
                if 'target_mask' in data:
                    st.session_state.target_mask = data['target_mask']
                
                # 進捗値の更新
                if 'progress' in data:
                    st.session_state.progress_value = data['progress']
                
                needs_rerun = True
                
            elif data['type'] == 'complete':
                # 完了
                st.session_state.optimization_running = False
                st.session_state.progress_value = 1.0
                
                # 結果データの更新
                if 'results' in data:
                    results = data['results']
                    st.session_state.top_k_results = results.get('top_k_results', [])
                    st.session_state.capacitor_names = results.get('capacitor_names', [])
                    st.session_state.frequency_grid = results.get('frequency_grid')
                    st.session_state.target_mask = results.get('target_mask')
                
                st.success("最適化が完了しました")
                needs_rerun = True
                
            elif data['type'] == 'error':
                # エラー
                st.session_state.optimization_running = False
                st.session_state.progress_value = 0.0
                st.error(f"エラー: {data['message']}")
        
        # 画面更新が必要な場合
        if needs_rerun:
            st.rerun()
    
    except queue.Empty:
        pass
    except Exception as e:
        logger.error(f"キュー処理エラー: {e}")
        import traceback
        traceback.print_exc()


# メイン実行
def main():
    """メインエントリポイント"""
    initialize_session_state()
    create_sidebar()
    create_main_content()


if __name__ == "__main__":
    main()