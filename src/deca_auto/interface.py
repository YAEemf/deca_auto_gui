import os
import sys
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from deca_auto.config import (
    UserConfig, load_config, save_config, validate_config,
    get_localized_text
)
from deca_auto.utils import (
    logger, ensure_numpy, parse_scientific_notation,
    get_custom_mask_freq_range, set_log_level
)
from deca_auto.main import run_optimization
from deca_auto.capacitor import calculate_all_capacitor_impedances
from deca_auto.evaluator import format_combination_name

# 環境変数からログレベルを取得して設定
_log_level_str = os.environ.get('DECA_LOG_LEVEL')
if _log_level_str:
    try:
        _log_level = int(_log_level_str)
        set_log_level(_log_level)
    except (ValueError, TypeError):
        pass


MAX_POINTS = 1024

ZPDN_PALETTE = [
    "#de425b",
    "#ef6b55",
    "#fa9257",
    "#ffb762",
    "#ffdc7a",
    "#ffff9d",
    "#cee88f",
    "#9fd184",
    "#72b97c",
    "#45a074",
]
TARGET_MASK_COLOR = "#03DAC6"
WITHOUT_DECAP_COLOR = "#018786"


def format_usage_range(min_val: Optional[int], max_val: Optional[int]) -> str:
    """MIN/MAX値を入力用文字列に整形"""
    if min_val is None and max_val is None:
        return ""
    if min_val is not None and max_val is not None:
        if int(min_val) == int(max_val):
            return str(int(min_val))
        return f"{int(min_val)}-{int(max_val)}"
    if min_val is not None:
        return f"{int(min_val)}-"
    if max_val is not None:
        return f"-{int(max_val)}"
    return ""


def parse_usage_range_input(text: str, max_total: int) -> Tuple[Optional[int], Optional[int]]:
    """使用範囲文字列をMIN/MAXに変換"""
    if text is None:
        return None, None

    raw = str(text).strip()
    if raw == "":
        return None, None

    normalized = raw.replace(" ", "")

    def _to_int(value: str) -> int:
        return int(float(value))

    min_val: Optional[int]
    max_val: Optional[int]

    if '-' not in normalized:
        value = _to_int(normalized)
        min_val = max_val = value
    elif normalized.endswith('-') and normalized.count('-') == 1:
        min_val = _to_int(normalized[:-1])
        max_val = None
    elif normalized.startswith('-') and normalized.count('-') == 1:
        min_val = None
        max_val = _to_int(normalized[1:])
    else:
        parts = normalized.split('-')
        if len(parts) != 2 or parts[0] == '' or parts[1] == '':
            raise ValueError(f"無効な範囲指定: {text}")
        min_val = _to_int(parts[0])
        max_val = _to_int(parts[1])
        if max_val < min_val:
            raise ValueError(f"最小値より小さい最大値が指定されました: {text}")

    if min_val is not None:
        min_val = max(min_val, 0)
    if max_val is not None:
        max_val = max(max_val, 0)

    if max_total is not None and max_total >= 0:
        if min_val is not None and min_val > max_total:
            min_val = max_total
        if max_val is not None and max_val > max_total:
            max_val = max_total
        if min_val is not None and max_val is not None and max_val < min_val:
            max_val = min_val

    return min_val, max_val

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

    if 'z_pdn_without_decap' not in st.session_state:
        st.session_state.z_pdn_without_decap = None

    if 'capacitor_names' not in st.session_state:
        st.session_state.capacitor_names = []

    if 'progress_value' not in st.session_state:
        st.session_state.progress_value = 0.0

    if 'no_search_mode' not in st.session_state:
        st.session_state.no_search_mode = os.environ.get('DECA_NO_SEARCH', '0') == '1'

    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # 0: Settings, 1: Results

    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None  # 前回のアップロードファイル名

    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = None

    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

    if 'top_k_show_flags' not in st.session_state:
        st.session_state.top_k_show_flags = []


@st.fragment
def render_weights_section():
    """
    評価重みセクションをフラグメント化

    このフラグメントにより、評価重みのスライダー操作時に
    メインコンテンツ全体を再実行せず、このセクションのみを再実行することで
    GUIのレスポンスを大幅に向上

    Note:
        各スライダーには一意のkeyを設定してStreamlitの重複キーエラーを回避
        キー命名規則: sidebar_weights_{widget_type}_{parameter_name}
    """
    config = st.session_state.config

    with st.expander(get_localized_text('weights', config) if config.language == 'jp' else 'Evaluation Weights', expanded=True):
        defaults = UserConfig()
        weight_fields = [
            'weight_max', 'weight_area', 'weight_mean', 'weight_anti',
            'weight_flat', 'weight_under', 'weight_parts',
            'weight_num_types', 'weight_resonance', 'weight_mc_worst'
        ]

        col1, col2 = st.columns(2)
        with col1:
            config.weight_max = st.slider("Max", 0.0, 1.0, config.weight_max, 0.1, key="sidebar_weights_slider_max")
            config.weight_area = st.slider("Area", 0.0, 1.0, config.weight_area, 0.1, key="sidebar_weights_slider_area")
            config.weight_anti = st.slider("Anti-resonance", 0.0, 1.0, config.weight_anti, 0.1, key="sidebar_weights_slider_anti")
            config.weight_parts = st.slider("Parts", 0.0, 2.0, config.weight_parts, 0.1, key="sidebar_weights_slider_parts")
            config.weight_flat = st.slider("Flatness", 0.0, 1.0, config.weight_flat, 0.1, key="sidebar_weights_slider_flat")
        with col2:
            config.weight_mean = st.slider("Mean", 0.0, 1.0, config.weight_mean, 0.1, key="sidebar_weights_slider_mean")
            config.weight_under = st.slider("Under", -1.0, 1.0, config.weight_under, 0.1, key="sidebar_weights_slider_under")
            config.weight_resonance = st.slider('Resonance', 0.0, 1.0, config.weight_resonance, 0.1, key="sidebar_weights_slider_resonance")
            config.weight_num_types = st.slider('Types', 0.0, 2.0, config.weight_num_types, 0.1, key="sidebar_weights_slider_num_types")
            config.weight_mc_worst = st.slider("MC worst", 0.0, 1.0, config.weight_mc_worst, 0.1, key="sidebar_weights_slider_mc_worst")

        config.ignore_safe_anti_resonance = st.checkbox(
            get_localized_text('ignore_safe_anti', config),
            value=config.ignore_safe_anti_resonance,
            key="sidebar_weights_checkbox_ignore_safe_anti"
        )

        if st.button(get_localized_text('reset_weights', config), use_container_width=True, key="sidebar_weights_button_reset"):
            for field in weight_fields:
                setattr(config, field, getattr(defaults, field))
            config.ignore_safe_anti_resonance = defaults.ignore_safe_anti_resonance
            st.rerun(scope="fragment")


def create_sidebar():
    """サイドバーの作成"""
    config = st.session_state.config
    
    with st.sidebar:
        st.title(get_localized_text('title', config))
        
        # ファイル操作
        st.header(get_localized_text("load_file", config))
        
        # ファイルアップロード（固定キーを使用）
        uploaded_file = st.file_uploader(
            get_localized_text('load_config', config),
            type=['toml'],
            help=get_localized_text('drop_config', config),
            key="config_file_uploader"  # 固定キーに変更
        )
        
        if uploaded_file is not None:
            # 新しいファイルの場合のみ処理
            current_file_name = uploaded_file.name if uploaded_file else None
            if current_file_name != st.session_state.last_uploaded_file:
                try:
                    # アップロードされたファイルを一時保存
                    temp_path = Path(f"temp_{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.read())

                    # 設定読み込み
                    new_config = load_config(temp_path, verbose=False)
                    st.session_state.config = new_config
                    st.success("設定ファイルを読み込みました")
                    
                    # 一時ファイル削除
                    temp_path.unlink()

                    # アップロードファイル名を記録
                    st.session_state.last_uploaded_file = current_file_name

                    # ★ file_upload_keyのインクリメントを削除
                    # これにより、file_uploaderのウィジェット状態が保持されます
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")
                    logger.error(f"設定ファイル読み込みエラー: {e}")
        
        # 保存ボタン
        if st.button(get_localized_text('save', config), width="stretch"):
            save_current_config()
    
        save_as_name = st.text_input("ファイル名", "config.toml")
        if st.button(get_localized_text('save_as', config), width="stretch"):
            save_current_config(save_as_name)
        
        st.divider()
        
        # 制御ボタン
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.optimization_running:
                if st.button(get_localized_text('start_search', config), type="primary", width="stretch"):
                    start_optimization()
                with col2:
                    if not st.session_state.optimization_running:
                        if st.button(get_localized_text('calculate_zc_only', config), width="stretch"):
                            calculate_zc_only()
            else:
                if st.button(get_localized_text('stop_search', config), type="secondary", width="stretch"):
                    stop_optimization()

        if st.session_state.stop_requested:
            st.info("停止処理中です…")
        
        st.divider()
        
        # パラメーター設定(エクスパンダー)
        
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

        # 評価重み(フラグメント化して軽量化)
        render_weights_section()
        
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

        # 寄生成分
        with st.expander(get_localized_text('stray_parameters', config)):
            col1, col2 = st.columns(2)
            with col1:
                config.R_vrm = st.number_input(
                    get_localized_text('label_R_vrm', config),
                    value=float(config.R_vrm),
                    min_value=0.0,
                    format="%.3e"
                )
                config.L_vrm = st.number_input(
                    get_localized_text('label_L_vrm', config),
                    value=float(config.L_vrm),
                    min_value=0.0,
                    format="%.3e"
                )
                config.R_sN = st.number_input(
                    get_localized_text('label_R_sN', config),
                    value=float(config.R_sN),
                    min_value=0.0,
                    format="%.3e"
                )
                config.L_sN = st.number_input(
                    get_localized_text('label_L_sN', config),
                    value=float(config.L_sN),
                    min_value=0.0,
                    format="%.3e"
                )
                config.L_mntN = st.number_input(
                    get_localized_text('label_L_mntN', config),
                    value=float(config.L_mntN),
                    min_value=0.0,
                    format="%.3e"
                )
                config.C_p = st.number_input(
                    get_localized_text('label_C_p', config),
                    value=float(config.C_p),
                    min_value=0.0,
                    format="%.3e"
                )
            with col2:
                config.R_s = st.number_input(
                    get_localized_text('label_R_s', config),
                    value=float(config.R_s),
                    min_value=0.0,
                    format="%.3e"
                )
                config.L_s = st.number_input(
                    get_localized_text('label_L_s', config),
                    value=float(config.L_s),
                    min_value=0.0,
                    format="%.3e"
                )
                config.R_v = st.number_input(
                    get_localized_text('label_R_v', config),
                    value=float(config.R_v),
                    min_value=0.0,
                    format="%.3e"
                )
                config.L_v = st.number_input(
                    get_localized_text('label_L_v', config),
                    value=float(config.L_v),
                    min_value=0.0,
                    format="%.3e"
                )
                config.R_p = st.number_input(
                    get_localized_text('label_R_p', config),
                    value=float(config.R_p),
                    min_value=0.0,
                    format="%.3e"
                )
                config.tan_delta_p = st.number_input(
                    get_localized_text('label_tan_delta_p', config),
                    value=float(config.tan_delta_p),
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f"
                )
        
        # GPU設定
        with st.expander(get_localized_text('gpu_settings', config)):
            config.force_numpy = st.checkbox(
                "Force NumPy (disable GPU)",
                value=config.force_numpy
            )
            if not config.force_numpy:
                # CUDA Device選択
                from deca_auto.utils import list_cuda_devices
                cuda_devices = list_cuda_devices()

                if cuda_devices:
                    # デバイスリストから選択肢を作成
                    device_options = [f"{dev['id']}:{dev['name']}" for dev in cuda_devices]
                    current_device = config.cuda

                    # 現在のデバイスがリストにあるか確認
                    if current_device < len(device_options):
                        selected_index = current_device
                    else:
                        selected_index = 0
                        config.cuda = 0

                    selected_device = st.selectbox(
                        "CUDA device",
                        options=device_options,
                        index=selected_index,
                        key="cuda_device_selector"
                    )

                    # 選択されたデバイスIDを取得
                    config.cuda = int(selected_device.split(':')[0])
                else:
                    st.warning("利用可能なCUDA Deviceが見つかりません")
                    config.force_numpy = True

                config.max_vram_ratio_limit = st.slider(
                    "Max VRAM ratio",
                    min_value=0.1,
                    max_value=1.0,
                    value=config.max_vram_ratio_limit,
                    step=0.1
                )

        def _on_change_language():
            sel = st.session_state["_lang_display"]
            st.session_state.config.language = "jp" if sel == "日本語" else "en"

        with st.expander(get_localized_text('system', st.session_state.config)):
            current_lang_display = "日本語" if st.session_state.config.language == "jp" else "English"
            lang = st.selectbox(
                get_localized_text('language', st.session_state.config),
                options=["日本語", "English"],
                index=["日本語", "English"].index(current_lang_display),
                key="_lang_display",
                on_change=_on_change_language
            )


def create_main_content():
    """メインコンテンツの作成"""
    config = st.session_state.config
    
    # タブ作成
    tab1, tab2 = st.tabs([
        f"⚙️ {get_localized_text('settings', config)}",
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
    
    # st.header(get_localized_text('settings', config))

    # コンデンサリスト
    st.subheader(get_localized_text('capacitor_list', config))

    # データフレーム作成(コンデンサリストをテーブル表示用に変換)
    cap_data = []
    for cap in config.capacitors:
        # pathの取得(空の場合はRLCモード)
        path = cap.get('path', "") or ""
        has_path = bool(path)

        # RLCモードの場合はデフォルト値、SPICEモードの場合は設定値(なければ0)
        c_val = cap.get('C', 0.0 if not has_path else 0.0)
        esr_val = cap.get('ESR', 15e-3 if not has_path else 0.0)
        esl_val = cap.get('ESL', 2e-10 if not has_path else 0.0)

        cap_data.append({
            'Name': cap.get('name', ''),
            'Path': path,
            'C [F]': format_value(c_val),
            'ESR [Ω]': format_value(esr_val),
            'ESL [H]': format_value(esl_val),
            'L_mnt [H]': format_value(cap.get('L_mnt', config.L_mntN)),
            'usage_range': format_usage_range(cap.get('MIN'), cap.get('MAX'))
        })

    df = pd.DataFrame(cap_data, columns=['Name', 'Path', 'C [F]', 'ESR [Ω]', 'ESL [H]', 'L_mnt [H]', 'usage_range'])

    # 編集可能なデータエディタ
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="capacitor_editor",
        column_config={
            'usage_range': st.column_config.TextColumn(get_localized_text('usage_range', config))
        }
    )

    # 編集内容を反映
    if st.button(get_localized_text("update_caplist", config)):
        new_caps: List[Dict[str, Any]] = []
        existing_names: set[str] = set()

        for _, row in edited_df.iterrows():
            raw_name = row.get('Name')
            if pd.isna(raw_name):
                name_val = ""
            else:
                name_val = str(raw_name).strip()

            raw_path = row.get('Path')
            if pd.isna(raw_path):
                path_val = ""
            else:
                path_val = str(raw_path).strip()

            has_path = bool(path_val)

            if not name_val and has_path:
                base_name = Path(path_val).stem
                candidate = base_name or ""
                suffix = 1
                while candidate in existing_names and candidate:
                    candidate = f"{base_name}_{suffix}"
                    suffix += 1
                name_val = candidate

            if not name_val:
                continue

            base_name = name_val
            suffix = 1
            while name_val in existing_names:
                name_val = f"{base_name}_{suffix}"
                suffix += 1
            existing_names.add(name_val)

            if not has_path:
                c_val = parse_value(row['C [F]'], 0.0)
                esr_val = parse_value(row['ESR [Ω]'], 15e-3)
                esl_val = parse_value(row['ESL [H]'], 2e-10)
            else:
                c_val = parse_value(row['C [F]'], 0.0)
                esr_val = parse_value(row['ESR [Ω]'], 0.0)
                esl_val = parse_value(row['ESL [H]'], 0.0)

            l_mnt_val = parse_value(row['L_mnt [H]'], config.L_mntN)

            try:
                min_count, max_count = parse_usage_range_input(row.get('usage_range', ''), config.max_total_parts)
            except ValueError as exc:
                st.error(str(exc))
                return

            cap: Dict[str, Any] = {'name': name_val}
            if path_val:
                cap['path'] = path_val

            if c_val is not None and c_val != 0.0:
                cap['C'] = c_val
            if esr_val is not None and esr_val != 0.0:
                cap['ESR'] = esr_val
            if esl_val is not None and esl_val != 0.0:
                cap['ESL'] = esl_val
            if l_mnt_val is not None:
                cap['L_mnt'] = l_mnt_val

            if min_count is not None:
                cap['MIN'] = int(min_count)
            if max_count is not None:
                cap['MAX'] = int(max_count)

            new_caps.append(cap)

        config.capacitors = new_caps
        st.success(get_localized_text("update_caplist", config))

    st.divider()

    # 目標マスク設定
    st.subheader(get_localized_text('target_mask', config))

    # モード選択
    mode_options = ["auto", "flat", "custom"]
    mode_labels = {
        "auto": "Auto (自動計算)",
        "flat": "Flat (一定)",
        "custom": "Custom (カスタムマスク)"
    }

    current_mode_index = mode_options.index(config.target_impedance_mode) if config.target_impedance_mode in mode_options else 0
    selected_mode = st.selectbox(
        "Target impedance mode",
        options=mode_options,
        format_func=lambda x: mode_labels[x],
        index=current_mode_index,
        key="target_impedance_mode_selector"
    )
    config.target_impedance_mode = selected_mode

    # モードに応じたパラメーター表示
    if selected_mode == "flat":
        # フラットモード: 一定入力
        new_target = parse_value(
            st.text_input("Target impedance [Ω]", format_value(config.z_target), key="flat_z_target"),
            config.z_target
        )
        if new_target is not None:
            config.z_target = new_target

    elif selected_mode == "auto":
        # 自動計算モード: 電源仕様から計算
        st.markdown("**電源仕様パラメーター**")

        # 電源電圧
        v_supply_val = parse_value(
            st.text_input("Supply voltage [V]", format_value(config.v_supply), key="auto_v_supply"),
            config.v_supply
        )
        if v_supply_val is not None:
            config.v_supply = v_supply_val

        # リップル指定方法の選択
        ripple_mode = st.radio(
            "Ripple specification",
            options=["ratio", "voltage"],
            format_func=lambda x: "Ripple ratio [%]" if x == "ratio" else "Ripple voltage [V]",
            index=0 if config.ripple_ratio is not None else 1,
            key="auto_ripple_mode"
        )

        if ripple_mode == "ratio":
            ripple_ratio_val = parse_value(
                st.text_input("Ripple ratio [%]", format_value(config.ripple_ratio or 5.0), key="auto_ripple_ratio"),
                config.ripple_ratio or 5.0
            )
            if ripple_ratio_val is not None:
                config.ripple_ratio = ripple_ratio_val
                config.ripple_voltage = None
        else:
            ripple_voltage_val = parse_value(
                st.text_input("Ripple voltage [V]", format_value(config.ripple_voltage or 0.05), key="auto_ripple_voltage"),
                config.ripple_voltage or 0.05
            )
            if ripple_voltage_val is not None:
                config.ripple_voltage = ripple_voltage_val
                config.ripple_ratio = None

        # 最大電流
        i_max_val = parse_value(
            st.text_input("Max current [A]", format_value(config.i_max), key="auto_i_max"),
            config.i_max
        )
        if i_max_val is not None:
            config.i_max = i_max_val

        # 過渡電流指定方法の選択
        transient_mode = st.radio(
            "Transient current specification",
            options=["activity", "current"],
            format_func=lambda x: "Switching activity [0-1]" if x == "activity" else "Transient current [A]",
            index=0 if config.switching_activity is not None else 1,
            key="auto_transient_mode"
        )

        if transient_mode == "activity":
            switching_activity_val = parse_value(
                st.text_input("Switching activity [0-1]", format_value(config.switching_activity or 0.5), key="auto_switching_activity"),
                config.switching_activity or 0.5
            )
            if switching_activity_val is not None:
                config.switching_activity = switching_activity_val
                config.i_transient = None
        else:
            i_transient_val = parse_value(
                st.text_input("Transient current [A]", format_value(config.i_transient or 5.0), key="auto_i_transient"),
                config.i_transient or 5.0
            )
            if i_transient_val is not None:
                config.i_transient = i_transient_val
                config.switching_activity = None

        # デザインマージン
        design_margin_val = parse_value(
            st.text_input("Design margin [%]", format_value(config.design_margin), key="auto_design_margin"),
            config.design_margin
        )
        if design_margin_val is not None:
            config.design_margin = design_margin_val

        # 計算結果のプレビュー
        try:
            from deca_auto.utils import calculate_target_impedance_auto
            z_auto = calculate_target_impedance_auto(
                config.v_supply,
                config.ripple_ratio,
                config.ripple_voltage,
                config.i_max,
                config.switching_activity,
                config.i_transient,
                config.design_margin
            )
            st.info(f"計算された目標インピーダンス: {format_value(z_auto)} Ω")
        except Exception as e:
            st.warning(f"計算エラー: {e}")

    elif selected_mode == "custom":
        # カスタムマスクモード
        st.markdown("**カスタムインピーダンスマスク**")

        # カスタムマスク編集
        if config.z_custom_mask:
            # 既存のカスタムマスクを表示
            mask_data = pd.DataFrame(config.z_custom_mask, columns=['Frequency [Hz]', 'Impedance [Ω]'])
            # 値をフォーマット
            mask_data['Frequency [Hz]'] = mask_data['Frequency [Hz]'].apply(format_value)
            mask_data['Impedance [Ω]'] = mask_data['Impedance [Ω]'].apply(format_value)
        else:
            # デフォルトのカスタムマスクを作成
            default_mask = [
                (1e3, 10e-3),
                (5e3, 10e-3),
                (2e4, 8e-3),
                (2e6, 8e-3),
                (1e8, 0.45),
            ]
            mask_data = pd.DataFrame(default_mask, columns=['Frequency [Hz]', 'Impedance [Ω]'])
            mask_data['Frequency [Hz]'] = mask_data['Frequency [Hz]'].apply(format_value)
            mask_data['Impedance [Ω]'] = mask_data['Impedance [Ω]'].apply(format_value)
            config.z_custom_mask = default_mask

        edited_mask = st.data_editor(
            mask_data,
            num_rows="dynamic",
            use_container_width=True,
            key="mask_editor"
        )

        if st.button(get_localized_text("update_mask", config), key="update_custom_mask_button"):
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
                    # 目標マスクを再生成
                    if st.session_state.frequency_grid is not None:
                        from deca_auto.utils import create_target_mask, get_backend
                        xp, _, _ = get_backend(config.force_numpy, config.cuda)
                        st.session_state.target_mask = ensure_numpy(create_target_mask(
                            st.session_state.frequency_grid,
                            config.z_target,
                            config.z_custom_mask,
                            xp,
                            mode=config.target_impedance_mode,
                            config=config
                        ))
                else:
                    st.error("有効なマスクポイントがありません")
            else:
                st.error("マスクポイントを少なくとも1つ設定してください")


@st.fragment
def render_zpdn_results():
    """
    PDN結果表示(グラフとテーブル)をフラグメント化

    このフラグメントは以下を実現:
    1. Top-kグラフの表示
    2. チェックボックスでのグラフ表示制御
    3. チェック変更時の即座な反映(フラグメント内での再実行)

    Note:
        @st.fragment により、このセクションのみを部分的に再実行可能
        チェックボックスの変更検知時にst.rerun(scope="fragment")を使用して
        全体ではなくフラグメント内のみを再実行し、パフォーマンスを向上
    """
    config = st.session_state.config

    # グラフ2: Top-kのZ_pdn特性
    st.subheader("PDNインピーダンス特性 |Z_pdn|")
    if st.session_state.top_k_results and st.session_state.frequency_grid is not None:
        try:
            zpdn_chart = create_zpdn_chart()
            st.altair_chart(zpdn_chart, use_container_width=True)
        except Exception as e:
            st.error(f"グラフ描画エラー: {e}")
    else:
        st.info("探索を実行してください")

    # Top-k結果テーブル
    if st.session_state.top_k_results:
        st.subheader("Top-k 結果")
        try:
            results_df = create_results_dataframe(include_show=True)
            edited_df = st.data_editor(
                results_df,
                use_container_width=True,
                hide_index=True,
                key="topk_selector",
                column_config={
                    'show': st.column_config.CheckboxColumn(
                        get_localized_text('show_column', config),
                        help=get_localized_text('show_column_help', config)
                    )
                },
                disabled=['Rank', 'Combination', 'Total Score', 'Types', 'Parts', 'MC Worst']
            )

            # チェックボックスの状態変更を検知してグラフを即座に更新
            if len(edited_df) == len(st.session_state.top_k_results):
                new_flags = [
                    False if pd.isna(val) else bool(val)
                    for val in edited_df['show'].tolist()
                ]
                # 変更があった場合のみsession_stateを更新してフラグメントを再実行
                if new_flags != st.session_state.top_k_show_flags:
                    st.session_state.top_k_show_flags = new_flags
                    st.rerun(scope="fragment")  # フラグメントのみを再実行(高速)
        except Exception as e:
            st.error(f"テーブル作成エラー: {e}")


def create_results_tab():
    """結果タブの内容"""
    config = st.session_state.config

    # 最適化実行中の場合、自動更新を有効化
    if st.session_state.optimization_running:
        # 定期的な更新のためのプレースホルダー
        progress_placeholder = st.empty()
        graph1_placeholder = st.empty()
        graph2_placeholder = st.empty()
        table_placeholder = st.empty()

        # ポーリングループ(0.5秒ごと)
        import time
        max_iterations = 1000  # 最大500秒(約8分)

        for i in range(max_iterations):
            # キューを処理
            process_result_queue()

            # プログレス表示
            with progress_placeholder.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    prog = st.progress(st.session_state.progress_value)
                    # ★ 下中央に % 表示
                    pct = int(round(st.session_state.progress_value * 100))
                    cL, cC, cR = st.columns([1, 2, 1])
                    with cC:
                        st.markdown(f"**{pct}%**", help="進捗率")
                with col2:
                    st.info("🔄 最適化実行中...")

            # グラフ1: コンデンサのZ_c特性
            with graph1_placeholder.container():
                st.subheader("コンデンサインピーダンス特性 |Z_c|")
                if st.session_state.capacitor_impedances and st.session_state.frequency_grid is not None:
                    try:
                        zc_chart = create_zc_chart()
                        st.altair_chart(zc_chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフ描画エラー: {e}")
                else:
                    st.info("コンデンサのインピーダンス計算中...")

            # グラフ2: Top-kのZ_pdn特性
            with graph2_placeholder.container():
                st.subheader("PDNインピーダンス特性 |Z_pdn|")
                if st.session_state.top_k_results and st.session_state.frequency_grid is not None:
                    try:
                        zpdn_chart = create_zpdn_chart()
                        st.altair_chart(zpdn_chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"グラフ描画エラー: {e}")
                else:
                    st.info("探索実行中...")

            # Top-k結果テーブル
            with table_placeholder.container():
                if st.session_state.top_k_results:
                    st.subheader("Top-k 結果")
                    try:
                        results_df = create_results_dataframe()
                        st.dataframe(results_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"テーブル作成エラー: {e}")

            # 最適化が完了したらループを抜ける
            if not st.session_state.optimization_running:
                with progress_placeholder.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(1.0)
                        cL, cC, cR = st.columns([1, 2, 1])
                        with cC:
                            st.markdown("**100%**")
                    with col2:
                        st.success("✅ 最適化完了")
                st.rerun()
                break

            # 0.5秒待機
            time.sleep(0.5)

    else:
        # 最適化実行中でない場合は通常の表示
        # グラフ1: コンデンサのZ_c特性
        st.subheader("コンデンサインピーダンス特性 |Z_c|")
        if st.session_state.capacitor_impedances and st.session_state.frequency_grid is not None:
            try:
                zc_chart = create_zc_chart()
                st.altair_chart(zc_chart, use_container_width=True)
            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")
        else:
            st.info("コンデンサのインピーダンスを計算してください")

        st.divider()

        # フラグメント化されたPDN結果表示
        render_zpdn_results()


def create_zc_chart() -> alt.Chart:
    """コンデンサZ_c特性のグラフ作成"""
    from deca_auto.utils import create_decimated_indices

    config = st.session_state.config

    # データ準備
    data_list = []
    f_grid = st.session_state.frequency_grid
    f_min = None
    f_max = None

    if f_grid is not None:
        f_grid_np = ensure_numpy(f_grid)
        f_min = float(f_grid_np[0])
        f_max = float(f_grid_np[-1])

        # 間引きインデックスを作成
        indices = create_decimated_indices(len(f_grid_np), MAX_POINTS)

        for name, z_c in st.session_state.capacitor_impedances.items():
            z_c_np = ensure_numpy(z_c)

            # 間引いたデータを追加
            for i in indices:
                data_list.append({
                    'Frequency': float(f_grid_np[i]),
                    'Impedance': float(np.abs(z_c_np[i])),
                    'Capacitor': str(name)
                })

    # データが空の場合の処理
    if len(data_list) == 0:
        return alt.Chart(pd.DataFrame()).mark_line()

    df = pd.DataFrame(data_list)

    # X軸のスケール設定(domainで範囲を固定)
    x_scale = alt.Scale(type='log', base=10)
    if f_min is not None and f_max is not None:
        x_scale = alt.Scale(type='log', base=10, domain=[f_min, f_max])

    # Altairチャート作成
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Frequency:Q',
                scale=x_scale,
                axis=alt.Axis(title='Frequency [Hz]', grid=True, format='.1e')),
        y=alt.Y('Impedance:Q',
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(title='|Z_c| [Ω]', grid=True)),
        color=alt.Color('Capacitor:N', legend=alt.Legend(title='Capacitor')),
        tooltip=['Capacitor:N',
                alt.Tooltip('Frequency:Q', format='.3e'),
                alt.Tooltip('Impedance:Q', format='.3e')]
    ).properties(
        width=750,
        height=400,
        title='Capacitor Impedance Characteristics'
    ).configure_axis(
        gridOpacity=0.5
    )

    return chart


def create_zpdn_chart() -> alt.Chart:
    """PDN Z_pdn特性のグラフ作成"""
    from deca_auto.utils import create_decimated_indices

    config = st.session_state.config

    f_grid = st.session_state.frequency_grid
    target_mask = st.session_state.target_mask
    z_without = st.session_state.z_pdn_without_decap
    top_k_results = st.session_state.top_k_results

    if f_grid is None or (not top_k_results and target_mask is None and z_without is None):
        return alt.Chart(pd.DataFrame()).mark_line()

    f_grid_np = ensure_numpy(f_grid)
    f_min = float(f_grid_np[0])
    f_max = float(f_grid_np[-1])

    # 間引きインデックスを作成
    indices = create_decimated_indices(len(f_grid_np), MAX_POINTS)
    indices_target = create_decimated_indices(len(f_grid_np), 100)

    data_list = []
    running = st.session_state.optimization_running
    show_flags = st.session_state.get('top_k_show_flags', [])

    for i, result in enumerate(top_k_results[:10]):
        if not running:
            if i >= len(show_flags) or not show_flags[i]:
                continue
        z_pdn = result.get('z_pdn')
        if z_pdn is None:
            continue
        z_pdn_np = ensure_numpy(z_pdn)
        if len(z_pdn_np) == 0:
            continue
        for j in indices:
            if j < len(z_pdn_np):
                data_list.append({
                    'Frequency': float(f_grid_np[j]),
                    'Impedance': float(np.abs(z_pdn_np[j])),
                    'Type': f"Top-{i+1}",
                    'Order': i,
                    'StrokeDash': 'Solid'
                })

    if target_mask is not None:
        target_np = ensure_numpy(target_mask)
        f_lo, f_hi = None, None
        if config.z_custom_mask:
            f_lo, f_hi = get_custom_mask_freq_range(config.z_custom_mask)

        for j in indices_target:
            if j < len(target_np):
                fval = float(f_grid_np[j])
                if (f_lo is not None and fval < f_lo) or (f_hi is not None and fval > f_hi):
                    continue
                data_list.append({
                    'Frequency': fval,
                    'Impedance': float(target_np[j]),
                    'Type': 'Target Mask',
                    'Order': 1000,
                    'StrokeDash': 'Target Mask'
                })

    if z_without is not None:
        z_base_np = ensure_numpy(z_without)
        if len(z_base_np) == len(f_grid_np):
            for j in indices:
                data_list.append({
                    'Frequency': float(f_grid_np[j]),
                    'Impedance': float(np.abs(z_base_np[j])),
                    'Type': 'Without Decap',
                    'Order': 1001,
                    'StrokeDash': 'Without Decap'
                })

    if not data_list:
        return alt.Chart(pd.DataFrame()).mark_line()

    df = pd.DataFrame(data_list)

    # 実際にデータに含まれるTypeを抽出
    actual_types = df['Type'].unique().tolist()

    # Top-kの候補とその他の固定項目を分離
    top_k_types = [t for t in actual_types if t.startswith('Top-')]
    other_types = [t for t in actual_types if not t.startswith('Top-')]

    # Top-kをソート(Top-1, Top-2, ...の順)
    top_k_types.sort(key=lambda x: int(x.split('-')[1]))

    # color_domainとcolor_rangeを構築
    color_domain = []
    color_range = []

    # Top-k候補を追加
    for i, t in enumerate(top_k_types):
        color_domain.append(t)
        # ZPDN_PALETTEから対応する色を取得(Top-1はindex 0、Top-2はindex 1...)
        top_index = int(t.split('-')[1]) - 1
        color_range.append(ZPDN_PALETTE[top_index % len(ZPDN_PALETTE)])

    # その他の項目を追加
    if 'Without Decap' in other_types:
        color_domain.append('Without Decap')
        color_range.append(WITHOUT_DECAP_COLOR)
    if 'Target Mask' in other_types:
        color_domain.append('Target Mask')
        color_range.append(TARGET_MASK_COLOR)

    # X軸のスケール設定(domainで範囲を固定)
    x_scale = alt.Scale(type='log', base=10, domain=[f_min, f_max])

    chart = alt.Chart(df).mark_line(clip=True).encode(
        x=alt.X(
            'Frequency:Q',
            scale=x_scale,
            axis=alt.Axis(title='Frequency [Hz]', grid=True, format='.1e')
        ),
        y=alt.Y(
            'Impedance:Q',
            scale=alt.Scale(type='log', base=10),
            axis=alt.Axis(title='|Z_pdn| [Ω]', grid=True)
        ),
        color=alt.Color(
            'Type:N',
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title='Configuration')
        ),
        strokeDash=alt.StrokeDash(
            'StrokeDash:N',
            scale=alt.Scale(
                domain=['Target Mask', 'Without Decap', 'Solid'],
                range=[[4, 4], [4, 4], [0]]
            ),
            legend=None
        ),
        order='Order:O',
        tooltip=[
            'Type:N',
            alt.Tooltip('Frequency:Q', format='.3e'),
            alt.Tooltip('Impedance:Q', format='.3e')
        ]
    ).properties(
        width=800,
        height=450,
        title='PDN Impedance Characteristics'
    ).configure_axis(
        gridOpacity=0.5
    ).interactive()

    return chart


def create_results_dataframe(include_show: bool = False) -> pd.DataFrame:
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
    
    show_flags = st.session_state.get('top_k_show_flags', [])

    for i, result in enumerate(st.session_state.top_k_results):
        try:
            count_vec = result.get('count_vector', [])
            if count_vec is not None:
                count_vec = ensure_numpy(count_vec)
                combo_str = format_combination_name(count_vec, cap_names)

                num_types = int(np.count_nonzero(count_vec)) if len(count_vec) > 0 else 0
                show_flag = bool(show_flags[i]) if include_show and i < len(show_flags) else True

                data.append({
                    'show': show_flag,
                    'Rank': result.get('rank', i+1),
                    'Combination': combo_str,
                    'Total Score': f"{result.get('total_score', 0):.6f}",
                    'Types': num_types,
                    'Parts': int(np.sum(count_vec)) if len(count_vec) > 0 else 0,
                    'MC Worst': f"{result.get('mc_worst_score', 0):.6f}" if 'mc_worst_score' in result else 'N/A'
                })
        except Exception as e:
            logger.error(f"結果行作成エラー: {e}")
            continue

    if not data:
        # データが空の場合、空のDataFrameを返す
        columns = ['Rank', 'Combination', 'Total Score', 'Types', 'Parts', 'MC Worst']
        if include_show:
            columns = ['show'] + columns
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(data)
    df = df.sort_values('Rank', kind='stable').reset_index(drop=True)

    if not include_show:
        df = df.drop(columns=['show'])

    return df


def format_value(value: Optional[float]) -> str:
    """数値を表示用にフォーマット"""
    if value is None:
        return ""
    if value == 0:
        return "0"
    elif abs(value) < 1e-3 or abs(value) >= 1e3:
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
        filepath = Path("config.toml")
    
    try:
        if save_config(config, filepath):
            st.success(f"設定を保存しました: {filepath}")
        else:
            st.error("設定の保存に失敗しました")
    except Exception as e:
        st.error(f"保存エラー: {e}")
        logger.error(f"設定保存エラー: {e}")


def optimization_worker(config: UserConfig, result_queue: queue.Queue, stop_event: Optional[threading.Event] = None):
    """最適化処理のワーカースレッド"""
    try:
        def gui_callback(data: Dict):
            """GUI更新用コールバック"""
            result_queue.put(data)
        
        # 最適化実行
        results = run_optimization(config, gui_callback, stop_event)
        
        # 完了通知(結果を含める)
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
    st.session_state.top_k_show_flags = []
    st.session_state.z_pdn_without_decap = None

    # キューをクリア
    while not st.session_state.result_queue.empty():
        try:
            st.session_state.result_queue.get_nowait()
        except:
            break

    st.session_state.stop_event = threading.Event()
    st.session_state.stop_requested = False

    # ワーカースレッド開始
    thread = threading.Thread(
        target=optimization_worker,
        args=(st.session_state.config, st.session_state.result_queue, st.session_state.stop_event),
        daemon=True
    )
    thread.start()
    st.session_state.optimization_thread = thread
    
    st.success("最適化を開始しました")
    
    # タブが切り替わるようにrerunを呼ぶ
    st.rerun()


def stop_optimization():
    """最適化を停止"""
    event = st.session_state.get('stop_event')
    if event is not None and not event.is_set():
        event.set()
        st.session_state.stop_requested = True
        st.warning("停止リクエストを送信しました")
    else:
        st.info("停止処理中です…")


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
            xp,
            mode=config.target_impedance_mode,
            config=config
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
        
        # タブを切り替えてrerun
        st.session_state.active_tab = 1
        st.rerun()
        
    except Exception as e:
        st.error(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()


def process_result_queue():
    """結果キューを処理"""
    try:
        processed = False
        
        while not st.session_state.result_queue.empty():
            data = st.session_state.result_queue.get_nowait()
            processed = True
            
            if data['type'] == 'capacitor_update':
                # コンデンサインピーダンス更新
                st.session_state.capacitor_impedances[data['name']] = data['z_c']
                
                # 周波数グリッドも更新(初回のみ)
                if 'frequency' in data and st.session_state.frequency_grid is None:
                    st.session_state.frequency_grid = data['frequency']
            
            elif data['type'] == 'grid_update':
                # 周波数グリッドと目標マスク更新
                if 'frequency_grid' in data:
                    st.session_state.frequency_grid = data['frequency_grid']
                if 'target_mask' in data:
                    st.session_state.target_mask = data['target_mask']
                if 'z_without_decap' in data:
                    st.session_state.z_pdn_without_decap = data['z_without_decap']

            elif data['type'] == 'top_k_update':
                # Top-k更新
                st.session_state.top_k_results = data['top_k']
                st.session_state.capacitor_names = data.get('capacitor_names', [])
                st.session_state.top_k_show_flags = [True] * len(st.session_state.top_k_results)

                # 周波数グリッドと目標マスクの更新
                if 'frequency_grid' in data:
                    st.session_state.frequency_grid = data['frequency_grid']
                if 'target_mask' in data:
                    st.session_state.target_mask = data['target_mask']
                
                # 進捗値の更新
                if 'progress' in data:
                    st.session_state.progress_value = data['progress']
                
            elif data['type'] == 'complete':
                # 完了
                st.session_state.optimization_running = False
                stopped = False
                if 'results' in data and data['results'] is not None:
                    stopped = data['results'].get('stopped', False)
                st.session_state.progress_value = 0.0 if stopped else 1.0
                st.session_state.stop_event = None
                st.session_state.stop_requested = False

                # 結果データの更新
                if 'results' in data:
                    results = data['results']
                    st.session_state.top_k_results = results.get('top_k_results', [])
                    st.session_state.capacitor_names = results.get('capacitor_names', [])
                    st.session_state.frequency_grid = results.get('frequency_grid')
                    st.session_state.target_mask = results.get('target_mask')
                    st.session_state.z_pdn_without_decap = results.get('z_pdn_without_decap')

                    total_results = len(st.session_state.top_k_results)
                    default_selected = min(10, total_results)
                    st.session_state.top_k_show_flags = [idx < default_selected for idx in range(total_results)]

                if stopped:
                    logger.info("探索停止処理が完了しました")
                    st.info("探索を停止しました")
                else:
                    logger.info("最適化完了")
                    st.success("最適化が完了しました")

            elif data['type'] == 'error':
                # エラー
                st.session_state.optimization_running = False
                st.session_state.progress_value = 0.0
                st.session_state.stop_event = None
                st.session_state.stop_requested = False
                st.error(f"エラー: {data['message']}")
        
        return processed
    
    except queue.Empty:
        return False
    except Exception as e:
        logger.error(f"キュー処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# メイン実行
def main():
    """メインエントリポイント"""
    initialize_session_state()
    create_sidebar()
    create_main_content()


if __name__ == "__main__":
    main()
