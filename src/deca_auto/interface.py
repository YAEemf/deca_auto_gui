import os
import queue
import threading
from copy import deepcopy
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from deca_auto.__main__ import calculate_zc
from deca_auto.config import (
    WEIGHT_FIELDS, WEIGHT_RANGE, UserConfig, default_weights, get_localized_text,
    load_config, save_config, validate_config,
)
from deca_auto.evaluator import format_combination_name
from deca_auto.main import run_optimization
from deca_auto.utils import (
    calculate_target_impedance_auto, create_target_mask, ensure_numpy,
    get_custom_mask_freq_range, list_cuda_devices, logger, parse_scientific_notation,
    set_log_level,
)


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



CAP_COLUMNS = {
    "name": "Name", "path": "Path", "C": "C [F]", "ESR": "ESR [Ω]",
    "ESL": "ESL [H]", "L_mnt": "L_mnt [H]",
}
CAP_EDITOR_KEY = "settings_config_capacitors_data_editor"
MASK_EDITOR_KEY = "settings_config_target_data_editor_custom"
MASK_COLUMNS = ["Frequency [Hz]", "Impedance [Ω]"]
UPLOAD_KEY = "sidebar_file_load_file_uploader"
SAVE_NAME_KEY = "sidebar_file_save_text_input_filename"
WEIGHT_TOGGLE = "ignore_safe_anti_resonance"


def initialize_session_state():
    state = st.session_state
    if "config" not in state:
        config_files = os.environ.get("DECA_CONFIG_FILES", "").split(",")
        config = load_config(config_files[0]) if config_files[0] else UserConfig()
        for field, env, convert in (
            ("force_numpy", "DECA_FORCE_NUMPY", lambda value: value.strip().lower() in ("1", "true", "yes", "on")),
            ("cuda", "DECA_CUDA_DEVICE", int),
            ("excel_path", "DECA_EXCEL_PATH", str),
        ):
            if env in os.environ:
                try:
                    setattr(config, field, convert(os.environ[env]))
                except ValueError:
                    logger.warning("無効な環境変数を無視します: %s", env)
        state.config = config
        state.input_errors = {}
        state.editor_sources = {}
        state.result_queue = queue.Queue()
        state.optimization_running = False
        state.stop_event = None
        state.stop_requested = False
        clear_results(deepcopy(config))


def clear_results(config):
    st.session_state.update(
        result_config=config,
        frequency_grid=None,
        target_mask=None,
        z_pdn_without_decap=None,
        capacitor_impedances={},
        capacitor_names=[],
        top_k_results=[],
        top_k_show_flags={},
        progress_value=0.0,
        result_notice=None,
    )


def localized(key):
    return get_localized_text(key, st.session_state.config)


def format_value(value):
    """再入力時に精度が変わらない表示を使う。"""
    return "" if value is None else str(value)


def parse_number(text):
    value = parse_scientific_notation(text)
    if not isfinite(value):
        raise ValueError(f"有限の数値を入力してください: {text}")
    return value


def update_config_field(field, key, parse):
    state = st.session_state
    try:
        value = parse(state[key]) if parse else state[key]
    except (ValueError, OverflowError) as exc:
        state.input_errors[key] = str(exc)
        return
    state.input_errors.pop(key, None)
    setattr(state.config, field, value)


def config_widget(section, field, widget, label, *, page="sidebar", mode=None, **kwargs):
    """Configから初期化し、編集時はコールバックでConfigへ反映する。"""
    key = f"{page}_config_{section}_{widget}_{field.lower()}"
    if mode:
        key += f"_{mode}"
    state = st.session_state
    if key not in state:
        value = getattr(state.config, field)
        if widget == "text_input":
            value = format_value(value)
        elif widget == "slider":
            value = float(value)
            low, high = kwargs["min_value"], kwargs["max_value"]
            if not low <= value <= high:
                state.input_errors[key] = f"{label}: {low}～{high}の範囲で入力してください。"
                value = min(high, max(low, value))
        elif widget == "selectbox" and value not in kwargs["options"]:
            state.input_errors[key] = f"{label}: {value!r} は選択できません。"
            value = kwargs["options"][0]
        state[key] = value
    optional = kwargs.pop("optional", False)
    parse = None
    if widget == "text_input" and (optional or not isinstance(getattr(state.config, field), str)):
        parse = lambda text: None if optional and not text.strip() else parse_number(text)
    value = getattr(st, widget)(
        label, key=key, on_change=update_config_field, args=(field, key, parse), **kwargs
    )
    if key in state.input_errors:
        st.error(state.input_errors[key])
    return value


def reset_weights():
    state = st.session_state
    for field, value in default_weights().items():
        widget = "checkbox" if field == WEIGHT_TOGGLE else "slider"
        key = f"sidebar_config_weights_{widget}_{field}"
        state[key] = value
        state.input_errors.pop(key, None)
        setattr(state.config, field, value)


@st.fragment
def render_weights_section():
    with st.expander(localized("weights"), expanded=True):
        minimum, maximum, step = WEIGHT_RANGE
        for field in WEIGHT_FIELDS:
            config_widget(
                "weights", field, "slider", localized(f"label_{field}"),
                min_value=minimum, max_value=maximum, step=step, help=localized(f"help_{field}"),
            )
        config_widget(
            "weights", WEIGHT_TOGGLE, "checkbox", localized(f"label_{WEIGHT_TOGGLE}"),
            help=localized(f"help_{WEIGHT_TOGGLE}"),
        )
        st.button(localized("reset_weights"), key="sidebar_config_weights_button_reset",
                  on_click=reset_weights, width="stretch")


@st.fragment
def render_frequency_section():
    with st.expander(localized("frequency_grid")):
        config_widget("frequency", "f_start", "text_input", "f_start [Hz]")
        config_widget("frequency", "f_stop", "text_input", "f_stop [Hz]")
        config_widget("frequency", "num_points_per_decade", "number_input",
                      "Points per decade", min_value=1, step=1)
    with st.expander(localized("evaluation_band")):
        config_widget("evaluation", "f_L", "text_input", "f_L [Hz]")
        config_widget("evaluation", "f_H", "text_input", "f_H [Hz]")


@st.fragment
def render_search_section():
    with st.expander(localized("search_settings")):
        config_widget("search", "max_total_parts", "number_input", "Max total parts",
                      min_value=1, step=1)
        config_widget("search", "min_total_parts_ratio", "slider", "Min total parts ratio",
                      min_value=0.0, max_value=1.0, step=0.1)
        config_widget("search", "top_k", "number_input", "Top-k", min_value=1, step=1)
        config_widget("search", "shuffle_evaluation", "checkbox", "Shuffle evaluation")


@st.fragment
def render_mc_section():
    with st.expander(localized("monte_carlo")):
        config_widget("monte_carlo", "mc_enable", "checkbox", "Enable Monte Carlo")
        config_widget("monte_carlo", "mc_samples", "number_input", "MC samples",
                      min_value=1, step=1)
        for field, label in (
            ("tol_C", "Capacitance tolerance"), ("tol_ESR", "ESR tolerance"),
            ("tol_ESL", "ESL tolerance"), ("mlcc_derating", "MLCC derating"),
        ):
            config_widget("monte_carlo", field, "slider", label,
                          min_value=0.0, max_value=1.0, step=0.01)


@st.fragment
def render_parasitics_section():
    with st.expander(localized("stray_parameters")):
        for column, fields in zip(st.columns(2), (
            ("R_vrm", "L_vrm", "R_sN", "L_sN", "L_mntN", "C_p"),
            ("R_s", "L_s", "R_v", "L_v", "R_p", "tan_delta_p"),
        )):
            with column:
                for field in fields:
                    config_widget("parasitics", field, "text_input", localized(f"label_{field}"))
    with st.expander("SPICE"):
        config_widget("spice", "dc_bias", "text_input", "DC bias [V]")
        config_widget("spice", "model_path", "text_input", "Model directory")


@st.fragment
def render_gpu_section():
    with st.expander(localized("gpu_settings")):
        force_numpy = config_widget("gpu", "force_numpy", "checkbox", "Force NumPy (disable GPU)")
        if not force_numpy:
            devices = list_cuda_devices()
            if devices:
                st.caption(", ".join(f"{device['id']}: {device['name']}" for device in devices))
            else:
                st.warning("利用可能なCUDA Deviceが見つかりません。")
        config_widget("gpu", "cuda", "number_input", "CUDA device", min_value=0, step=1)
        config_widget("gpu", "max_vram_ratio_limit", "slider", "Max VRAM ratio",
                      min_value=0.0, max_value=1.0, step=0.01)


@st.fragment
def render_system_fields():
    config_widget("system", "seed", "number_input", "Random seed", step=1)
    config_widget("system", "dtype_r", "selectbox", "Real precision",
                  options=["float16", "float32", "float64", "float128"])
    config_widget("system", "dtype_c", "selectbox", "Complex precision",
                  options=["complex64", "complex128", "complex256"])
    config_widget("system", "use_gui", "checkbox", "Launch GUI by default")
    config_widget("system", "server_port", "number_input", "Server port", min_value=1, step=1)
    config_widget("system", "dark_theme", "checkbox", "Dark theme at launch")
    config_widget("output", "excel_path", "text_input", "Excel output directory")
    config_widget("output", "excel_name", "text_input", "Excel output name")


def load_uploaded_config():
    state = st.session_state
    uploaded = state[UPLOAD_KEY]
    state.file_notice = None
    if uploaded is None:
        return
    try:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_bytes(uploaded.getvalue())
            config = load_config(path, verbose=False, strict=True)
    except Exception as exc:
        state.file_notice = ("error", f"設定ファイル読み込みエラー: {exc}")
        return

    # ウィジェットの再生成前に、以前の入力値とデータエディタの差分を破棄する。
    for key in list(state):
        if key.startswith(("sidebar_config_", "settings_config_")):
            del state[key]
    state.config = config
    state.input_errors.clear()
    state.editor_sources.clear()
    state.file_notice = ("success", "設定ファイルを読み込みました")


def save_current_config(named=False):
    state = st.session_state
    if state.input_errors:
        state.file_notice = ("error", "入力エラーを修正してから保存してください。")
        return
    filename = state[SAVE_NAME_KEY] if named else "config.toml"
    filepath = Path(filename)
    if save_config(state.config, filepath):
        state.file_notice = ("success", f"設定を保存しました: {filepath}")
    else:
        state.file_notice = ("error", "設定の保存に失敗しました")


def create_sidebar():
    state = st.session_state
    with st.sidebar:
        st.title(localized("title"))
        st.header(localized("load_file"))
        st.file_uploader(localized("load_config"), type=["toml"], key=UPLOAD_KEY,
                         help=localized("drop_config"), on_change=load_uploaded_config)
        st.button(localized("save"), key="sidebar_file_save_button_default",
                  on_click=save_current_config, width="stretch")
        st.text_input("ファイル名", "config.toml", key=SAVE_NAME_KEY)
        st.button(localized("save_as"), key="sidebar_file_save_button_named",
                  on_click=save_current_config, args=(True,), width="stretch")
        if state.get("file_notice"):
            kind, text = state.file_notice
            getattr(st, kind)(text)

        st.divider()
        if state.optimization_running:
            st.button(localized("stop_search"), key="sidebar_run_search_button_stop",
                      on_click=stop_optimization, disabled=state.stop_requested, width="stretch")
            if state.stop_requested:
                st.info("停止処理中です…")
        else:
            search, preview = st.columns(2)
            if os.environ.get("DECA_NO_SEARCH") != "1":
                search.button(localized("start_search"), key="sidebar_run_search_button_start",
                              on_click=start_optimization, type="primary", width="stretch")
            preview.button(localized("calculate_zc_only"), key="sidebar_run_preview_button_calculate",
                           on_click=calculate_zc_only, width="stretch")

        st.divider()
        render_frequency_section()
        render_search_section()
        render_weights_section()
        render_mc_section()
        render_parasitics_section()
        render_gpu_section()
        with st.expander(localized("system")):
            # 言語は設定画面全体に反映する。
            config_widget("system", "language", "selectbox", localized("language"),
                          options=["jp", "en"], format_func={"jp": "日本語", "en": "English"}.get)
            render_system_fields()


def format_usage_range(minimum, maximum):
    if minimum == maximum:
        return format_value(minimum)
    return f"{format_value(minimum)}-{format_value(maximum)}"


def parse_usage_range(text):
    if not text:
        return None, None
    low, separator, high = text.partition("-")
    minimum = int(low) if low else None
    maximum = (int(high) if high else None) if separator else minimum
    if maximum is not None and (maximum < 0 or (minimum is not None and maximum < minimum)):
        raise ValueError(f"無効な使用数範囲: {text}")
    if text == "-":
        raise ValueError(f"無効な使用数範囲: {text}")
    return minimum, maximum


def capacitor_row(cap):
    return {
        **{column: format_value(cap.get(field)) for field, column in CAP_COLUMNS.items()},
        "usage_range": format_usage_range(cap.get("MIN"), cap.get("MAX")),
    }


def cell_text(value):
    return "" if pd.isna(value) else str(value).strip()


def parse_capacitors(edited, source):
    caps = []
    names = set()
    for _, row in edited.iterrows():
        original = source[int(row["_source"])] if pd.notna(row["_source"]) else {}
        cap = original.copy()
        original_row = capacitor_row(original)
        for field, column in CAP_COLUMNS.items():
            text = cell_text(row[column])
            if text == original_row[column]:
                continue
            if not text:
                cap.pop(field, None)
            else:
                cap[field] = text if field in ("name", "path") else parse_number(text)

        usage = cell_text(row["usage_range"])
        if usage != original_row["usage_range"]:
            for field, value in zip(("MIN", "MAX"), parse_usage_range(usage)):
                if value is None:
                    cap.pop(field, None)
                else:
                    cap[field] = value

        name = cap.get("name") or Path(cap.get("path") or "").stem
        if not name:
            raise ValueError("コンデンサの名前またはモデルパスを入力してください。")
        base = name
        suffix = 1
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        cap["name"] = name
        names.add(name)
        caps.append(cap)
    return caps


@st.fragment
def render_capacitors():
    state = st.session_state
    st.subheader(localized("capacitor_list"))
    if CAP_EDITOR_KEY not in state.editor_sources:
        state.editor_sources[CAP_EDITOR_KEY] = deepcopy(state.config.capacitors)
    source = state.editor_sources[CAP_EDITOR_KEY]
    # 元の行を識別し、省略値とGUIに列のない設定を編集後も保持する。
    rows = [dict(capacitor_row(cap), _source=str(i)) for i, cap in enumerate(source)]
    columns = [*CAP_COLUMNS.values(), "usage_range", "_source"]
    original = pd.DataFrame(rows, columns=columns, dtype="string")
    edited = st.data_editor(
        original, num_rows="dynamic", width="stretch", hide_index=True, key=CAP_EDITOR_KEY,
        column_config={
            "_source": None,
            "usage_range": st.column_config.TextColumn(localized("usage_range")),
        },
    )
    try:
        state.config.capacitors = parse_capacitors(edited, source)
    except ValueError as exc:
        state.input_errors[CAP_EDITOR_KEY] = str(exc)
        st.error(str(exc))
    else:
        state.input_errors.pop(CAP_EDITOR_KEY, None)


def render_custom_mask():
    state = st.session_state
    if MASK_EDITOR_KEY not in state.editor_sources:
        state.editor_sources[MASK_EDITOR_KEY] = deepcopy(state.config.z_custom_mask)
    source = state.editor_sources[MASK_EDITOR_KEY]
    rows = [tuple(map(format_value, point)) for point in (source or [])]
    original = pd.DataFrame(rows, columns=MASK_COLUMNS, dtype="string")
    edited = st.data_editor(
        original, num_rows="dynamic", width="stretch", hide_index=True, key=MASK_EDITOR_KEY,
    )
    try:
        if edited.equals(original):
            state.config.z_custom_mask = source
        else:
            points = [tuple(map(parse_number, row)) for row in edited.itertuples(index=False, name=None)]
            if any(f <= 0 or z <= 0 for f, z in points):
                raise ValueError("周波数とインピーダンスには正の値を入力してください。")
            state.config.z_custom_mask = sorted(points)
    except ValueError as exc:
        state.input_errors[MASK_EDITOR_KEY] = str(exc)
        st.error(str(exc))
    else:
        state.input_errors.pop(MASK_EDITOR_KEY, None)


@st.fragment
def render_target_section():
    state = st.session_state
    config = state.config
    st.subheader(localized("target_mask"))
    mode_labels = {"auto": "Auto (自動計算)", "flat": "Flat (一定)", "custom": "Custom (カスタムマスク)"}
    mode = config_widget("target", "target_impedance_mode", "selectbox", "Target impedance mode",
                         page="settings", options=list(mode_labels), format_func=mode_labels.get)
    if mode != "custom":
        state.editor_sources.pop(MASK_EDITOR_KEY, None)
        state.input_errors.pop(MASK_EDITOR_KEY, None)
        if MASK_EDITOR_KEY in state:
            del state[MASK_EDITOR_KEY]
    for key in list(state.input_errors):
        if key.startswith("settings_config_target_text_input_") and not key.endswith(f"_{mode}"):
            del state.input_errors[key]

    if mode == "custom":
        render_custom_mask()
    elif mode == "flat":
        config_widget("target", "z_target", "text_input", "Target impedance [Ω]",
                      page="settings", mode=mode)
    else:
        st.caption(localized("auto_target_help"))
        for field, label in (
            ("v_supply", "Supply voltage [V]"), ("ripple_ratio", "Ripple ratio [%]"),
            ("ripple_voltage", "Ripple voltage [V]"), ("i_max", "Max current [A]"),
            ("switching_activity", "Switching activity [0–1]"),
            ("i_transient", "Transient current [A]"), ("design_margin", "Design margin [%]"),
        ):
            config_widget(
                "target", field, "text_input", label, page="settings", mode=mode,
                optional=field in ("ripple_ratio", "ripple_voltage", "switching_activity", "i_transient"),
            )
        try:
            target = calculate_target_impedance_auto(
                config.v_supply, config.ripple_ratio, config.ripple_voltage, config.i_max,
                config.switching_activity, config.i_transient, config.design_margin,
            )
            st.info(f"計算された目標インピーダンス: {format_value(target)} Ω")
        except (ValueError, ZeroDivisionError) as exc:
            st.warning(f"計算エラー: {exc}")


def render_chart(title, key, ready, chart, empty_text):
    st.subheader(title)
    if ready:
        try:
            st.altair_chart(chart(), key=key, width="stretch")
        except Exception as exc:
            logger.exception("グラフ描画エラー")
            st.error(f"グラフ描画エラー: {exc}")
    else:
        st.info(empty_text)


def render_pdn_results():
    state = st.session_state
    results = state.top_k_results
    chart_container = st.container()
    if results:
        st.subheader("Top-k 結果")
        table = create_results_dataframe(include_show=True)
        if state.optimization_running:
            st.dataframe(table.drop(columns="show"), width="stretch")
        else:
            edited = st.data_editor(
                table, key="results_search_topk_data_editor", width="stretch", hide_index=True,
                disabled=[column for column in table if column != "show"],
                column_config={"show": st.column_config.CheckboxColumn(
                    localized("show_column"), help=localized("show_column_help")
                )},
            )
            state.top_k_show_flags = dict(zip(edited["Rank"], edited["show"]))
    with chart_container:
        render_chart(
            "PDNインピーダンス特性 |Z_pdn|", "results_search_pdn_altair_chart",
            state.frequency_grid is not None and (
                bool(results) or state.z_pdn_without_decap is not None or state.target_mask is not None
            ),
            create_zpdn_chart, "探索を実行してください",
        )


render_finished_pdn_results = st.fragment(render_pdn_results)


def render_result_overview():
    state = st.session_state
    if state.optimization_running:
        progress = state.progress_value
        st.progress(progress, text=f"最適化実行中… {progress:.0%}")
    if state.result_notice:
        kind, text = state.result_notice
        getattr(st, kind)(text)
    render_chart(
        "コンデンサインピーダンス特性 |Z_c|", "results_preview_capacitors_altair_chart",
        bool(state.capacitor_impedances) and state.frequency_grid is not None,
        create_zc_chart, "コンデンサのインピーダンスを計算してください",
    )
    st.divider()


@st.fragment(run_every=0.5)
def poll_results():
    process_result_queue()
    if not st.session_state.optimization_running:
        # ポーリングを解除し、サイドバーを探索開始ボタンに戻す。
        st.rerun()
    render_result_overview()
    render_pdn_results()


def create_results_tab():
    if st.session_state.optimization_running:
        poll_results()
    else:
        render_result_overview()
        render_finished_pdn_results()


def create_zc_chart() -> alt.Chart:
    """コンデンサZ_c特性のグラフ作成"""
    from deca_auto.utils import create_decimated_indices

    # データ準備
    data_list = []
    f_grid_np = st.session_state.frequency_grid
    f_min = None
    f_max = None

    if f_grid_np is not None:
        f_min = float(f_grid_np[0])
        f_max = float(f_grid_np[-1])

        # 間引きインデックスを作成
        indices = create_decimated_indices(len(f_grid_np), MAX_POINTS)

        for name, z_c_np in st.session_state.capacitor_impedances.items():

            # 間引いたデータを追加
            for i in indices:
                data_list.append({
                    'Frequency': float(f_grid_np[i]),
                    'Impedance': float(np.abs(z_c_np[i])),
                    'Capacitor': str(name)
                })

    # データが空の場合の処理
    if len(data_list) == 0:
        return alt.Chart(pd.DataFrame({'Frequency': [], 'Impedance': []})).mark_line().encode(x='Frequency:Q', y='Impedance:Q')

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

    config = st.session_state.result_config

    f_grid_np = st.session_state.frequency_grid
    target_mask = st.session_state.target_mask
    z_without = st.session_state.z_pdn_without_decap
    top_k_results = st.session_state.top_k_results

    if f_grid_np is None or (not top_k_results and target_mask is None and z_without is None):
        return alt.Chart(pd.DataFrame({'Frequency': [], 'Impedance': []})).mark_line().encode(x='Frequency:Q', y='Impedance:Q')

    f_min = float(f_grid_np[0])
    f_max = float(f_grid_np[-1])

    # 間引きインデックスを作成
    indices = create_decimated_indices(len(f_grid_np), MAX_POINTS)
    indices_target = create_decimated_indices(len(f_grid_np), 100)

    data_list = []
    running = st.session_state.optimization_running
    show_flags = st.session_state.get('top_k_show_flags', {})

    for i, result in enumerate(top_k_results[:10] if running else top_k_results):
        rank = result.get('rank', i + 1)
        if not running and not show_flags.get(rank, rank <= 10):
            continue
        z_pdn_np = result.get('z_pdn')
        if z_pdn_np is None:
            continue
        for j in indices:
            if j < len(z_pdn_np):
                data_list.append({
                    'Frequency': float(f_grid_np[j]),
                    'Impedance': float(np.abs(z_pdn_np[j])),
                    'Type': f"Top-{rank}",
                    'Order': rank,
                    'StrokeDash': 'Solid'
                })

    if target_mask is not None:
        # 評価帯域と同じロジックでTarget Maskの描画範囲を決定
        f_lo, f_hi = config.f_L, config.f_H
        # カスタムマスクモードの時のみカスタムマスクの帯域を使用
        if config.target_impedance_mode == "custom" and config.z_custom_mask:
            custom_f_L, custom_f_H = get_custom_mask_freq_range(config.z_custom_mask)
            if custom_f_L is not None and custom_f_H is not None:
                f_lo, f_hi = custom_f_L, custom_f_H

        for j in indices_target:
            if j < len(target_mask):
                fval = float(f_grid_np[j])
                if fval < f_lo or fval > f_hi:
                    continue
                data_list.append({
                    'Frequency': fval,
                    'Impedance': float(target_mask[j]),
                    'Type': 'Target Mask',
                    'Order': 1000,
                    'StrokeDash': 'Target Mask'
                })

    if z_without is not None:
        if len(z_without) == len(f_grid_np):
            for j in indices:
                data_list.append({
                    'Frequency': float(f_grid_np[j]),
                    'Impedance': float(np.abs(z_without[j])),
                    'Type': 'Without Decap',
                    'Order': 1001,
                    'StrokeDash': 'Without Decap'
                })

    if not data_list:
        return alt.Chart(pd.DataFrame({'Frequency': [], 'Impedance': []})).mark_line().encode(x='Frequency:Q', y='Impedance:Q')

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
    for t in top_k_types:
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



def create_results_dataframe(include_show=False):
    state = st.session_state
    results = state.top_k_results
    names = state.capacitor_names
    if not names:
        count = max((len(result["count_vector"]) for result in results), default=0)
        names = [f"Cap_{i + 1}" for i in range(count)]

    rows = []
    for i, result in enumerate(results):
        counts = result["count_vector"]
        rank = result.get("rank", i + 1)
        mc_worst = (result.get("components") or {}).get("mc_worst")
        rows.append({
            "show": rank <= 10,
            "Rank": rank,
            "Combination": format_combination_name(counts, names),
            "Total Score": f"{result['total_score']:.6f}",
            "Types": np.count_nonzero(counts),
            "Parts": int(np.sum(counts)),
            "MC Worst": f"{mc_worst:.6f}" if mc_worst is not None else "N/A",
        })
    columns = ["show", "Rank", "Combination", "Total Score", "Types", "Parts", "MC Worst"]
    table = pd.DataFrame(rows, columns=columns).sort_values("Rank", kind="stable").reset_index(drop=True)
    return table if include_show else table.drop(columns="show")


def optimization_worker(config, result_queue, stop_event):
    try:
        results = run_optimization(config, result_queue.put, stop_event)
        result_queue.put({"type": "complete", "results": results})
    except Exception as exc:
        logger.exception("最適化エラー")
        result_queue.put({"type": "error", "message": str(exc)})


def start_optimization():
    state = st.session_state
    if state.optimization_running:
        return
    config = deepcopy(state.config)
    if state.input_errors or not validate_config(config):
        state.result_notice = ("error", "設定が無効です")
        return

    clear_results(deepcopy(config))
    state.result_queue = queue.Queue()
    state.stop_event = threading.Event()
    state.stop_requested = False
    state.optimization_running = True
    thread = threading.Thread(
        name="deca-auto-optimization", target=optimization_worker,
        args=(config, state.result_queue, state.stop_event), daemon=True,
    )
    thread.start()


def stop_optimization():
    state = st.session_state
    if state.stop_event is not None:
        state.stop_event.set()
        state.stop_requested = True


def calculate_zc_only():
    state = st.session_state
    if state.input_errors:
        state.result_notice = ("error", "入力エラーを修正してください。")
        return
    config = deepcopy(state.config)
    try:
        grid, impedances = calculate_zc(config, allow_gpu=True)
        target = create_target_mask(
            grid, config.z_target, config.z_custom_mask, np,
            mode=config.target_impedance_mode, config=config,
        )
    except Exception as exc:
        logger.exception("コンデンサインピーダンス計算エラー")
        state.result_notice = ("error", f"計算エラー: {exc}")
        return

    clear_results(config)
    state.frequency_grid = grid
    state.target_mask = target
    state.capacitor_impedances = impedances
    state.result_notice = ("success", "Z_c計算が完了しました")


def update_result_arrays(data, *fields):
    for field in fields:
        if field in data:
            value = data[field]
            st.session_state[field] = None if value is None else ensure_numpy(value)


def update_top_k(results, names):
    state = st.session_state
    state.top_k_results = [
        {**result, "count_vector": ensure_numpy(result["count_vector"]),
         "z_pdn": ensure_numpy(result["z_pdn"]) if result.get("z_pdn") is not None else None}
        for result in results
    ]
    state.capacitor_names = names
    state.top_k_show_flags = {}


def process_result_queue():
    state = st.session_state
    while True:
        try:
            data = state.result_queue.get_nowait()
        except queue.Empty:
            return

        kind = data["type"]
        if kind == "capacitor_update":
            state.capacitor_impedances[data["name"]] = ensure_numpy(data["z_c"])
            if state.frequency_grid is None and "frequency" in data:
                update_result_arrays({"frequency_grid": data["frequency"]}, "frequency_grid")
        elif kind == "grid_update":
            update_result_arrays(data, "frequency_grid", "target_mask")
            if "z_without_decap" in data:
                update_result_arrays({"z_pdn_without_decap": data["z_without_decap"]}, "z_pdn_without_decap")
        elif kind == "top_k_update":
            update_top_k(data["top_k"], data.get("capacitor_names", []))
            update_result_arrays(data, "frequency_grid", "target_mask")
            state.progress_value = data.get("progress", state.progress_value)
        elif kind in ("complete", "error"):
            state.optimization_running = False
            state.stop_event = None
            state.stop_requested = False
            if kind == "error":
                state.result_notice = ("error", f"エラー: {data['message']}")
                continue

            results = data.get("results") or {}
            if "top_k_results" in results:
                update_top_k(results["top_k_results"], results.get("capacitor_names", []))
            update_result_arrays(results, "frequency_grid", "target_mask", "z_pdn_without_decap")
            if "capacitor_impedances" in results:
                state.capacitor_impedances = {
                    name: ensure_numpy(values) for name, values in results["capacitor_impedances"].items()
                }
            if results.get("stopped"):
                state.result_notice = ("info", "探索を停止しました")
            else:
                state.progress_value = 1.0
                state.result_notice = ("success", "最適化が完了しました")


def main():
    st.set_page_config(
        page_title="PDN Impedance Optimization Tool", page_icon="⚡",
        layout="wide", initial_sidebar_state="expanded",
    )
    if level := os.environ.get("DECA_LOG_LEVEL"):
        try:
            set_log_level(int(level))
        except ValueError:
            logger.warning("無効なログレベルを無視します: %s", level)
    initialize_session_state()
    create_sidebar()
    settings, results = st.tabs([f"⚙️ {localized('settings')}", f"📊 {localized('results')}"])
    with settings:
        render_capacitors()
        st.divider()
        render_target_section()
    with results:
        create_results_tab()


if __name__ == "__main__":
    main()
