"""GUIのデータエディタ（コンデンサリスト・カスタムマスク）の入力解釈と設定検証"""

import pandas as pd
import pytest

from deca_auto.config import UserConfig, config_error
from deca_auto.interface import (
    CAP_COLUMNS, MASK_COLUMNS, capacitor_row, parse_capacitors, parse_custom_mask,
)

SOURCE = [
    {"name": "C_a", "C": 1e-7, "ESR": 0.02, "ESL": 3e-10},
    {"name": "C_b", "C": 1e-5, "ESR": 0.01, "ESL": 4e-10, "MAX": 4},
]
COLUMNS = [*CAP_COLUMNS.values(), "usage_range", "_source"]


def cap_frame(extra_rows=()):
    """data_editorが返す形: 既存行は_sourceを持ち、＋で追加した行は全セルが<NA>"""
    rows = [dict(capacitor_row(cap), _source=str(i)) for i, cap in enumerate(SOURCE)]
    for extra in extra_rows:
        rows.append({column: extra.get(column) for column in COLUMNS})
    return pd.DataFrame(rows, columns=COLUMNS, dtype="string")


def test_unedited_rows_keep_original_values():
    assert parse_capacitors(cap_frame(), SOURCE) == SOURCE


def test_blank_added_row_is_ignored():
    # ＋を押した直後（何も入力していない行）でエラーにしない
    assert parse_capacitors(cap_frame([{}]), SOURCE) == SOURCE
    assert parse_capacitors(cap_frame([{}, {}]), SOURCE) == SOURCE


def test_added_row_is_parsed():
    caps = parse_capacitors(cap_frame([{"Name": "C_new", "C [F]": "2.2e-6", "usage_range": "1-3"}]), SOURCE)
    assert caps[-1] == {"name": "C_new", "C": pytest.approx(2.2e-6), "MIN": 1, "MAX": 3}


def test_added_row_without_capacitance_or_model_is_rejected():
    # 容量もモデルも無い行を既定値で黙って計算しない
    with pytest.raises(ValueError, match="容量 C またはモデルパス"):
        parse_capacitors(cap_frame([{"Name": "C_new"}]), SOURCE)


def test_model_path_only_row_takes_name_from_file():
    caps = parse_capacitors(cap_frame([{"Path": "model/GRM155.mod"}]), SOURCE)
    assert caps[-1] == {"name": "GRM155", "path": "model/GRM155.mod"}


def test_cleared_cells_and_duplicate_names():
    frame = cap_frame([{"Name": "C_a", "C [F]": "1e-6"}])
    frame.loc[1, "usage_range"] = ""  # MAX=4 を消す
    caps = parse_capacitors(frame, SOURCE)
    assert "MAX" not in caps[1]
    assert [cap["name"] for cap in caps] == ["C_a", "C_b", "C_a_1"]


@pytest.mark.parametrize("column, text", [("C [F]", "abc"), ("usage_range", "5-2"), ("ESR [Ω]", "inf")])
def test_invalid_cell_raises_value_error(column, text):
    frame = cap_frame()
    frame.loc[0, column] = text
    with pytest.raises(ValueError):
        parse_capacitors(frame, SOURCE)


def test_deleted_rows_are_dropped():
    assert parse_capacitors(cap_frame().iloc[[1]], SOURCE) == [SOURCE[1]]


def mask_frame(rows):
    return pd.DataFrame(rows, columns=MASK_COLUMNS, dtype="string")


def test_custom_mask_ignores_blank_rows_and_sorts():
    points = parse_custom_mask(mask_frame([("1e6", "0.02"), (None, None), ("1e3", "0.01")]))
    assert points == [(1e3, 0.01), (1e6, 0.02)]


@pytest.mark.parametrize("rows, message", [
    ([("1e3", None)], "両方を入力"),
    ([(None, None)], "1点以上"),
    ([], "1点以上"),
    ([("1e3", "-1")], "正の値"),
    ([("x", "1")], "数値"),
])
def test_custom_mask_errors(rows, message):
    with pytest.raises(ValueError, match=message):
        parse_custom_mask(mask_frame(rows))


def test_config_error_reports_reason():
    config = UserConfig()
    assert config_error(config) is None
    config.capacitors = [{"name": "C_x"}]
    assert "容量 C またはモデルパス" in config_error(config)
    config.capacitors = [{"name": "C_x", "path": "model/x.mod"}]
    assert config_error(config) is None
    config.target_impedance_mode, config.z_custom_mask = "custom", []
    assert "カスタムマスク" in config_error(config)
