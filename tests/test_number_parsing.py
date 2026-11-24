"""数値入力の表記（指数表記・SI接頭辞・RKM表記・単位記号・全角）"""

import pandas as pd
import pytest

from deca_auto.config import load_config
from deca_auto.interface import (
    CAP_COLUMNS, MASK_COLUMNS, parse_capacitors, parse_custom_mask, parse_usage_range,
)
from deca_auto.utils import parse_scientific_notation as parse


@pytest.mark.parametrize("text, expected", [
    # 従来の指数表記
    ("1e-9", 1e-9), ("10e3", 1e4), ("1.5E+6", 1.5e6), ("-3e-3", -3e-3), ("e3", 1e3), ("0.01", 0.01),
    # SI接頭辞（10進で桁をずらすので 220u は 220e-6 と完全に一致する）
    ("220u", 220e-6), ("220μ", 220e-6), ("220µ", 220e-6), ("4.7n", 4.7e-9), ("0.4n", 0.4e-9),
    ("10p", 10e-12), ("1f", 1e-15), ("10m", 10e-3), ("1k", 1e3), ("1K", 1e3),
    ("100M", 100e6), ("1meg", 1e6), ("1MEG", 1e6), ("1G", 1e9), ("2T", 2e12), (".5u", 0.5e-6),
    # 末尾の単位記号
    ("220uF", 220e-6), ("220 uF", 220e-6), ("0.4nH", 0.4e-9), ("10mΩ", 10e-3), ("10mohm", 10e-3),
    ("100MHz", 100e6), ("1.2V", 1.2), ("10A", 10.0), ("1F", 1.0), ("1fF", 1e-15),
    # RKM表記
    ("4u7", 4.7e-6), ("4n7", 4.7e-9), ("4k7", 4.7e3), ("2R2", 2.2),
    # 全角入力（IME）
    ("２２０ｕ", 220e-6), ("２２０μＦ", 220e-6), (" 47u ", 47e-6),
])
def test_accepted_notations(text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize("text", ["abc", "1x", "u", "1uu", "1mm", "1e", "4u7u", "1 2", "1%", "1.2.3", "--1", ""])
def test_rejected_notations(text):
    with pytest.raises(ValueError):
        parse(text)


def test_error_message_shows_examples():
    with pytest.raises(ValueError, match="2.2u"):
        parse("220x")


def test_toml_string_values_accept_prefixes(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[frequency]\nf_stop = "1G"\n'
        '[target]\ntarget_impedance_mode = "custom"\nz_custom_mask = [["1k", "8m"], ["100MHz", "0.2"]]\n'
        '[pdn]\nL_sN = "0.4nH"\nR_sN = "0.5m"\n'
        '[[capacitors]]\nname = "C_220u"\nC = "220u"\nESR = "10mΩ"\nESL = "4n7"\n',
        encoding="utf-8",
    )
    config = load_config(path, verbose=False, strict=True)
    assert config.f_stop == 1e9
    assert config.z_custom_mask == [(1e3, 8e-3), (100e6, 0.2)]
    assert (config.L_sN, config.R_sN) == (0.4e-9, 0.5e-3)
    assert config.capacitors == [{"name": "C_220u", "C": 220e-6, "ESR": 10e-3, "ESL": 4.7e-9}]


def test_gui_editors_accept_prefixes():
    columns = [*CAP_COLUMNS.values(), "usage_range", "_source"]
    added = {"Name": "C_bulk", "C [F]": "220u", "ESL [H]": "1.5nH", "usage_range": "１～３"}
    frame = pd.DataFrame([{column: added.get(column) for column in columns}], columns=columns, dtype="string")
    caps = parse_capacitors(frame, [])
    assert caps == [{"name": "C_bulk", "C": 220e-6, "ESL": 1.5e-9, "MIN": 1, "MAX": 3}]
    mask = parse_custom_mask(pd.DataFrame([("1k", "8m"), ("100M", "0.2")], columns=MASK_COLUMNS, dtype="string"))
    assert mask == [(1e3, 8e-3), (100e6, 0.2)]


@pytest.mark.parametrize("text, expected", [
    ("3", (3, 3)), ("-5", (None, 5)), ("2-", (2, None)), ("3-5", (3, 5)),
    ("３－５", (3, 5)), ("3～5", (3, 5)), ("3〜5", (3, 5)), ("3~5", (3, 5)), ("", (None, None)),
])
def test_usage_range_notations(text, expected):
    assert parse_usage_range(text) == expected


@pytest.mark.parametrize("text", ["-", "5-2", "--5", "a", "1.5"])
def test_usage_range_rejected(text):
    with pytest.raises(ValueError, match="使用数範囲"):
        parse_usage_range(text)
