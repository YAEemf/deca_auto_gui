"""スコア成分の意味（正規化・飽和・部品数の扱い）を確認する"""

import numpy as np

from deca_auto import evaluator
from deca_auto.config import UserConfig, WEIGHT_FIELDS

F = np.logspace(3, 8, 501)
TARGET = np.full_like(F, 10e-3)
MASK = np.ones_like(F, dtype=bool)


def _components(z_rows, counts, **overrides):
    cfg = UserConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    meta = evaluator.build_eval_metadata(F, MASK, TARGET, np)
    comps = evaluator.calculate_score_components_batch(
        np.asarray(z_rows, dtype=np.complex128), np.asarray(counts), meta, cfg, np)
    return comps, cfg


def test_component_names_follow_weight_fields():
    assert evaluator.SCORE_COMPONENTS == tuple(name.removeprefix("weight_") for name in WEIGHT_FIELDS)


def test_flat_impedance_levels():
    rows = [np.full_like(F, 10e-3 * r) for r in (0.1, 0.5, 1.0, 2.0)]
    comps, _ = _components(rows, [[1]] * 4)
    # 目標の1/2以下は飽和（過剰設計を評価しない）、目標で1、2倍（+6dB超過）で 1 + VIOLATION_GAIN
    expected = [0.0, 0.0, 1.0, 1.0 + evaluator.VIOLATION_GAIN]
    np.testing.assert_allclose(comps["max"], expected, atol=1e-9)
    np.testing.assert_allclose(comps["mean"], expected, atol=1e-9)
    np.testing.assert_allclose(comps["area"], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(comps["anti"], 0.0, atol=1e-12)
    np.testing.assert_allclose(comps["resonance"], 0.0, atol=1e-12)
    np.testing.assert_allclose(comps["flat"], 0.0, atol=1e-9)


def test_anti_resonance_peak_prominence():
    """+6dB の鋭いピークは anti ≈ 1、ignore_safe では目標以下のピークを無視する"""
    log_f = np.log10(F)
    bump = np.exp(-0.5 * ((log_f - 5.5) / 0.05) ** 2) * np.log10(2.0)
    below = 10e-3 * 0.25 * 10 ** bump   # 目標の1/4から+6dB（目標以下）
    above = 10e-3 * 0.9 * 10 ** bump    # 目標を超えるピーク
    comps, _ = _components([below, above], [[1], [1]])
    np.testing.assert_allclose(comps["anti"], [1.0, 1.0], rtol=0.02)
    comps, _ = _components([below, above], [[1], [1]], ignore_safe_anti_resonance=True)
    assert comps["anti"][0] == 0.0 and comps["anti"][1] > 0.95


def test_prominence_matches_scipy():
    """突出量/谷の深さは scipy.signal.peak_prominences と一致する（窓幅に依存しない）"""
    from scipy.signal import peak_prominences
    rng = np.random.default_rng(5)
    e = np.cumsum(rng.normal(size=(20, 300)), axis=1) * 0.02
    for peak in (True, False):
        x = e if peak else -e
        mask = np.zeros_like(e, dtype=bool)
        mask[:, 1:-1] = (x[:, 1:-1] >= x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
        got = evaluator._extremum_excursion(e, mask, peak, np)
        for row in range(e.shape[0]):
            idx = np.nonzero(mask[row])[0]
            np.testing.assert_allclose(got[row, idx], peak_prominences(x[row], idx)[0], atol=1e-12)


def test_wide_anti_resonance_is_detected():
    """1デケード以上の幅を持つ反共振（VRM-バルク間など）も突出量として評価される"""
    log_f = np.log10(F)
    wide = 10e-3 * 0.5 * 10 ** (np.exp(-0.5 * ((log_f - 5.5) / 0.6) ** 2) * np.log10(2.0))
    comps, _ = _components([wide], [[1]])
    np.testing.assert_allclose(comps["anti"], [1.0], rtol=0.05)


def test_parts_cost_independent_of_max_total_parts():
    counts = [[3, 2], [6, 4]]
    rows = [np.full_like(F, 1e-3)] * 2
    small, _ = _components(rows, counts, max_total_parts=10)
    large, _ = _components(rows, counts, max_total_parts=40)
    np.testing.assert_allclose(small["parts"], [0.5, 1.0])
    np.testing.assert_allclose(large["parts"], small["parts"])
    np.testing.assert_allclose(small["num_types"], [0.4, 0.4])


def test_overdesign_is_not_rewarded():
    """目標を大きく下回っても成分は飽和し、部品が多いほど総合スコアは悪化する"""
    rows = [np.full_like(F, 4e-3), np.full_like(F, 1e-3)]
    comps, cfg = _components(rows, [[4], [12]])
    totals = evaluator.weighted_sum(comps, cfg)
    assert totals[0] < totals[1]
