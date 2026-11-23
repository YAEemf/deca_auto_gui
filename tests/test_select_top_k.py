"""Monte Carloを含む総合スコアのtop_k選択が、全候補にMCを適用した総当たりと一致することを確認する"""

import itertools

import numpy as np
import pytest

from deca_auto import evaluator, pdn
from deca_auto.config import UserConfig

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    BACKENDS = [np, cp]
except Exception:  # CuPy/GPUが無い環境ではNumPyのみ
    BACKENDS = [np]

CAPS = [
    dict(name="a", C=100e-9, ESR=20e-3, ESL=3e-10),
    dict(name="b", C=1e-6, ESR=10e-3, ESL=4e-10),
    dict(name="c", C=10e-6, ESR=5e-3, ESL=5e-10),
    dict(name="d", C=220e-6, ESR=12e-3, ESL=1.2e-9),
]
F = np.logspace(3, 8, 121)


def _to_host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


@pytest.mark.parametrize("xp", BACKENDS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("mc_enable", [True, False])
def test_select_top_k_matches_brute_force(xp, mc_enable):
    cfg = UserConfig()
    cfg.capacitors = [dict(c) for c in CAPS]
    cfg.mc_enable, cfg.mc_samples, cfg.weight_mc_worst = mc_enable, 8, 3.0
    cfg.tol_C, cfg.tol_ESR, cfg.tol_ESL = 0.3, 0.5, 0.3
    counts = np.array([c for c in itertools.product(range(5), repeat=len(CAPS)) if 0 < sum(c) <= 10], np.int32)
    w = 2 * np.pi * F
    z_c = {c["name"]: xp.asarray(c["ESR"] + 1j * w * c["ESL"] + 1 / (1j * w * c["C"])) for c in CAPS}
    rlc = {c["name"]: {"C": c["C"], "ESL": c["ESL"]} for c in CAPS}
    prepared = pdn.prepare_pdn_components(z_c, cfg, xp.asarray(F), xp, "complex128", rlc)
    meta = evaluator.build_eval_metadata(xp.asarray(F), xp.ones(F.size, dtype=bool), xp.full(F.size, 5e-3), xp)

    cv = xp.asarray(counts)
    scores, comps = evaluator.evaluate_combinations(pdn.calculate_pdn_impedance_batch(cv, prepared, xp), cv, meta, cfg, xp)
    mc_all = evaluator.monte_carlo_worst(cv, comps, prepared, meta, cfg, xp)
    totals_all = _to_host(scores + cfg.weight_mc_worst * mc_all)

    k = 10
    reference = np.argsort(totals_all, kind="stable")[:k]
    if mc_enable:  # MCで順位が入れ替わる条件で検証していること
        assert not np.array_equal(np.sort(reference), np.sort(np.argsort(_to_host(scores))[:k]))
    idx, totals, mc = evaluator.select_top_k(scores, comps, cv, k, None, prepared, meta, cfg, xp)
    np.testing.assert_allclose(_to_host(totals), totals_all[reference], rtol=1e-12)
    np.testing.assert_allclose(_to_host(mc), _to_host(mc_all)[idx], rtol=1e-12)

    # 小さなMCバッチで逐次評価しても、グローバルのk位で足切りしても結果は同じ
    threshold = float(totals_all[reference[6]])
    small_batch = pytest.MonkeyPatch.context()
    with small_batch as patch:
        patch.setattr(evaluator, "MC_BATCH_ELEMENTS", 1)
        idx2, totals2, _ = evaluator.select_top_k(scores, comps, cv, k, threshold, prepared, meta, cfg, xp)
    np.testing.assert_allclose(_to_host(totals2), totals_all[reference[:6]], rtol=1e-12)
    np.testing.assert_array_equal(idx2, idx[:6])
