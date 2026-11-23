"""PDNラダー計算を、回路図（images/PDN.jpg）を節点解析(MNA)で直接解いた参照解と比較する"""

import numpy as np
import pytest

from deca_auto import pdn
from deca_auto.config import UserConfig

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    BACKENDS = [np, cp]
except Exception:  # CuPy/GPUが無い環境ではNumPyのみ
    BACKENDS = [np]

CAPS = [  # 容量の小さい順
    dict(name="a", C=47e-9, ESR=20e-3, ESL=2e-10, L_mnt=1e-10),
    dict(name="b", C=1e-6, ESR=12e-3, ESL=3e-10),  # L_mnt 未指定 -> L_mntN
    dict(name="c", C=10e-6, ESR=10e-3, ESL=4e-10, L_mnt=3e-10),
    dict(name="d", C=100e-6, ESR=100e-3, ESL=1.5e-9, L_mnt=0.0),
]
COUNTS = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [2, 1, 0, 1], [3, 2, 2, 3], [0, 5, 0, 0]]
F = np.logspace(2, 9, 301)
W = 2 * np.pi * F


def _config(**overrides):
    cfg = UserConfig()
    cfg.capacitors = [dict(c) for c in CAPS]
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _z_c(cap):
    return cap["ESR"] + 1j * W * cap["ESL"] + 1 / (1j * W * cap["C"])


def _mna(cfg, counts, caps=CAPS):
    """Load(ポート0)に1Aを注入したときのポート電圧"""
    zv = cfg.R_v + 1j * W * cfg.L_v
    zs = cfg.R_s + 1j * W * cfg.L_s
    zsn = cfg.R_sN + 1j * W * cfg.L_sN
    planar = W * cfg.C_p * (cfg.tan_delta_p + 1j)
    yp = planar / (1 + cfg.R_p * planar)
    yvrm = 1 / (cfg.R_vrm + 1j * W * cfg.L_vrm) + yp
    stages = [c for c, n in zip(caps, counts) for _ in range(n)]
    n_nodes = 3 + len(stages)
    out = np.empty(len(F), complex)
    for i in range(len(F)):
        Y = np.zeros((n_nodes, n_nodes), complex)

        def series(a, b, z):
            y = 1 / z
            Y[a, a] += y
            Y[b, b] += y
            Y[a, b] -= y
            Y[b, a] -= y

        series(0, 1, zv[i])
        series(1, 2, zs[i])
        Y[2, 2] += yp[i]
        prev = 2
        for k, cap in enumerate(stages):
            node = 3 + k
            series(prev, node, zsn[i])
            l_mnt = cfg.L_mntN if cap.get("L_mnt") is None else cap["L_mnt"]
            Y[node, node] += 1 / (_z_c(cap)[i] + 1j * W[i] * l_mnt)
            prev = node
        Y[prev, prev] += yvrm[i]
        rhs = np.zeros(n_nodes, complex)
        rhs[0] = 1
        out[i] = np.linalg.solve(Y, rhs)[0]
    return out


def _to_host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def _prepare(xp, cfg, dtype="complex128"):
    zdict = {c["name"]: xp.asarray(_z_c(c)) for c in CAPS}
    rlc = {c["name"]: {"C": c["C"], "ESL": c["ESL"]} for c in CAPS}
    return pdn.prepare_pdn_components(zdict, cfg, xp.asarray(F), xp, dtype, rlc)


@pytest.mark.parametrize("xp", BACKENDS, ids=lambda m: m.__name__)
def test_ladder_matches_mna(xp):
    cfg = _config()
    z = _to_host(pdn.calculate_pdn_impedance_batch(xp.asarray(np.array(COUNTS, np.int32)), _prepare(xp, cfg), xp))
    for counts, row in zip(COUNTS, z):
        ref = _mna(cfg, counts)
        assert np.max(np.abs(row - ref) / np.abs(ref)) < 1e-9, counts


@pytest.mark.parametrize("xp", BACKENDS, ids=lambda m: m.__name__)
def test_complex64_close_to_complex128(xp):
    cfg = _config()
    cv = xp.asarray(np.array(COUNTS, np.int32))
    z64 = _to_host(pdn.calculate_pdn_impedance_batch(cv, _prepare(xp, cfg, "complex64"), xp))
    z128 = _to_host(pdn.calculate_pdn_impedance_batch(cv, _prepare(xp, cfg), xp))
    assert np.max(np.abs(z64 - z128) / np.abs(z128)) < 1e-4


@pytest.mark.parametrize("xp", BACKENDS, ids=lambda m: m.__name__)
def test_monte_carlo_zero_tolerance_equals_nominal(xp):
    cfg = _config(tol_C=0.0, tol_ESR=0.0, tol_ESL=0.0, mlcc_derating=0.0)
    cv = xp.asarray(np.array(COUNTS, np.int32))
    prep = _prepare(xp, cfg)
    nominal = _to_host(pdn.calculate_pdn_impedance_batch(cv, prep, xp))
    mc = _to_host(pdn.calculate_pdn_impedance_monte_carlo(cv, prep, cfg, 3, xp))
    assert np.max(np.abs(mc - nominal[:, None, :]) / np.abs(nominal[:, None, :])) < 1e-9


@pytest.mark.parametrize("xp", BACKENDS, ids=lambda m: m.__name__)
def test_monte_carlo_derating_equals_reduced_capacitance(xp):
    cfg = _config(tol_C=0.0, tol_ESR=0.0, tol_ESL=0.0, mlcc_derating=0.2)
    counts = COUNTS[3]
    mc = _to_host(pdn.calculate_pdn_impedance_monte_carlo(
        xp.asarray(np.array([counts], np.int32)), _prepare(xp, cfg), cfg, 2, xp))
    ref = _mna(cfg, counts, [dict(c, C=c["C"] * 0.8) for c in CAPS])
    assert np.max(np.abs(mc[0, 0] - ref) / np.abs(ref)) < 1e-9


def test_monte_carlo_is_backend_and_batch_independent():
    """同じ組み合わせは、バッチ内の位置やバックエンドに依らず同じ公差サンプルになる"""
    cfg = _config()
    results = []
    for xp in BACKENDS:
        prep = _prepare(xp, cfg)
        both = _to_host(pdn.calculate_pdn_impedance_monte_carlo(
            xp.asarray(np.array([COUNTS[3], COUNTS[4]], np.int32)), prep, cfg, 4, xp))
        alone = _to_host(pdn.calculate_pdn_impedance_monte_carlo(
            xp.asarray(np.array([COUNTS[4]], np.int32)), prep, cfg, 4, xp))
        np.testing.assert_allclose(both[1], alone[0], rtol=1e-12)
        results.append(alone[0])
    for other in results[1:]:
        np.testing.assert_allclose(other, results[0], rtol=1e-9)
