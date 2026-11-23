import numpy as np
import pytest

from deca_auto import evaluator
from deca_auto.config import UserConfig
from deca_auto.main import build_combination_space
from deca_auto.utils import transfer_to_host


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount() > 0
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA unavailable: {error}")
    if not available:
        pytest.skip("No CUDA device")
    return cupy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ignore_safe", [False, True])
@pytest.mark.parametrize("n_eval", [0, 1, 2, 257])
def test_score_components_match_numpy(cp, dtype, ignore_safe, n_eval, monkeypatch):
    rng = np.random.default_rng(734)
    f = np.geomspace(1e2, 1e8, 521, dtype=dtype)
    mask = np.zeros(f.shape, dtype=bool)
    mask[rng.choice(f.size, n_eval, replace=False)] = True
    target = rng.uniform(0.01, 0.2, f.size).astype(dtype)
    z_dtype = np.complex64 if dtype == np.float32 else np.complex128
    samples = rng.normal(size=(64, 2 * f.size, 2))
    z = (10 ** samples[..., 0] * np.exp(1j * samples[..., 1])).astype(z_dtype)
    z[0, :] = 0  # Exercise the magnitude floor.
    counts = rng.integers(0, 4, size=(64, 18), dtype=np.int32 if dtype == np.float32 else np.int64)
    config = UserConfig(ignore_safe_anti_resonance=ignore_safe)

    host_metadata = evaluator.build_eval_metadata(f, mask, target, np)
    expected = evaluator.calculate_score_components_batch(z[:, ::2], counts[:, ::2], host_metadata, config)
    gpu_metadata = evaluator.build_eval_metadata(cp.asarray(f), cp.asarray(mask), cp.asarray(target), cp)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("The array-based extrema calculation ran on CUDA")

    monkeypatch.setattr(evaluator, "_sliding_extreme", unexpected_fallback)
    actual = evaluator.calculate_score_components_batch(
        cp.asarray(z)[:, ::2], cp.asarray(counts)[:, ::2], gpu_metadata, config, cp,
    )
    assert actual.keys() == expected.keys()
    tolerance = 1e-4 if dtype == np.float32 else 1e-11
    for name, reference in expected.items():
        assert actual[name].dtype == dtype
        np.testing.assert_allclose(transfer_to_host(actual[name]), reference, rtol=tolerance, atol=tolerance,
                                   err_msg=name)


@pytest.mark.parametrize("ignore_safe", [False, True])
def test_score_peak_plateau_and_strict_threshold(cp, ignore_safe):
    threshold = evaluator.PEAK_MIN_DECADE
    e = np.array([
        [0, -0.5, -0.125, -0.125, -0.5, 0.25, 0],
        [0, threshold, 0, 0, -threshold, 0, 0],
        [0.5, 0.5, -0.5, 0.25, 0.25, 0.25, 0.25],
    ], dtype=np.float32)
    config = UserConfig(ignore_safe_anti_resonance=ignore_safe)
    actual = []
    for row in e:
        # log10(|Z|) is exactly zero, so the log target fixes the comparison values.
        metadata = {'indices': cp.arange(row.size, dtype=np.int64), 'n_eval': row.size,
                    'log_target': cp.asarray(-row), 'weights': cp.full(row.size, 1 / row.size, dtype=cp.float32),
                    'window': 2}
        components = evaluator.calculate_score_components_batch(
            cp.ones(row.size, dtype=cp.complex64), cp.array([1, 0], dtype=cp.int32), metadata, config, cp,
        )
        actual.append([transfer_to_host(components[name])[0] for name in ('anti', 'resonance')])

    expected = np.array([[0.25, 0] if ignore_safe else [0.625, 0.75], [0, 0], [0, 0.75]])
    np.testing.assert_allclose(np.array(actual), expected / evaluator.UNIT_DECADE, rtol=1e-6, atol=1e-6)


def test_unrank_matches_numpy_with_random_constraints(cp):
    rng = np.random.default_rng(281)
    nonempty = 0
    empty = 0
    for _ in range(64):
        n_types = int(rng.integers(1, 13))
        lo = rng.integers(0, 3, n_types)
        hi = lo + rng.integers(0, 5, n_types)
        min_total = int(rng.integers(0, hi.sum() + 1))
        max_total = int(rng.integers(min_total, hi.sum() + 1))
        space = build_combination_space(n_types, max_total, min_total, lo, hi)
        if space.total:
            nonempty += 1
            ranks = (np.arange(space.total, dtype=np.int64) if space.total <= 512 else
                     np.r_[0, space.total - 1, rng.integers(0, space.total, 512, dtype=np.int64)])
        else:
            empty += 1
            ranks = np.empty(0, dtype=np.int64)
        expected = space.unrank(ranks[::-1], np)
        actual = space.unrank(cp.asarray(ranks)[::-1], cp)
        assert actual.dtype == np.int32
        np.testing.assert_array_equal(transfer_to_host(actual), expected)
    assert nonempty > 30 and empty > 0


def test_unrank_with_int64_ranks(cp):
    space = build_combination_space(40, 20, 19)
    assert space.total > 2 ** 32
    ranks = np.array([0, 2 ** 32, space.total // 2, space.total - 1], dtype=np.int64)
    expected = space.unrank(ranks, np)
    for ranks_input in (cp.asarray(ranks), ranks, ranks.tolist()):
        np.testing.assert_array_equal(transfer_to_host(space.unrank(ranks_input, cp)), expected)
