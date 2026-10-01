"""Correctness checks.  Run with `python -m tests.test_core` (or pytest if installed)."""
import numpy as np

from prcert.audits import eb_radius, geom_radius, select
from prcert.geometry import (geodesic, inactive_rotation_pair, proj_dist, random_rotation, s_op,
                             sine_path_pair)
from prcert.sampling import CoupledSampler, IndepSampler, run_checkpoints
from prcert.simulators import planar_arm, sine_ridge


def _mc_risk(sim, U, n, rng):
    X = rng.standard_normal((n, sim.D)); Xi = rng.standard_normal((n, sim.D))
    XP = X @ U @ U.T + Xi - Xi @ U @ U.T
    z = 0.5 * (sim(X) - sim(XP)) ** 2
    return z.mean(), z.std() / np.sqrt(n)


def test_exact_risk_sine_and_arm():
    rng = np.random.default_rng(1)
    for sim, U in [(sine_ridge(3), sine_path_pair("first", "favorable", 0.3, 3)[1]),
                   (planar_arm(8), np.linalg.qr(rng.standard_normal((8, 3)))[0])]:
        m, se = _mc_risk(sim, U, 400_000, rng)
        assert abs(m - sim.risk(U)) < 5 * se, (sim.name, m, sim.risk(U), se)
    th = 0.7
    sim = sine_ridge(2)
    exact = np.exp(-1) * (np.sinh(1) - np.sinh(np.sin(th) ** 2))
    assert abs(sim.risk(sine_path_pair("first", "favorable", th - np.pi / 4, 2)[1]) - exact) < 1e-12


def test_reduced_sampler_matches_full_dimension():
    """Reduced q-dim sampling and naive D-dim sampling estimate the same exact gain."""
    rng = np.random.default_rng(2)
    sim = planar_arm(12)
    U0 = np.linalg.qr(rng.standard_normal((12, 2)))[0]
    U1 = random_rotation(U0, 0.4, rng)
    W = CoupledSampler(sim, U0, [U1]).draw(rng, 600_000)[:, 0]
    exact = sim.gain(U0, U1)
    assert abs(W.mean() - exact) < 5 * W.std() / np.sqrt(W.size)
    Wi = IndepSampler(sim, U0, U1).draw(rng, 600_000)[:, 0]
    assert abs(Wi.mean() - exact) < 5 * Wi.std() / np.sqrt(Wi.size)


def test_checkpoint_statistics():
    class Gauss:
        K = 1
        def draw(self, rng, n):
            return 2.0 + 3.0 * rng.standard_normal((n, 1))
    for chunk in (2 ** 20, 37):
        grid, m, v = run_checkpoints(Gauss(), [10, 40, 100], 3000, np.random.default_rng(3), chunk)
        assert np.allclose(m.mean(axis=0)[:, 0], 2.0, atol=0.05)
        assert np.allclose(m.var(axis=0)[:, 0] * grid, 9.0, rtol=0.1)
        assert np.allclose(v.mean(axis=0)[:, 0], 9.0, rtol=0.03)


def test_second_order_variance_limit():
    rng = np.random.default_rng(4)
    th = 0.01
    sim = sine_ridge(2)
    U0, U1 = sine_path_pair("second", "favorable", th, 2)
    W = CoupledSampler(sim, U0, [U1]).draw(rng, 2_000_000)[:, 0]
    ratio = W.var() / proj_dist(U0, U1) ** 2
    assert abs(ratio - (3 - 2 * np.exp(-4) - np.exp(-8)) / 8) < 0.01, ratio


def test_geometry():
    rng = np.random.default_rng(5)
    U0 = np.linalg.qr(rng.standard_normal((10, 3)))[0]
    U1 = random_rotation(U0, 0.6, rng)
    assert abs(proj_dist(U0, U1) - np.sqrt(6) * np.sin(0.6)) < 1e-10
    Um = geodesic(U0, U1, 0.5)
    assert abs(proj_dist(U0, Um) - np.sqrt(6) * np.sin(0.3)) < 1e-8
    A, B = inactive_rotation_pair(20, 3, 0.02, 0.2)
    sim = sine_ridge(20)
    a, b = sine_path_pair("first", "favorable", 0.02, 20)
    assert abs(sim.gain(A, B) - sim.gain(a, b)) < 1e-14
    assert abs(proj_dist(A, B) ** 2 - (2 * np.sin(0.02) ** 2 + 4 * np.sin(0.2) ** 2)) < 1e-12
    u0, u1 = sine_path_pair("first", "equal", 0.05, 3)
    assert abs(sim_gain := sine_ridge(3).gain(u0, u1)) < 1e-14 and s_op(u0, u1) > 0, sim_gain


def test_radii_validity_and_selection():
    rng = np.random.default_rng(6)
    sim = sine_ridge(3)
    delta = 0.05
    for move in ("harmful", "equal"):
        U0, U1 = sine_path_pair("first", move, 0.1, 3)
        grid, m, v = run_checkpoints(CoupledSampler(sim, U0, [U1]), [200, 2000], 2000, rng)
        d = proj_dist(U0, U1)
        fg = (m[..., 0] - geom_radius(4.0, 1.0, d, grid, 1, delta) > 0).mean()
        fe = (m[..., 0] - eb_radius(v[..., 0], grid, 4.0, 1, delta) > 0).mean()
        assert fg <= delta and fe <= delta, (move, fg, fe)
    mean = np.array([[0.3, 0.1], [-0.1, 0.05]]); rad = np.array([[0.2, 0.0], [0.0, 0.1]])
    assert list(select(mean, rad)) == [2, 0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
