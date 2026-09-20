import _thread
import itertools
import os
import re
import signal
import sys
import threading
import time
import unittest
import warnings

import numpy as np

import bkgco
from bkgco import GCO


def data_energy(D, l):
    return int(D[np.arange(D.shape[0]), l].sum()) if D.dtype.kind == "i" else float(
        D[np.arange(D.shape[0]), l].sum()
    )


def smooth_energy(V, edges, w, l):
    return sum(w[k] * V[l[i], l[j]] for k, (i, j) in enumerate(edges))


def total_energy(D, V, edges, w, l, label_costs=None):
    e = data_energy(D, l) + smooth_energy(V, edges, w, l)
    if label_costs is not None:
        e += sum(c for lab, c in enumerate(label_costs) if lab in set(l.tolist()))
    return e


def brute_force(D, V, edges, w, label_costs=None):
    ns, nl = D.shape
    best, arg = None, None
    for l in itertools.product(range(nl), repeat=ns):
        l = np.array(l)
        e = total_energy(D, V, edges, w, l, label_costs)
        if best is None or e < best:
            best, arg = e, l
    return best, arg


def grid_edges(h, w):
    e = []
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                e.append((y * w + x, y * w + x + 1))
            if y + 1 < h:
                e.append((y * w + x, (y + 1) * w + x))
    return e


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def readme_python_blocks():
    with open(os.path.join(REPO_ROOT, "README.md")) as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


class TestReadmeExamples(unittest.TestCase):
    """Every ```python block in README.md must run and produce what it claims."""

    def test_examples_run_and_match_their_comments(self):
        blocks = readme_python_blocks()
        self.assertTrue(blocks, "no python examples found in README.md")
        checked = set()
        for src in blocks:
            ns = {}
            exec(compile(src, "README.md", "exec"), ns)
            if "e_min" in ns:                                  # BKEnergy example
                checked.add("BKEnergy")
                self.assertEqual(ns["e_min"], 3)
                np.testing.assert_array_equal(ns["e_sol"], [0, 1, 1])
                self.assertEqual(ns["e_sol"].dtype, np.uint8)
            if "l" in ns:                                      # GCO example
                checked.add("GCO")
                self.assertEqual(ns["e"], 9)
                np.testing.assert_array_equal(ns["l"], [2, 2, 0, 0])
                self.assertEqual(ns["l"].dtype, np.int32)
        self.assertEqual(checked, {"BKEnergy", "GCO"}, "an example stopped defining its results")

    def test_gco_example_result_is_optimal(self):
        """Brute-force the very problem the README example builds (D, V, N, W come
        out of the executed block, so this follows any edit to it)."""
        ns = {}
        for src in readme_python_blocks():
            if "GCO(" in src:
                exec(compile(src, "README.md", "exec"), ns)
        D, V, N, W = ns["D"], ns["V"], ns["N"], ns["W"]
        edges = [tuple(n) for n in N]
        want, arg = brute_force(np.array(D), np.array(V), edges, W)
        self.assertEqual(ns["e"], want)
        np.testing.assert_array_equal(ns["l"], arg)

    def test_bkenergy_example_result_is_optimal(self):
        # the energy spelled out in that example's comment
        f = lambda x, y, z: 5 * x + 2 * y + 9 * (1 - z) + (1 - x) * y + 4 * (1 - y) * z
        best = min((f(*b), b) for b in itertools.product((0, 1), repeat=3))
        self.assertEqual(best, (3, (0, 1, 1)))


class TestBasics(unittest.TestCase):
    def test_readme_example(self):
        D = np.array([[0, 3, 5], [9, 0, 9], [2, 3, 0], [0, 3, 5]], np.int32)
        V = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], np.int32)
        with GCO(4, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_neighbors(np.array([[0, 1], [1, 2], [2, 3]]), [1, 1, 2])
            self.assertEqual(gc.expansion(), 4)
            np.testing.assert_array_equal(gc.get_labeling(), [0, 1, 0, 0])
            self.assertEqual((gc.compute_energy(), gc.data_energy(), gc.smooth_energy()), (4, 2, 2))
            self.assertEqual(gc.label_energy(), 0)

    def test_no_costs(self):
        with GCO(4, 3) as gc:
            self.assertEqual(gc.compute_energy(), 0)
            np.testing.assert_array_equal(gc.get_labeling(), [0, 0, 0, 0])
            gc.expansion()
            self.assertEqual(gc.compute_energy(), 0)

    def test_labeling_roundtrip(self):
        with GCO(5, 4) as gc:
            gc.set_labeling([3, 2, 1, 0, 3])
            np.testing.assert_array_equal(gc.get_labeling(), [3, 2, 1, 0, 3])
            gc.set_labeling([3, 0, 1, 0, 3])
            np.testing.assert_array_equal(gc.get_labeling(2, 2), [1, 0])
            np.testing.assert_array_equal(gc.get_labeling(3), [0, 3])

    def test_attributes(self):
        with GCO(6, 3) as gc:
            self.assertEqual((gc.num_sites, gc.num_labels, gc.is_grid), (6, 3, False))
        with GCO.grid((2, 3), 4) as gc:
            self.assertEqual((gc.num_sites, gc.num_labels, gc.is_grid, gc.shape), (6, 4, True, (2, 3)))


class TestOptimality(unittest.TestCase):
    """With 2 labels a single expansion/swap move is a global min-cut, so the
    result must equal brute force."""

    def _random_problem(self, rng, ns, nl, ne, dtype=np.int32):
        D = rng.integers(0, 20, size=(ns, nl)).astype(dtype)
        base = rng.integers(1, 5)
        V = (base * (1 - np.eye(nl))).astype(dtype)
        edges = sorted({tuple(sorted(rng.choice(ns, 2, replace=False))) for _ in range(ne)})
        w = rng.integers(1, 4, size=len(edges)).astype(dtype)
        return D, V, edges, w

    def test_two_labels_exact(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            D, V, edges, w = self._random_problem(rng, 10, 2, 14)
            opt, _ = brute_force(D, V, edges, w)
            for algo in ("expansion", "swap"):
                with GCO(10, 2) as gc:
                    gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(np.array(edges), w)
                    gc.set_labeling(np.zeros(10, np.int32))
                    getattr(gc, algo)()
                    l = gc.get_labeling()
                self.assertEqual(total_energy(D, V, edges, w, l), opt, "%s trial %d" % (algo, trial))

    def test_multilabel_within_factor_two(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            D, V, edges, w = self._random_problem(rng, 8, 3, 10)
            opt, _ = brute_force(D, V, edges, w)
            with GCO(8, 3) as gc:
                gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(np.array(edges), w)
                e = gc.expansion()
                self.assertEqual(e, gc.compute_energy())
                self.assertEqual(e, total_energy(D, V, edges, w, gc.get_labeling()))
                self.assertLessEqual(e, 2 * opt)

    def test_alpha_expansion_single_move(self):
        D = np.array([[5, 0, 5], [5, 5, 0], [0, 5, 5]], np.int32)
        V = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], np.int32)
        with GCO(3, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(np.array([[0, 1], [1, 2]]))
            gc.set_labeling([0, 0, 0])
            self.assertTrue(gc.alpha_expansion(1))
            self.assertFalse(gc.alpha_expansion(1))
            gc.set_labeling([1, 1, 2])
            gc.alpha_beta_swap(1, 2)
            self.assertEqual(gc.compute_energy(), total_energy(D, V, [(0, 1), (1, 2)], [1, 1], gc.get_labeling()))


class TestGrid(unittest.TestCase):
    def test_grid_energy_matches_4connected(self):
        rng = np.random.default_rng(2)
        h, w = 4, 5
        D = rng.integers(0, 10, size=(h * w, 3)).astype(np.int32)
        V = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], np.int32)
        l = rng.integers(0, 3, size=h * w).astype(np.int32)
        edges = grid_edges(h, w)
        with GCO.grid((h, w), 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_labeling(l)
            self.assertEqual(gc.data_energy(), data_energy(D, l))
            self.assertEqual(gc.smooth_energy(), smooth_energy(V, edges, np.ones(len(edges), int), l))
            gc.expansion()
            l2 = gc.get_labeling().ravel()
            self.assertEqual(gc.compute_energy(), total_energy(D, V, edges, np.ones(len(edges), int), l2))

    def test_grid_vh_weights(self):
        rng = np.random.default_rng(3)
        h, w = 3, 4
        D = rng.integers(0, 10, size=(h, w, 2)).astype(np.int32)
        V = np.array([[0, 1], [1, 0]], np.int32)
        vc = rng.integers(1, 5, size=(h, w)).astype(np.int32)
        hc = rng.integers(1, 5, size=(h, w)).astype(np.int32)
        l = rng.integers(0, 2, size=(h, w)).astype(np.int32)
        expect = 0
        for y in range(h):
            for x in range(w):
                if x + 1 < w:
                    expect += hc[y, x] * V[l[y, x], l[y, x + 1]]
                if y + 1 < h:
                    expect += vc[y, x] * V[l[y, x], l[y + 1, x]]
        with GCO.grid((h, w), 2) as gc:
            gc.set_data_cost(D).set_smooth_cost_vh(V, vc, hc)
            gc.set_labeling(l)
            self.assertEqual(gc.smooth_energy(), expect)

    def test_grid_denoising(self):
        h, w = 6, 7
        img = np.zeros((h, w), np.int32)
        img[:, w // 2:] = 9
        D = np.stack([(img - 0) ** 2, (img - 9) ** 2], axis=2).astype(np.int32)
        V = (4 * (1 - np.eye(2))).astype(np.int32)
        with GCO.grid((h, w), 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.expansion()
            l = gc.get_labeling().reshape(h, w)
        np.testing.assert_array_equal(l, (img == 9).astype(np.int32))


class TestLabelCosts(unittest.TestCase):
    def test_uniform_label_cost(self):
        D = np.array([[0, 1, 9], [1, 0, 9], [9, 9, 0]], np.int32)
        V = np.zeros((3, 3), np.int32)
        with GCO(3, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_neighbors(np.array([[0, 1], [1, 2]]))
            gc.set_label_cost(0)
            gc.expansion()
            self.assertEqual(gc.compute_energy(), 0)  # labels 0,1,2 free
            gc.set_label_cost(5)
            gc.set_labeling([0, 1, 2])
            self.assertEqual(gc.label_energy(), 15)
            gc.expansion()
            self.assertEqual(gc.compute_energy(), min(
                total_energy(D, V, [(0, 1), (1, 2)], [1, 1], np.array(l), [5, 5, 5])
                for l in itertools.product(range(3), repeat=3)))

    def test_label_cost_array_and_subset(self):
        D = np.zeros((2, 3), np.int32)
        V = np.zeros((3, 3), np.int32)
        with GCO(2, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_label_cost(np.array([1, 2, 3], np.int32))
            gc.set_labeling([0, 2])
            self.assertEqual(gc.label_energy(), 4)
            gc.set_label_cost(7, labels=[1, 2])
            self.assertEqual(gc.label_energy(), 4 + 7)
            gc.set_labeling([0, 0])
            self.assertEqual(gc.label_energy(), 1)


class TestSparseDataCost(unittest.TestCase):
    def test_sparse_matches_dense(self):
        rng = np.random.default_rng(4)
        ns, nl = 12, 3
        D = rng.integers(0, 15, size=(ns, nl)).astype(np.int32)
        V = (2 * (1 - np.eye(nl))).astype(np.int32)
        edges = np.array(grid_edges(3, 4))
        dense = GCO(ns, nl)
        dense.set_data_cost(D).set_smooth_cost(V).set_neighbors(edges)
        e_dense = dense.expansion()

        sparse = GCO(ns, nl)
        sparse.set_smooth_cost(V).set_neighbors(edges)
        sites = np.arange(ns, dtype=np.int32)
        for l in range(nl):
            sparse.set_data_cost_sparse(l, sites, np.ascontiguousarray(D[:, l]))
        e_sparse = sparse.expansion()
        self.assertEqual(e_dense, e_sparse)
        np.testing.assert_array_equal(dense.get_labeling(), sparse.get_labeling())
        dense.close()
        sparse.close()

    def test_sparse_infeasible_sites(self):
        with GCO(4, 2) as gc:
            gc.set_data_cost_sparse(0, np.array([0, 1], np.int32), np.array([1, 1], np.int32))
            gc.set_data_cost_sparse(1, np.array([0, 1, 2, 3], np.int32), np.array([5, 5, 5, 5], np.int32))
            gc.expansion()
            np.testing.assert_array_equal(gc.get_labeling(), [0, 0, 1, 1])

    def test_sparse_errors(self):
        with GCO(4, 2) as gc:
            with self.assertRaises(RuntimeError):
                gc.set_data_cost_sparse(0, np.array([1, 1], np.int32), np.array([1, 1], np.int32))
            with self.assertRaises(RuntimeError):
                gc.set_data_cost_sparse(0, np.array([9], np.int32), np.array([1], np.int32))
            with self.assertRaises(ValueError):
                gc.set_data_cost_sparse(5, np.array([0], np.int32), np.array([1], np.int32))
        with GCO(3, 2) as gc:
            gc.set_data_cost(np.ones((3, 2), np.int32))
            with self.assertRaises(RuntimeError):
                gc.set_data_cost_sparse(0, np.array([0], np.int32), np.array([1], np.int32))


class TestFloat64(unittest.TestCase):
    def test_matches_int_for_integral_costs(self):
        rng = np.random.default_rng(5)
        D = rng.integers(0, 20, size=(9, 3)).astype(np.int32)
        V = (2 * (1 - np.eye(3))).astype(np.int32)
        edges = np.array(grid_edges(3, 3))
        res = {}
        for dt in (np.int32, np.float64):
            with GCO(9, 3, dtype=dt) as gc:
                gc.set_data_cost(D.astype(dt)).set_smooth_cost(V.astype(dt))
                gc.set_neighbors(edges, np.ones(len(edges), dt))
                res[dt] = (gc.expansion(), gc.get_labeling())
        self.assertEqual(res[np.int32][0], res[np.float64][0])
        np.testing.assert_array_equal(res[np.int32][1], res[np.float64][1])
        self.assertIsInstance(res[np.float64][0], float)

    def test_fractional_costs(self):
        D = np.array([[0.0, 0.6], [0.6, 0.0]])
        V = np.array([[0.0, 0.5], [0.5, 0.0]])
        with GCO(2, 2, dtype=np.float64) as gc:
            gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(np.array([[0, 1]]))
            gc.expansion()
            self.assertAlmostEqual(gc.compute_energy(), 0.5)
            np.testing.assert_array_equal(gc.get_labeling(), [0, 1])
            gc.set_smooth_cost(np.array([[0.0, 0.7], [0.7, 0.0]]))
            gc.expansion()
            self.assertAlmostEqual(gc.compute_energy(), 0.6)
            self.assertEqual(len(set(gc.get_labeling().tolist())), 1)

    def test_int_dtype_rejects_fractional_scalars(self):
        with GCO(2, 2) as gc:
            with self.assertRaises(ValueError):
                gc.set_label_cost(1.5)
            gc.set_label_cost(2.0)


class TestDtypeDispatch(unittest.TestCase):
    """One module holds both instantiations; the term type is picked at runtime."""

    def test_inferred_from_first_cost_array(self):
        cases = [(np.float32, np.float64, 2), (np.float64, np.float64, 2),
                 (np.int64, np.int64, 1), (np.int32, np.int32, 0), (np.int16, np.int32, 0)]
        for src, want, code in cases:
            with GCO(4, 2) as gc:
                self.assertIsNone(gc.dtype)
                gc.set_data_cost(np.zeros((4, 2), src))
                self.assertEqual(gc.dtype, np.dtype(want), src)
                self.assertEqual(gc._h.dtype_code, code)
                self.assertEqual(bool(gc._h.is_float64), want is np.float64)

    def test_explicit_dtype_wins(self):
        with GCO(4, 2, dtype=np.float64) as gc:
            gc.set_data_cost(np.zeros((4, 2), np.int32))
            self.assertEqual(gc.dtype, np.dtype(np.float64))
            self.assertIsInstance(gc.compute_energy(), float)

    def test_dtype_is_sticky(self):
        with GCO(2, 2) as gc:
            gc.set_data_cost(np.zeros((2, 2), np.int32))
            gc.set_smooth_cost(np.array([[0.0, 1.0], [1.0, 0.0]]))  # integral floats convert
            self.assertEqual(gc.dtype, np.dtype(np.int32))
            with self.assertRaises(ValueError):
                gc.set_smooth_cost(np.array([[0.0, 0.5], [0.5, 0.0]]))

    def test_both_types_live_in_one_module(self):
        from bkgco import _bkgco

        rng = np.random.default_rng(11)
        D = rng.integers(0, 20, size=(16, 3)).astype(np.int32)
        V = (2 * (1 - np.eye(3))).astype(np.int32)
        edges = np.array(grid_edges(4, 4))
        gi = GCO(16, 3, np.int32)
        gf = GCO(16, 3, np.float64)
        for gc in (gi, gf):
            gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(edges)
        ei, ef = gi.expansion(), gf.expansion()  # interleaved use of both backends
        self.assertEqual(ei, ef)
        np.testing.assert_array_equal(gi.get_labeling(), gf.get_labeling())
        self.assertIsInstance(ei, int)
        self.assertIsInstance(ef, float)
        self.assertIs(type(gi._h), type(gf._h))
        self.assertEqual(_bkgco.GCO.__module__, "bkgco._bkgco")
        gi.close()
        gf.close()

    def test_int64_terms_beyond_int32_range(self):
        big = 3 * 10 ** 9  # > 2**31
        D = np.array([[0, big], [big, 0]], np.int64)
        with GCO(2, 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(np.zeros((2, 2), np.int64))
            self.assertEqual(gc.dtype, np.dtype(np.int64))
            self.assertEqual(gc.expansion(), 0)
            gc.set_labeling([1, 1])
            self.assertEqual(gc.compute_energy(), big)
        with GCO(2, 2, dtype=np.int32) as gc:  # same costs do not fit int32
            with self.assertRaises(ValueError):
                gc.set_data_cost(D)

    def test_max_energy_term_per_dtype(self):
        self.assertEqual(bkgco.max_energy_term(np.int32), 10 ** 7)
        self.assertEqual(bkgco.max_energy_term(np.int64), 10 ** 12)
        self.assertEqual(bkgco.max_energy_term(np.float64), 1e7)
        self.assertEqual(bkgco.MAX_ENERGY_TERM, bkgco.max_energy_term(np.int32))
        D = np.full((2, 2), bkgco.max_energy_term(np.int64) + 1, np.int64)
        with GCO(2, 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(np.array([[0, 1], [1, 0]], np.int64))
            gc.set_neighbors(np.array([[0, 1]]))
            with self.assertRaises(RuntimeError):
                gc.expansion()

    def test_int64_scalar_costs(self):
        with GCO(2, 2, dtype=np.int64) as gc:
            gc.set_data_cost(np.zeros((2, 2), np.int64))
            gc.set_label_cost(5 * 10 ** 9)
            gc.set_labeling([0, 0])
            self.assertEqual(gc.label_energy(), 5 * 10 ** 9)
        with GCO(2, 2, dtype=np.int32) as gc:
            with self.assertRaises(ValueError):
                gc.set_label_cost(5 * 10 ** 9)

    def test_grid_infers_float64_from_costs(self):
        u = np.zeros((4, 5, 2), np.float64)
        u[:, :, 1] = 0.5
        with GCO.grid((4, 5), 2) as gc:
            gc.set_data_cost(u).set_smooth_cost(np.zeros((2, 2)))
            self.assertEqual(gc.dtype, np.dtype(np.float64))
            gc.expansion()
            np.testing.assert_array_equal(gc.get_labeling(), np.zeros(20, np.int32))


class TestZeroCopy(unittest.TestCase):
    def test_data_cost_is_not_copied(self):
        D = np.zeros((3, 2), np.int32)
        V = np.zeros((2, 2), np.int32)
        with GCO(3, 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_labeling([1, 1, 1])
            self.assertEqual(gc.data_energy(), 0)
            D[:, 1] = 4  # in-place change must be visible to the library
            gc.set_labeling([1, 1, 1])  # invalidates the cached per-site costs
            self.assertEqual(gc.data_energy(), 12)

    def test_set_data_cost_invalidates_the_cache(self):
        """The README tells users to call set_data_cost(D) again after mutating D."""
        D = np.zeros((3, 2), np.int32)
        with GCO(3, 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(np.zeros((2, 2), np.int32))
            gc.set_labeling([1, 1, 1])
            self.assertEqual(gc.data_energy(), 0)
            D[:, 1] = 4
            gc.set_data_cost(D)
            self.assertEqual(gc.data_energy(), 12)

    def test_buffer_is_pinned(self):
        D = np.zeros((3, 2), np.int32)
        gc = GCO(3, 2)
        gc.set_data_cost(D)
        with self.assertRaises(ValueError):
            D.resize((6, 2))  # exported buffer must block resizing
        gc.close()

    def test_conversion_warns_for_large_arrays(self):
        n = 1 << 20
        D = np.zeros((n, 2), np.int64)
        with GCO(n, 2, dtype=np.int32) as gc:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gc.set_data_cost(D)
            self.assertTrue(any("converting" in str(c.message) for c in caught))

    def test_no_warning_for_matching_dtype(self):
        n = 1 << 20
        D = np.zeros((n, 2), np.int32)
        with GCO(n, 2) as gc:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gc.set_data_cost(D)
            self.assertEqual([str(c.message) for c in caught], [])


class TestLabelOrder(unittest.TestCase):
    def test_explicit_and_random_order(self):
        rng = np.random.default_rng(6)
        D = rng.integers(0, 20, size=(9, 3)).astype(np.int32)
        V = (2 * (1 - np.eye(3))).astype(np.int32)
        edges = np.array(grid_edges(3, 3))
        energies = []
        for order in ([0, 1, 2], [2, 1, 0], None):
            with GCO(9, 3) as gc:
                gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(edges)
                gc.set_label_order(order) if order else gc.set_label_order(random=True)
                energies.append(gc.expansion())
        self.assertEqual(len(set(energies)), 1)

    def test_restricted_label_subset(self):
        D = np.array([[0, 9, 4], [0, 9, 4]], np.int32)
        V = (1 - np.eye(3)).astype(np.int32)
        with GCO(2, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V).set_neighbors(np.array([[0, 1]]))
            gc.set_labeling([1, 1])
            gc.set_label_order([2])  # only label 2 may be expanded onto
            gc.expansion()
            np.testing.assert_array_equal(gc.get_labeling(), [2, 2])


class TestErrors(unittest.TestCase):
    def test_construction(self):
        with self.assertRaises(ValueError):
            GCO(0, 2)
        with self.assertRaises(ValueError):
            GCO(4, 1)
        with self.assertRaises(ValueError):
            GCO(5, 2, shape=(2, 2))
        with self.assertRaises(TypeError):
            GCO(4, 2, dtype=np.float32)
        with self.assertRaises(TypeError):
            GCO(4, 2, dtype=np.int16)

    def test_shape_mismatch(self):
        with GCO(4, 3) as gc:
            with self.assertRaises(ValueError):
                gc.set_data_cost(np.zeros((4, 2), np.int32))
            with self.assertRaises(ValueError):
                gc.set_data_cost(np.zeros((3, 3), np.int32))
            with self.assertRaises(ValueError):
                gc.set_smooth_cost(np.zeros((4, 3), np.int32))
            with self.assertRaises(ValueError):
                gc.set_labeling([0, 1, 2])
            with self.assertRaises(ValueError):
                gc.set_labeling([0, 1, 2, 3])

    def test_graph_kind(self):
        with GCO.grid((2, 2), 2) as gc:
            with self.assertRaises(RuntimeError):
                gc.set_neighbors(np.array([[0, 1]]))
        with GCO(4, 2) as gc:
            with self.assertRaises(RuntimeError):
                gc.set_smooth_cost_vh(np.zeros((2, 2), np.int32), np.ones(4, np.int32), np.ones(4, np.int32))

    def test_bad_edges(self):
        with GCO(4, 2) as gc:
            with self.assertRaises(ValueError):
                gc.set_neighbors(np.array([[0, 0]]))
            with self.assertRaises(ValueError):
                gc.set_neighbors(np.array([[0, 4]]))
            with self.assertRaises(ValueError):
                gc.set_neighbors(np.array([[0, 1, 2]]))
            with self.assertRaises(ValueError):
                gc.set_neighbors((np.array([0, 1]), np.array([1])))

    def test_out_of_range(self):
        with GCO(4, 2) as gc:
            with self.assertRaises(ValueError):
                gc.set_labeling([0, 2, 0, 0])       # label out of range
            with self.assertRaises(ValueError):
                gc.alpha_expansion(2)
            with self.assertRaises(ValueError):
                gc.alpha_beta_swap(0, 0)
            with self.assertRaises(ValueError):
                gc.get_labeling(3, 2)
            with self.assertRaises(ValueError):
                gc.verbosity = 3

    def test_reusable_after_error_mid_move(self):
        """An exception thrown mid-expansion must leave the object consistent:
        stale internal state used to make the next move build out-of-range graph
        edges (wild write with NDEBUG, segfault on Linux)."""
        n = 20000
        D = np.zeros((n, 2), np.int32)
        D[:, 1] = 1
        V = np.array([[0, 2], [2, 0]], np.int32)
        with GCO(n, 2) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_neighbors((np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)))
            D[n // 2, 1] = bkgco.MAX_ENERGY_TERM + 1  # blows up while setting up the move
            gc.set_data_cost(D)
            with self.assertRaises(RuntimeError):
                gc.alpha_expansion(1)  # every site is active, then it throws
            D[n // 2, 1] = 1
            gc.set_data_cost(D)
            labeling = np.ones(n, np.int32)
            labeling[n - 1] = 0  # exactly one active site, below the stale entries
            gc.set_labeling(labeling)
            gc.alpha_expansion(1)
            np.testing.assert_array_equal(gc.get_labeling(), np.ones(n, np.int32))
            self.assertEqual(gc.compute_energy(), n)
            self.assertEqual(gc.expansion(), 0)  # and it still optimizes correctly

    def test_use_after_close(self):
        gc = GCO(4, 2)
        gc.close()
        with self.assertRaises(RuntimeError):
            gc.expansion()
        gc.close()  # idempotent

    def test_huge_terms_rejected(self):
        D = np.array([[0, bkgco.MAX_ENERGY_TERM + 1], [bkgco.MAX_ENERGY_TERM + 1, 0]], np.int32)
        with GCO(2, 2) as gc:
            gc.set_data_cost(D)
            gc.set_smooth_cost(np.array([[0, 1], [1, 0]], np.int32))
            gc.set_neighbors(np.array([[0, 1]]))
            with self.assertRaises(RuntimeError):
                gc.expansion()

    def test_non_metric_smooth_cost_detected(self):
        V = np.array([[0, 100, 1], [100, 0, 1], [1, 1, 0]], np.int32)  # V(0,1) > V(0,2)+V(2,1)
        D = np.array([[0, 50, 50], [50, 0, 50]], np.int32)
        with GCO(2, 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            gc.set_neighbors(np.array([[0, 1]]))
            gc.set_labeling([0, 1])
            with self.assertRaises(RuntimeError):
                gc.alpha_expansion(2)


class TestInterrupt(unittest.TestCase):
    def _interrupt_mid_expansion(self, trigger):
        """Fire `trigger` from a timer thread during a long expansion; the object
        must raise KeyboardInterrupt and stay usable and resumable."""
        rng = np.random.default_rng(9)
        n, k = 300, 8
        D = rng.integers(0, 100, size=(n * n, k)).astype(np.int32)
        V = (4 * (1 - np.eye(k))).astype(np.int32)
        with GCO.grid((n, n), k) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            timer = threading.Timer(0.02, trigger)
            timer.start()
            interrupted = False
            try:
                gc.expansion()
            except KeyboardInterrupt:
                interrupted = True
            finally:
                timer.cancel()
            if not interrupted:
                try:
                    time.sleep(0.05)    # absorb an interrupt that arrived a moment late
                except KeyboardInterrupt:
                    pass
                self.skipTest("expansion finished before the interrupt arrived")
            self.assertEqual(gc.get_labeling().shape, (n * n,))  # still usable
            self.assertGreater(gc.expansion(1), 0)  # and resumable

    def test_ctrl_c_during_expansion(self):
        # _thread.interrupt_main() sets the same pending-SIGINT flag that a console
        # Ctrl-C does, and works everywhere -- on Windows os.kill() cannot deliver
        # SIGINT, it calls TerminateProcess() instead.
        self._interrupt_mid_expansion(_thread.interrupt_main)

    @unittest.skipIf(sys.platform == "win32", "os.kill() cannot deliver SIGINT on Windows")
    def test_real_sigint_during_expansion(self):
        self._interrupt_mid_expansion(lambda: os.kill(os.getpid(), signal.SIGINT))


def bk_brute_force(nvars, unary, pairs, triples, const=0):
    """Exhaustive minimum of the same energy BKEnergy is given."""
    best, arg = None, None
    for bits in itertools.product((0, 1), repeat=nvars):
        e = const + sum(unary[i][bits[i]] for i in range(nvars))
        for (i, j), c in pairs:
            e += c[2 * bits[i] + bits[j]]
        for (i, j, k), c in triples:
            e += c[4 * bits[i] + 2 * bits[j] + bits[k]]
        if best is None or e < best:
            best, arg = e, np.array(bits, np.uint8)
    return best, arg


def bk_build(nvars, unary, pairs, triples, const=0, dtype=None):
    e = bkgco.BKEnergy(dtype)
    e.add_term1(np.asarray(unary, dtype) if dtype else unary)
    if const:
        e.add_constant(const)
    if pairs:
        e.add_term2([p[0] for p in pairs], [p[1] for p in pairs])
    if triples:
        e.add_term3([t[0] for t in triples], [t[1] for t in triples])
    return e


class TestBKEnergy(unittest.TestCase):
    def test_docstring_example(self):
        # E(x,y,z) = x - 2y + 3(1-z) - 4xy + 5|y-z|
        e = bkgco.BKEnergy()
        e.add_term1([[0, 1], [0, -2], [3, 0]])
        e.add_term2([[0, 1], [1, 2]], [[0, 0, 0, -4], [0, 5, 5, 0]])
        self.assertEqual(e.minimize(), -5)
        np.testing.assert_array_equal(e.get_solution(), [1, 1, 1])
        self.assertEqual(e.get_solution().dtype, np.uint8)
        self.assertEqual((e.num_vars, e.minimized), (3, True))

    def test_unary_only(self):
        e = bkgco.BKEnergy()
        e.add_term1(np.array([[0, 5], [5, 0], [2, 2]], np.int32))
        self.assertEqual(e.minimize(), 0 + 0 + 2)
        np.testing.assert_array_equal(e.get_solution()[:2], [0, 1])

    def test_pairwise_matches_brute_force(self):
        rng = np.random.default_rng(12)
        n = 10
        for _ in range(20):
            unary = rng.integers(-5, 6, size=(n, 2)).tolist()
            pairs = []
            for _ in range(12):
                i, j = sorted(rng.choice(n, 2, replace=False).tolist())
                c = rng.integers(-5, 6, size=4)
                c[3] = c[1] + c[2] - c[0] - rng.integers(0, 4)   # force regularity
                pairs.append(((i, j), c.tolist()))
            want, _ = bk_brute_force(n, unary, pairs, [])
            with bk_build(n, unary, pairs, []) as e:
                got = e.minimize()
                x = e.get_solution()
            self.assertEqual(got, want)
            check, _ = bk_brute_force(n, unary, pairs, [])
            self.assertEqual(
                sum(unary[i][x[i]] for i in range(n))
                + sum(c[2 * x[i] + x[j]] for (i, j), c in pairs), check)

    def test_triple_terms_both_branches(self):
        # -3*x*y*z takes the pi >= 0 branch, -3*(1-x)(1-y)(1-z) the pi < 0 branch
        for costs in ([0, 0, 0, 0, 0, 0, 0, -3], [-3, 0, 0, 0, 0, 0, 0, 0]):
            unary = [[0, 1], [0, 1], [0, 1]]
            triples = [((0, 1, 2), costs)]
            want, sol = bk_brute_force(3, unary, [], triples)
            with bk_build(3, unary, [], triples) as e:
                self.assertEqual(e.minimize(), want)
                np.testing.assert_array_equal(e.get_solution(), sol)

    def test_triple_matches_brute_force(self):
        rng = np.random.default_rng(13)
        n = 8

        def regular(c):
            proj = [(0, 3, 1, 2), (4, 7, 5, 6), (0, 5, 1, 4),
                    (2, 7, 3, 6), (0, 6, 2, 4), (1, 7, 3, 5)]
            return all(c[a] + c[b] <= c[p] + c[q] for a, b, p, q in proj)

        for _ in range(15):
            unary = rng.integers(-4, 5, size=(n, 2)).tolist()
            triples = []
            while len(triples) < 4:
                c = rng.integers(-4, 5, size=8).tolist()
                if regular(c):
                    triples.append((tuple(rng.choice(n, 3, replace=False).tolist()), c))
            want, _ = bk_brute_force(n, unary, [], triples)
            with bk_build(n, unary, [], triples) as e:
                self.assertEqual(e.minimize(), want)

    def test_mixed_orders_and_constant(self):
        rng = np.random.default_rng(14)
        n = 9
        unary = rng.integers(-4, 5, size=(n, 2)).tolist()
        pairs = [((0, 1), [0, 3, 3, 0]), ((2, 5), [-1, 0, 0, -1]), ((7, 8), [0, 2, 2, 0])]
        triples = [((1, 3, 4), [0, 0, 0, 0, 0, 0, 0, -6]), ((5, 6, 7), [-2, 0, 0, 0, 0, 0, 0, 0])]
        want, sol = bk_brute_force(n, unary, pairs, triples, const=7)
        with bk_build(n, unary, pairs, triples, const=7) as e:
            self.assertEqual(e.minimize(), want)
            np.testing.assert_array_equal(e.get_solution(), sol)

    def test_accumulating_unary_batches(self):
        e = bkgco.BKEnergy(np.int32)
        e.add_term1([[0, 1], [0, 1]])
        e.add_term1([[0, 1], [0, -5]])       # accumulates onto the same variables
        self.assertEqual(e.minimize(), -4)
        np.testing.assert_array_equal(e.get_solution(), [0, 1])

    def test_dtypes(self):
        self.assertEqual(bkgco.BKEnergy().add_term1([[0, 1]] * 2).dtype, np.dtype(np.int64))
        self.assertEqual(bkgco.BKEnergy().add_term1(np.zeros((2, 2), np.int32)).dtype,
                         np.dtype(np.int32))
        self.assertEqual(bkgco.BKEnergy().add_term1(np.zeros((2, 2), np.float32)).dtype,
                         np.dtype(np.float64))
        with bkgco.BKEnergy() as e:                      # float64 keeps fractions
            e.add_term1([[0.0, 0.5], [0.25, 0.0]])
            e.add_term2([[0, 1]], [[0.0, 0.75, 0.75, 0.0]])
            self.assertAlmostEqual(e.minimize(), 0.25)
            self.assertEqual(e.dtype, np.dtype(np.float64))
        with bkgco.BKEnergy() as e:                      # int64 beyond int32 range
            big = 5 * 10 ** 9
            e.add_term1(np.array([[0, -big], [0, big]], np.int64))
            self.assertEqual(e.minimize(), -big)
            np.testing.assert_array_equal(e.get_solution(), [1, 0])

    def test_minimize_is_idempotent(self):
        with bk_build(3, [[0, 1], [0, -2], [3, 0]], [((0, 1), [0, 0, 0, -4])], []) as e:
            first = e.minimize()
            self.assertEqual(e.minimize(), first)
            np.testing.assert_array_equal(e.get_solution(), e.get_solution())

    def test_errors(self):
        e = bkgco.BKEnergy()
        with self.assertRaises(RuntimeError):            # nothing sized yet
            e.add_term2([[0, 1]], [[0, 0, 0, 0]])
        with self.assertRaises(RuntimeError):
            e.minimize()
        e.add_term1([[0, 1], [0, 1], [0, 1]])
        with self.assertRaises(RuntimeError):            # not minimized yet
            e.get_solution()
        with self.assertRaises(ValueError):              # index out of range
            e.add_term2([[0, 3]], [[0, 0, 0, 0]])
        with self.assertRaises(ValueError):              # a term needs distinct variables
            e.add_term2([[1, 1]], [[0, 0, 0, 0]])
        with self.assertRaises(ValueError):              # E00 + E11 > E01 + E10
            e.add_term2([[0, 1]], [[0, 0, 0, 4]])
        with self.assertRaises(ValueError):              # +x*y*z is not regular
            e.add_term3([[0, 1, 2]], [[0, 0, 0, 0, 0, 0, 0, 3]])
        with self.assertRaises(ValueError):              # shapes
            e.add_term2([[0, 1]], [[0, 0, 0]])
        with self.assertRaises(ValueError):
            e.add_term2([[0, 1], [1, 2]], [[0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            e.add_term1([[0, 1], [0, 1]])                # wrong number of variables
        with self.assertRaises(ValueError):              # over the dtype ceiling
            e.add_term1([[0, bkgco.max_energy_term(e.dtype) + 1], [0, 0], [0, 0]])
        e.minimize()
        with self.assertRaises(RuntimeError):            # frozen after minimize
            e.add_term1([[0, 1], [0, 1], [0, 1]])
        with self.assertRaises(RuntimeError):
            e.add_term2([[0, 1]], [[0, 0, 0, 0]])
        e.close()
        with self.assertRaises(RuntimeError):
            e.minimize()

    def test_auxiliary_variables_stay_hidden(self):
        with bk_build(3, [[0, 0], [0, 0], [0, 0]],
                      [], [((0, 1, 2), [0, 0, 0, 0, 0, 0, 0, -3])]) as e:
            e.minimize()
            self.assertEqual(e.num_vars, 3)
            self.assertEqual(len(e.get_solution()), 3)   # no auxiliary variable leaks out


def bk_min_cut(tcaps, edges):
    """Exhaustive min cut (== max flow): bit 0 keeps a node on the source side."""
    n = len(tcaps)
    best, arg = None, None
    for bits in itertools.product((0, 1), repeat=n):
        cut = sum(tcaps[i][0] if bits[i] else tcaps[i][1] for i in range(n))
        for (i, j), (cap, rcap) in edges:
            if bits[i] == 0 and bits[j] == 1:
                cut += cap
            if bits[j] == 0 and bits[i] == 1:
                cut += rcap
        if best is None or cut < best:
            best, arg = cut, np.array(bits, np.uint8)
    return best, arg


def bkg_build(tcaps, edges, dtype=None, edge_capacity=0):
    g = bkgco.BKGraph(edge_capacity, dtype)
    g.add_tweights(np.asarray(tcaps, dtype) if dtype else tcaps)
    if edges:
        g.add_edges([e[0] for e in edges], [e[1] for e in edges])
    return g


class TestBKGraph(unittest.TestCase):
    def test_docstring_example(self):
        g = bkgco.BKGraph()
        g.add_tweights([[1, 5], [2, 6]])
        g.add_edges([[0, 1]], [[3, 4]])
        self.assertEqual(g.maxflow(), 3)          # the source can only push 1 + 2
        np.testing.assert_array_equal(g.get_segments(), [1, 1])
        self.assertEqual(g.get_segments().dtype, np.uint8)
        self.assertEqual((g.num_nodes, g.flowed), (2, True))

    def test_single_edge_bottleneck(self):
        # S ->10-> 0 ->2-> 1 ->10-> T : the middle edge is the cut
        with bkg_build([[10, 0], [0, 10]], [((0, 1), (2, 0))]) as g:
            self.assertEqual(g.maxflow(), 2)
            np.testing.assert_array_equal(g.get_segments(), [0, 1])

    def test_matches_brute_force_min_cut(self):
        rng = np.random.default_rng(21)
        n = 8
        for _ in range(25):
            tcaps = rng.integers(0, 10, size=(n, 2)).tolist()
            edges = []
            for _ in range(10):
                i, j = rng.choice(n, 2, replace=False).tolist()
                edges.append(((i, j), tuple(rng.integers(0, 8, size=2).tolist())))
            want, _ = bk_min_cut(tcaps, edges)
            with bkg_build(tcaps, edges) as g:
                self.assertEqual(g.maxflow(), want)
                seg = g.get_segments()
                # the reported segmentation must realize that cut
                cut = sum(tcaps[i][0] if seg[i] else tcaps[i][1] for i in range(n))
                for (i, j), (cap, rcap) in edges:
                    cut += cap if (seg[i] == 0 and seg[j] == 1) else 0
                    cut += rcap if (seg[j] == 0 and seg[i] == 1) else 0
                self.assertEqual(cut, want)

    def test_negative_terminal_capacities(self):
        tcaps = [[-3, 2], [4, -1], [0, 0]]
        edges = [((0, 1), (2, 2)), ((1, 2), (1, 0))]
        want, _ = bk_min_cut(tcaps, edges)
        with bkg_build(tcaps, edges) as g:
            self.assertEqual(g.maxflow(), want)

    def test_accumulating_tweights(self):
        with bkg_build([[1, 0], [0, 1]], [((0, 1), (5, 5))]) as g:
            g.add_tweights([[2, 0], [0, 2]])       # now S->0 = 3, 1->T = 3
            self.assertEqual(g.maxflow(), 3)

    def test_incremental_maxflow(self):
        """Capacities may be added after a maxflow; the next call returns the total."""
        rng = np.random.default_rng(22)
        n = 6
        for _ in range(10):
            t1 = rng.integers(0, 8, size=(n, 2)).tolist()
            t2 = rng.integers(0, 8, size=(n, 2)).tolist()
            pick = lambda: tuple(rng.choice(n, 2, replace=False).tolist())
            e1 = [(pick(), tuple(rng.integers(0, 6, size=2).tolist())) for _ in range(4)]
            e2 = [(pick(), tuple(rng.integers(0, 6, size=2).tolist())) for _ in range(4)]
            both = [[a[0] + b[0], a[1] + b[1]] for a, b in zip(t1, t2)]
            want, _ = bk_min_cut(both, e1 + e2)
            with bkg_build(t1, e1) as g:
                g.maxflow()
                g.add_tweights(t2)
                g.add_edges([e[0] for e in e2], [e[1] for e in e2])
                self.assertEqual(g.maxflow(), want)

    def test_default_segment_for_free_nodes(self):
        # node 1 is isolated with no capacity: it may sit on either side
        with bkg_build([[4, 0], [0, 0]], []) as g:
            g.maxflow()
            np.testing.assert_array_equal(g.get_segments(), [0, 0])
            np.testing.assert_array_equal(g.get_segments(default=1), [0, 1])

    def test_edge_capacity_hint_is_only_a_hint(self):
        edges = [((0, 1), (2, 2)), ((1, 2), (3, 1)), ((0, 2), (1, 1))]
        tcaps = [[5, 0], [0, 0], [0, 5]]
        want, _ = bk_min_cut(tcaps, edges)
        for hint in (0, 1, 1000):
            with bkg_build(tcaps, edges, edge_capacity=hint) as g:
                self.assertEqual(g.maxflow(), want)
        self.assertEqual(bkgco.BKGraph(64).edge_capacity, 64)
        with self.assertRaises(ValueError):
            bkgco.BKGraph(-1)

    def test_dtypes(self):
        self.assertEqual(bkgco.BKGraph().add_tweights([[1, 2]] * 2).dtype, np.dtype(np.int64))
        self.assertEqual(bkgco.BKGraph().add_tweights(np.zeros((2, 2), np.int32)).dtype,
                         np.dtype(np.int32))
        with bkgco.BKGraph(dtype=np.float64) as g:
            g.add_tweights([[1.5, 0.0], [0.0, 2.5]])
            g.add_edges([[0, 1]], [[0.75, 0.75]])
            self.assertAlmostEqual(g.maxflow(), 0.75)
        with bkgco.BKGraph() as g:                      # int64 beyond int32 range
            big = 5 * 10 ** 9
            g.add_tweights(np.array([[big, 0], [0, big]], np.int64))
            g.add_edges([[0, 1]], np.array([[big, big]], np.int64))
            self.assertEqual(g.maxflow(), big)

    def test_no_advanced_interface(self):
        g = bkgco.BKGraph().add_tweights([[1, 1], [1, 1]])
        for name in ("add_node", "reset", "mark_node", "get_trcap", "set_trcap",
                     "get_rcap", "set_rcap", "get_first_arc", "get_arc_ends"):
            self.assertFalse(hasattr(g, name), name)
            self.assertFalse(hasattr(g._h, name), name)

    def test_errors(self):
        g = bkgco.BKGraph()
        with self.assertRaises(RuntimeError):            # no nodes yet
            g.add_edges([[0, 1]], [[1, 1]])
        with self.assertRaises(RuntimeError):
            g.maxflow()
        g.add_tweights([[1, 1], [1, 1], [1, 1]])
        with self.assertRaises(RuntimeError):            # no flow computed yet
            g.get_segments()
        with self.assertRaises(ValueError):              # node index out of range
            g.add_edges([[0, 3]], [[1, 1]])
        with self.assertRaises(ValueError):              # self loop
            g.add_edges([[2, 2]], [[1, 1]])
        with self.assertRaises(ValueError):              # negative edge capacity
            g.add_edges([[0, 1]], [[-1, 1]])
        with self.assertRaises(ValueError):              # shapes
            g.add_edges([[0, 1]], [[1, 1, 1]])
        with self.assertRaises(ValueError):
            g.add_edges([[0, 1], [1, 2]], [[1, 1]])
        with self.assertRaises(ValueError):              # wrong number of nodes
            g.add_tweights([[1, 1], [1, 1]])
        with self.assertRaises(ValueError):              # over the dtype ceiling
            g.add_tweights([[bkgco.max_energy_term(g.dtype) + 1, 0], [0, 0], [0, 0]])
        g.maxflow()
        with self.assertRaises(ValueError):
            g.get_segments(default=2)
        g.close()
        with self.assertRaises(RuntimeError):
            g.maxflow()


class TestCppInterface(unittest.TestCase):
    """tests/cpp_all_types.cpp: all three instantiations plus the historical C++ names."""

    def test_all_instantiations_binary(self):
        import shutil
        import subprocess
        import sysconfig
        import tempfile

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = os.path.join(root, "tests", "cpp_all_types.cpp")
        candidates = [os.environ.get("CXX"), "c++", "g++", "clang++"]
        errors = []
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "cpp_all_types")
            for cxx in candidates:
                if not cxx or not shutil.which(cxx.split()[0]):
                    continue
                cmd = cxx.split() + ["-std=c++11", "-DNDEBUG", "-O1", "-w",
                                     "-I" + os.path.join(root, "cpp", "gco"), "-o", exe, src]
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode == 0:
                    out = subprocess.run([exe], capture_output=True, text=True)
                    self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
                    self.assertIn("PASSED", out.stdout)
                    return
                errors.append("%s:\n%s" % (cxx, p.stderr[-2000:]))
        self.skipTest("no usable C++ compiler:\n" + "\n".join(errors))


class TestThreading(unittest.TestCase):
    def test_parallel_instances(self):
        rng = np.random.default_rng(7)
        D = rng.integers(0, 20, size=(400, 4)).astype(np.int32)
        V = (2 * (1 - np.eye(4))).astype(np.int32)
        out = {}

        def run(k):
            with GCO.grid((20, 20), 4) as gc:
                gc.set_data_cost(D).set_smooth_cost(V)
                out[k] = gc.expansion()

        threads = [threading.Thread(target=run, args=(k,)) for k in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(set(out.values())), 1)
        self.assertEqual(len(out), 4)

    def test_shared_instance_is_serialized(self):
        D = np.random.default_rng(8).integers(0, 20, size=(400, 3)).astype(np.int32)
        V = (2 * (1 - np.eye(3))).astype(np.int32)
        with GCO.grid((20, 20), 3) as gc:
            gc.set_data_cost(D).set_smooth_cost(V)
            energies = []

            def run():
                energies.append(gc.expansion())

            threads = [threading.Thread(target=run) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.assertEqual(len(set(energies)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
