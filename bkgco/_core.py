import warnings

import numpy as np

from . import _bkgco

INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
FLOAT64 = np.dtype(np.float64)
DTYPES = (INT32, INT64, FLOAT64)          # index == the C dtype code
MAX_ENERGY_TERM = _bkgco.max_energy_terms[0]  # int32 ceiling; see max_energy_term()
_COPY_WARN = 1 << 20
_IDX = INT32


def _check_dtype(dtype):
    d = np.dtype(dtype)
    if d not in DTYPES:
        raise TypeError("dtype must be int32, int64 or float64")
    return d


def _infer_dtype(a):
    """float64 for floating-point input, int64 for 8-byte integers, else int32."""
    d = np.asarray(a).dtype
    if d.kind in "fc":
        return FLOAT64
    return INT64 if d.itemsize > 4 else INT32


def max_energy_term(dtype):
    """Largest energy term the library accepts for `dtype`."""
    return _bkgco.max_energy_terms[DTYPES.index(_check_dtype(dtype))]

__all__ = ["GCO", "BKEnergy", "BKGraph", "DTYPES", "MAX_ENERGY_TERM", "max_energy_term"]


def _conv(a, dtype, what):
    src = np.asarray(a)
    out = np.ascontiguousarray(src, dtype=dtype)
    if out is not src and dtype.kind == "i" and not np.array_equal(out, src):
        raise ValueError("%s: values are not exactly representable as %s "
                         "(use a wider dtype=)" % (what, dtype))
    if out is not src and src.size >= _COPY_WARN:
        warnings.warn(
            "%s: converting %s array of %d elements to %s (copy)" % (what, src.dtype, src.size, dtype),
            stacklevel=3,
        )
    return out


class GCO:
    """Multi-label energy minimization by graph cuts (alpha-expansion, alpha-beta-swap).

    E(l) = sum_p D_p(l_p) + sum_pq w_pq V(l_p,l_q) + sum_L' h_L'(l)

    Sites and labels use 0-based indices. Cost arrays are held by reference (not
    copied) when they are already C-contiguous and of the object's dtype.

    The energy-term type is int32, int64 or float64. With dtype=None (the default) it
    is chosen from the first cost array passed in -- floating-point input selects
    float64, 8-byte integers int64, smaller integers int32 -- and is fixed from then on.
    """

    def __init__(self, num_sites, num_labels, dtype=None, shape=None):
        self._ns = int(num_sites)
        self._nl = int(num_labels)
        self._w, self._ht = (0, 0) if shape is None else (int(shape[1]), int(shape[0]))
        self.shape = None if shape is None else (self._ht, self._w)
        self.dtype = None if dtype is None else _check_dtype(dtype)
        if self._ns < 1 or self._nl < 2:
            raise ValueError("num_sites must be >= 1 and num_labels >= 2")
        if shape is not None and (self._w < 2 or self._ht < 2 or self._w * self._ht != self._ns):
            raise ValueError("grid requires width >= 2, height >= 2 and num_sites == width*height")
        self._handle = None
        self._closed = False
        self._refs = {}
        self._verbosity = 0
        if self.dtype is not None:
            self._create(self.dtype)

    @classmethod
    def grid(cls, shape, num_labels, dtype=None):
        """4-connected grid graph over an image of `shape` == (height, width)."""
        h, w = int(shape[0]), int(shape[1])
        return cls(h * w, num_labels, dtype, shape=(h, w))

    def _create(self, dtype):
        self.dtype = dtype
        self._handle = _bkgco.GCO(self._ns, self._nl, DTYPES.index(dtype), self._w, self._ht)

    @property
    def _h(self):
        """The C handle; creating it fixes the dtype (int32 if nothing decided it)."""
        if self._closed:
            raise RuntimeError("GCO object has been destroyed")
        if self._handle is None:
            self._create(INT32)
        return self._handle

    def _cost(self, a, what):
        """Convert a cost array, choosing the backend dtype from it if still undecided."""
        if self._handle is None and not self._closed:
            self._create(_infer_dtype(a))
        return _conv(a, self.dtype, what)

    # ---- properties -------------------------------------------------------
    @property
    def num_sites(self):
        return self._ns

    @property
    def num_labels(self):
        return self._nl

    @property
    def is_grid(self):
        return self.shape is not None

    # ---- costs ------------------------------------------------------------
    def set_data_cost(self, cost):
        """Dense data costs, shape (num_sites, num_labels) or (h, w, num_labels)."""
        d = self._cost(cost, "data cost")
        if d.shape[-1] != self.num_labels or d.size != self.num_sites * self.num_labels:
            raise ValueError(
                "data cost must have shape (%d, %d)" % (self.num_sites, self.num_labels)
            )
        d = d.reshape(self.num_sites, self.num_labels)
        self._h.set_data_cost(d)
        self._refs["dc"] = d
        return self

    def set_data_cost_sparse(self, label, sites, cost):
        """Data costs of `label` for a subset of sites; other sites become infeasible.

        `sites` must be sorted in increasing order. Costs are copied internally.
        """
        s = _conv(sites, _IDX, "sparse sites").ravel()
        c = self._cost(cost, "sparse costs").ravel()
        if s.size != c.size:
            raise ValueError("sites and cost must have the same length")
        self._h.set_data_cost_sparse(int(label), s, c)
        return self

    def set_smooth_cost(self, cost):
        """Label compatibility V, shape (num_labels, num_labels)."""
        v = self._cost(cost, "smooth cost")
        if v.shape != (self.num_labels, self.num_labels):
            raise ValueError("smooth cost must have shape (%d, %d)" % (self.num_labels, self.num_labels))
        self._h.set_smooth_cost(v)
        self._refs["sc"] = v
        return self

    def set_smooth_cost_vh(self, cost, v_weights, h_weights):
        """Grid only: V plus per-site vertical/horizontal edge weights."""
        v = self._cost(cost, "smooth cost")
        if v.shape != (self.num_labels, self.num_labels):
            raise ValueError("smooth cost must have shape (%d, %d)" % (self.num_labels, self.num_labels))
        vc = self._cost(v_weights, "vertical weights").ravel()
        hc = self._cost(h_weights, "horizontal weights").ravel()
        self._h.set_smooth_cost_vh(v, vc, hc)
        self._refs["sc"] = v
        return self

    def set_neighbors(self, edges, weights=None):
        """General graph only: declare each unordered neighbor pair exactly once.

        `edges` may be an (E, 2) array, a tuple (sites1, sites2) of arrays, or a
        (num_sites, num_sites) scipy sparse matrix whose upper triangle holds w_pq.
        """
        if hasattr(edges, "tocoo"):
            coo = edges.tocoo()
            keep = coo.row < coo.col
            i, j = coo.row[keep], coo.col[keep]
            if weights is None:
                weights = coo.data[keep]
        elif isinstance(edges, tuple):
            i, j = edges
        else:
            e = np.asarray(edges)
            if e.ndim != 2 or e.shape[1] != 2:
                raise ValueError("edges must have shape (num_edges, 2)")
            i, j = e[:, 0], e[:, 1]
        i = _conv(i, _IDX, "edge sites").ravel()
        j = _conv(j, _IDX, "edge sites").ravel()
        if i.size != j.size:
            raise ValueError("edge site arrays must have the same length")
        w = None if weights is None else self._cost(weights, "edge weights").ravel()
        self._h.set_neighbors(i, j, w)
        return self

    def set_label_cost(self, cost, labels=None):
        """Cost charged once if a label (or any label of `labels`) is used at all."""
        if labels is not None:
            l = _conv(labels, _IDX, "label subset").ravel()
            self._h.set_label_subset_cost(l, float(cost))
        elif np.ndim(cost) == 0:
            self._h.set_label_cost(float(cost))
        else:
            c = self._cost(cost, "label cost").ravel()
            self._h.set_label_cost_array(c)
        return self

    # ---- labeling ---------------------------------------------------------

    def get_labeling(self, start=0, count=None):
        """The current labeling as int32, one entry per site (or a slice of it)."""
        n = self.num_sites - start if count is None else count
        out = np.empty(max(n, 0), dtype=_IDX)
        self._h.get_labeling(out, start)
        return out

    def set_labeling(self, labeling):
        """Set the current labeling; one int32 label per site, in range 0..num_labels-1."""
        self._h.set_labeling(_conv(labeling, _IDX, "labeling").ravel())
        return self

    def set_label_order(self, order=None, random=False):
        """Visit labels in `order` (subsets allowed), or randomly if random=True."""
        if order is not None:
            self._h.set_label_order(_conv(order, _IDX, "label order").ravel())
        else:
            self._h.set_label_order_random(bool(random))
        return self

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level):
        self._h.set_verbosity(int(level))
        self._verbosity = int(level)

    # ---- optimization -----------------------------------------------------
    def expansion(self, max_cycles=-1):
        return self._h.expansion(int(max_cycles))

    def alpha_expansion(self, alpha):
        return self._h.alpha_expansion(int(alpha))

    def swap(self, max_cycles=-1):
        return self._h.swap(int(max_cycles))

    def alpha_beta_swap(self, alpha, beta):
        self._h.alpha_beta_swap(int(alpha), int(beta))
        return self

    def compute_energy(self):
        return self._h.compute_energy()

    def data_energy(self):
        return self._h.data_energy()

    def smooth_energy(self):
        return self._h.smooth_energy()

    def label_energy(self):
        return self._h.label_energy()

    # ---- lifetime ---------------------------------------------------------
    def close(self):
        """Free the C++ object and drop the references to the cost arrays.

        Optional: this happens on its own once the last reference to this object
        goes away. Call it (or use `with`) only when you want the cost arrays
        released -- and their buffers un-pinned, so numpy can resize or free them --
        at a chosen moment rather than whenever the object is collected.
        """
        if self._handle is not None:
            self._handle.destroy()
        self._closed = True
        self._refs.clear()

    destroy = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        kind = "grid%s" % (self.shape,) if self.shape else "general"
        return "<GCO %s num_sites=%d num_labels=%d dtype=%s>" % (
            kind,
            self.num_sites,
            self.num_labels,
            self.dtype if self.dtype is not None else "undecided",
        )


class BKEnergy:
    """Global minimum of a regular pseudo-boolean energy of order at most three
    (Kolmogorov & Zabih, "What Energy Functions can be Minimized via Graph Cuts?").

        E(x) = sum_i E_i(x_i) + sum_ij E_ij(x_i,x_j) + sum_ijk E_ijk(x_i,x_j,x_k)

    Terms are added in batches -- one row per term -- to keep the per-term Python
    overhead out of the way. The first `add_term1` call fixes both the number of
    binary variables (its rows) and the energy-term dtype (int32, int64 or float64,
    inferred from the array unless `dtype` was given).

        e = BKEnergy()
        e.add_term1([[0, 1], [0, -2], [3, 0]])            # E(x), E(y), E(z)
        e.add_term2([[0, 1], [1, 2]],                     # (x,y) and (y,z)
                    [[0, 0, 0, -4], [0, 5, 5, 0]])        # E00 E01 E10 E11
        energy = e.minimize()
        x = e.get_solution()                              # uint8, one entry per variable

    Every term must be regular; `add_term3` also creates auxiliary variables
    internally, which are not visible here.
    """

    def __init__(self, dtype=None):
        self.dtype = None if dtype is None else _check_dtype(dtype)
        self._h = None
        self._nvars = 0
        self._closed = False

    @property
    def num_vars(self):
        return self._nvars

    @property
    def minimized(self):
        return bool(self._h.solved) if self._h is not None else False

    def _rows(self, a, cols, what):
        arr = np.asarray(a)
        if arr.ndim == 1 and arr.size == cols:
            arr = arr.reshape(1, cols)
        if arr.ndim != 2 or arr.shape[1] != cols:
            raise ValueError("%s: expected shape (num_terms, %d)" % (what, cols))
        return arr

    def _started(self):
        if self._closed:
            raise RuntimeError("BKEnergy object has been destroyed")
        if self._h is None:
            raise RuntimeError("call add_term1() first; it fixes the number of variables")
        return self._h

    def add_term1(self, cost):
        """Unary terms, shape (num_vars, 2): columns are E(0), E(1)."""
        if self._closed:
            raise RuntimeError("BKEnergy object has been destroyed")
        c = self._rows(cost, 2, "unary terms")
        if self._h is None:
            if self.dtype is None:
                self.dtype = _infer_dtype(c)
            c = _conv(c, self.dtype, "unary terms")
            self._nvars = c.shape[0]
            self._h = _bkgco.BKEnergy(self._nvars, DTYPES.index(self.dtype))
        else:
            c = _conv(c, self.dtype, "unary terms")
            if c.shape[0] != self._nvars:
                raise ValueError("unary terms: expected shape (%d, 2)" % self._nvars)
        self._h.add_term1(c)
        return self

    def _add_termn(self, variables, cost, arity):
        h = self._started()
        what = "term%d" % arity
        v = self._rows(variables, arity, what + " variables")
        c = self._rows(cost, 1 << arity, what + " costs")
        if v.shape[0] != c.shape[0]:
            raise ValueError("%s: got %d variable rows but %d cost rows"
                             % (what, v.shape[0], c.shape[0]))
        if v.shape[0]:
            add = h.add_term2 if arity == 2 else h.add_term3
            add(_conv(v, _IDX, what + " variables"), _conv(c, self.dtype, what + " costs"))
        return self

    def add_term2(self, variables, cost):
        """Pairwise terms: `variables` (num_terms, 2), `cost` (num_terms, 4) holding
        E00, E01, E10, E11. Each term must satisfy E00 + E11 <= E01 + E10."""
        return self._add_termn(variables, cost, 2)

    def add_term3(self, variables, cost):
        """Triple terms: `variables` (num_terms, 3), `cost` (num_terms, 8) holding
        E000, E001, E010, E011, E100, E101, E110, E111 (last index varies fastest).
        Every projection onto two variables must be regular."""
        return self._add_termn(variables, cost, 3)

    def add_constant(self, value):
        """Add a constant to the energy."""
        self._started().add_constant(float(value))
        return self

    def minimize(self):
        """Minimize and return the minimum energy. Terms cannot be added afterwards."""
        return self._started().minimize()

    def get_solution(self):
        """The optimal assignment as uint8, one entry per variable."""
        out = np.empty(self._nvars, dtype=np.uint8)
        self._started().get_solution(out)
        return out

    def close(self):
        """Free the underlying graph.

        Optional: it is freed anyway once the last reference to this object goes
        away. Call it (or use `with`) only to release that memory -- which can be
        large -- at a chosen moment. Term arrays are copied in, so nothing of yours
        is held.
        """
        if self._h is not None:
            self._h.destroy()
        self._closed = True

    destroy = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "<BKEnergy num_vars=%d dtype=%s%s>" % (
            self._nvars,
            self.dtype if self.dtype is not None else "undecided",
            " minimized" if self.minimized else "",
        )


class BKGraph:
    """Boykov-Kolmogorov max-flow / min-cut on a directed graph with terminals
    (graph.h), driven in batches.

        g = BKGraph()
        g.add_tweights([[1, 5],            # node 0: source and sink capacities
                        [2, 6]])           # node 1
        g.add_edges([[0, 1]], [[3, 4]])    # cap 0->1 and 1->0

        flow = g.maxflow()                 # 3: the source can push 1 + 2
        segs = g.get_segments()            # uint8: 0 = SOURCE side, 1 = SINK side

    The first `add_tweights` call fixes the number of nodes (its rows) and the
    capacity dtype (int32, int64 or float64, inferred unless `dtype` was given);
    there is no way to add nodes later. `edge_capacity` preallocates room for that
    many edges -- purely a performance hint, the graph grows as needed.

    Only the library's basic interface is exposed: the advanced one (tree reuse,
    residual capacities, arc iteration, reset) is not.
    """

    def __init__(self, edge_capacity=0, dtype=None):
        self.edge_capacity = int(edge_capacity)
        if self.edge_capacity < 0:
            raise ValueError("edge_capacity must be >= 0")
        self.dtype = None if dtype is None else _check_dtype(dtype)
        self._h = None
        self._nnodes = 0
        self._closed = False

    @property
    def num_nodes(self):
        return self._nnodes

    @property
    def flowed(self):
        return bool(self._h.flowed) if self._h is not None else False

    def _rows(self, a, cols, what):
        arr = np.asarray(a)
        if arr.ndim == 1 and arr.size == cols:
            arr = arr.reshape(1, cols)
        if arr.ndim != 2 or arr.shape[1] != cols:
            raise ValueError("%s: expected shape (num_rows, %d)" % (what, cols))
        return arr

    def _started(self):
        if self._closed:
            raise RuntimeError("BKGraph object has been destroyed")
        if self._h is None:
            raise RuntimeError("call add_tweights() first; it fixes the number of nodes")
        return self._h

    def add_tweights(self, capacities):
        """Terminal capacities, shape (num_nodes, 2): columns are SOURCE->i and i->SINK.
        May be negative, and may be called repeatedly (capacities accumulate)."""
        if self._closed:
            raise RuntimeError("BKGraph object has been destroyed")
        c = self._rows(capacities, 2, "terminal capacities")
        if self._h is None:
            if self.dtype is None:
                self.dtype = _infer_dtype(c)
            c = _conv(c, self.dtype, "terminal capacities")
            self._nnodes = c.shape[0]
            self._h = _bkgco.BKGraph(self._nnodes, DTYPES.index(self.dtype), self.edge_capacity)
        else:
            c = _conv(c, self.dtype, "terminal capacities")
            if c.shape[0] != self._nnodes:
                raise ValueError("terminal capacities: expected shape (%d, 2)" % self._nnodes)
        self._h.add_tweights(c)
        return self

    def add_edges(self, nodes, capacities):
        """Bidirectional edges: `nodes` (m, 2) int32 pairs, `capacities` (m, 2) holding
        cap(i->j) and cap(j->i). Both capacities must be non-negative."""
        h = self._started()
        v = self._rows(nodes, 2, "edge nodes")
        c = self._rows(capacities, 2, "edge capacities")
        if v.shape[0] != c.shape[0]:
            raise ValueError("edges: got %d node rows but %d capacity rows"
                             % (v.shape[0], c.shape[0]))
        if v.shape[0]:
            h.add_edges(_conv(v, _IDX, "edge nodes"), _conv(c, self.dtype, "edge capacities"))
        return self

    def maxflow(self):
        """Compute the maximum flow and return it. May be called again after adding
        more capacities; the returned flow is always the total for the whole graph."""
        return self._started().maxflow()

    def get_segments(self, default=0):
        """Min-cut side of every node as uint8: 0 = SOURCE, 1 = SINK. Nodes that could
        be on either side get `default`."""
        out = np.empty(self._nnodes, dtype=np.uint8)
        self._started().get_segments(out, int(default))
        return out

    def close(self):
        """Free the underlying graph.

        Optional: it is freed anyway once the last reference to this object goes
        away. Call it (or use `with`) only to release that memory -- which can be
        large -- at a chosen moment. Capacity arrays are copied in, so nothing of
        yours is held.
        """
        if self._h is not None:
            self._h.destroy()
        self._closed = True

    destroy = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "<BKGraph num_nodes=%d dtype=%s%s>" % (
            self._nnodes,
            self.dtype if self.dtype is not None else "undecided",
            " flowed" if self.flowed else "",
        )
