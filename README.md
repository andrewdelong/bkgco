# bkgco — Python wrapper for the BK and GCO optimization libraries

The `bkgco` Python package wraps two related C++ libraries:
* `maxflow`: the Boykov-Kolmogorov (BK) algorithm for graph-cuts, and
* `gco`: graph-cut optimization (GCO) algorithms for minimizing multi-label energies.


Three types are provided by `bkgco`:
* `BKGraph` constructs a graph from arc weights and maximizes an $s$-$t$ maximum flow.
* `BKEnergy` constructs a graph from binary energy terms and minimizes the energy.
* `GCO` iteratively minimizes a multi-label energy, using a graph-cut at each iteration.

Cost terms, neighbour indices, and weights are all passed as NumPy arrays for efficiency.

### BKGraph example

```python
from bkgco import BKGraph

with BKGraph() as g:

    # Arc weights (from source, to sink) for each node
    g.add_tweights([[5, 0],
                    [2, 0],
                    [0, 9]])

    # Arc weights between nodes
    g.add_edges([[0, 1], [1, 2]],  # Node index pairs (i,j)
                [[1, 0], [4, 0]])  # Weights for each (i->j) and (j->i)

    # Maximum flow is 3 with only node 0 disconnected from the sink.
    flow = g.maxflow()       # 3
    segs = g.get_segments()  # array([0, 1, 1], dtype=uint8)
```
The above code builds the $s$-$t$ flow graph below and maximizes the flow, cutting off node 0 (leftmost) from the sink (t).
```
      ┌────────(s)
      │         │
     5│        2│
      │         │
      ▼         ▼   
     (0)──────▶(1)─────▶(2)
           1        4    │
                         │ 9
                         │
               (t)◀──────┘
```

### BKEnergy example

The code below defines a binary energy corresponding exactly to the above s-t flow graph.

```python
from bkgco import BKEnergy

with BKEnergy() as e:

	# Add unary terms for E(x,y,z) = 5x + 2y + 9(1-z) + (1-x)y + 4(1-y)z
	e.add_term1([[0, 5],           # 5x
	             [0, 2],           # 2y
	             [9, 0]])          # 9(1-z)

	# Add pairwise terms
	e.add_term2([[0, 1], [1, 2]],  # Indices of (x,y) and (y,z) respectively.
	            [[0, 1, 0, 0],     # (1-x)y
	             [0, 4, 0, 0]])    # 4(1-y)z

	# Minimum is 3 with x=0, y=1, z=1
	e_min = e.minimize()      # 3
	e_sol = e.get_solution()  # array([0, 1, 1], dtype=uint8)
```


### GCO example

```python
from bkgco import GCO

D = [[0, 3, 5],  # Data cost (4 sites x 3 labels)
     [9, 9, 0],
     [2, 0, 3],
     [0, 3, 5]]

V = [[0, 1, 2],  # Smooth cost (3 labels x 3 labels)
     [1, 0, 1],
     [2, 1, 0]]

N = [[0, 1],     # Neighbours (i,j) at which to apply cost V
     [1, 2],
     [2, 3]]

W = [3, 1, 3]    # Weight of cost V for each neighbour in N

with GCO(num_sites=4, num_labels=3) as g:
    # Set costs and neighbours
    g.set_data_cost(D)
    g.set_smooth_cost(V)
    g.set_neighbors(N, W)

    # Minimum is 9 with labeling (2,2,0,0)
    e = g.expansion()         # 9
    l = g.get_labeling()      # array([2, 2, 0, 0], dtype=int32)
```


## Dependencies

* Python >= 3.9
* Numpy >= 1.20

## Building

Build in-place for testing:
```
python setup.py build_ext --inplace     # builds bkgco/_bkgco*.so
python -m unittest discover -s tests
```

Build and install locally, so that `import bkgco` works from anywhere:
```
pip install -e .
```

**Note for macOS.** If a non-Apple `clang++` is first on `PATH`, e.g. Homebrew LLVM, then prefix build
commands with `CC=/usr/bin/clang CXX=/usr/bin/clang++` or first run `export SDKROOT=$(xcrun --show-sdk-path)`.

## Notes

Features:
- **Label costs.** Label costs and label subset costs are supported on `GCO`.
- **Data types.** Arc weights and energy term costs can be NumPy arrays of `dtype=int32/int64/float64`.
- **No dynamic cuts.** The `reuse_trees` feature of `maxflow` is not yet exposed by this wrapper.
- **Ctrl-C.** Calls to `expansion/swap` are safely keyboard-interruptible, for interactive development in notebooks.

Usage:
- **Specifying a dtype.** With `dtype=None` (the default) the dtype is determined by the first cost array passed in.
- **Recommended dtype.** Map costs to the smallest `int32` range that suffices for your application.
  With float64, expansion/swap can rarely report a small energy *increase* from
  rounding inside max-flow, and Inf/NaN terms are undefined behavior. 

Technical:
- **Array sharing.** GCO's data and smooth cost arrays are passed to C++ by reference (no copy).
  Changes to the `D` and `V` arrays are seen immediately, but call `set_data_cost(D)` and `set_smooth_cost(V)` again (cheap) to invalidate
  cached energy values.
- **Threads.** The GIL is released so that independent `GCO` objects run in parallel;
  calls on one object are serialized by a per-object mutex. The module declares `Py_MOD_GIL_NOT_USED`,
  so it works on free-threaded CPython without re-enabling the GIL.
- **Errors.** Invalid arguments raise `ValueError`/`TypeError`; the C++ library's own checks
  (overflow, non-metric smooth costs, unsorted sparse costs, ...) raise `RuntimeError`.


## Citation / license

Usage of this wrapper is governed by the terms in `cpp/gco/GCO_README.TXT`.

## Use of AI

This wrapper was almost entirely implemented by Claude Code, under the prompting of Andrew Delong.
