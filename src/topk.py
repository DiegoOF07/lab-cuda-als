import os

import numpy as np

K = 10
THREADS = 128  # must match kernels.cu
_kernel = None


def top10_xp(xp, U, V, block=128):
    """Library top-10 (argpartition), sorted by score. Returns a NumPy array."""
    n = U.shape[0]
    out = xp.empty((n, K), dtype=xp.int32)
    Vt = xp.ascontiguousarray(V.T)
    for s in range(0, n, block):
        S = U[s:s + block] @ Vt
        idx = xp.argpartition(S, -K, axis=1)[:, -K:]
        order = xp.argsort(-xp.take_along_axis(S, idx, axis=1), axis=1)
        out[s:s + block] = xp.take_along_axis(idx, order, axis=1)
    return out if xp is np else xp.asnumpy(out)


def _get_kernel():
    global _kernel
    if _kernel is None:
        import cupy as cp
        with open(os.path.join(os.path.dirname(__file__), "kernels.cu")) as f:
            _kernel = cp.RawModule(code=f.read()).get_function("topk_rows")
    return _kernel


def top10_kernel(U, V, block=4096):
    """GPU top-10: cuBLAS computes scores per block, the custom kernel selects. Returns a NumPy array."""
    import cupy as cp

    n, m = U.shape[0], V.shape[0]
    out = cp.empty((n, K), dtype=cp.int32)
    kernel = _get_kernel()
    for s in range(0, n, block):
        e = min(s + block, n)
        S = U[s:e] @ V.T
        kernel((e - s,), (THREADS,), (S, np.int32(m), out[s:e]))
    return cp.asnumpy(out)
