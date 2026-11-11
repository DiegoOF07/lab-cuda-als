import numpy as np


def get_xp(device):
    if device == "gpu":
        import cupy as cp
        return cp
    return np


def solve_side(xp, R, B, F, lam, block):
    """Solve one ALS half-step. R: ratings CSR (n x m), B: R with ones, F: fixed factors (m x k)."""
    n, k = R.shape[0], F.shape[1]
    # Row j holds vec(f_j f_j^T), so B @ W sums the outer products of each row's items.
    W = (F[:, :, None] * F[:, None, :]).reshape(F.shape[0], k * k)
    counts = xp.maximum(xp.diff(R.indptr), 1).astype(xp.float32)
    eye = xp.eye(k, dtype=xp.float32)
    out = xp.empty((n, k), dtype=xp.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        A = (B[s:e] @ W).reshape(e - s, k, k)
        A += (lam * counts[s:e])[:, None, None] * eye
        b = R[s:e] @ F
        out[s:e] = xp.linalg.solve(A, b[:, :, None])[:, :, 0]
    return out


def als(R, Rt, k=32, iters=10, lam=0.05, device="cpu", seed=42, block=20000):
    xp = get_xp(device)
    B, Bt = R.copy(), Rt.copy()
    B.data[:] = 1
    Bt.data[:] = 1
    V0 = np.random.default_rng(seed).normal(0, 0.1, (Rt.shape[0], k)).astype(np.float32)
    V = xp.asarray(V0)
    for _ in range(iters):
        U = solve_side(xp, R, B, V, lam, block)
        V = solve_side(xp, Rt, Bt, U, lam, block)
    return U, V


def rmse(xp, U, V, u, i, r):
    pred = (U[u] * V[i]).sum(axis=1)
    return float(xp.sqrt(xp.mean((pred - r) ** 2)))
