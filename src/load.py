import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as ssp


@dataclass
class Data:
    R: object
    Rt: object
    u_test: object
    i_test: object
    r_test: object
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_ratings(self):
        return int(self.R.nnz + len(self.r_test))


def read_ratings(data_dir):
    t0 = time.perf_counter()
    df = pd.read_csv(
        f"{data_dir}/ratings.csv",
        usecols=[0, 1, 2],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    return df, time.perf_counter() - t0


def prepare(df, frac, test_size=0.1, seed=42):
    """Subsample, reindex, split train/test and build CSR matrices."""
    rng = np.random.default_rng(seed)
    if frac < 1.0:
        # Prefix of one permutation, so smaller fractions are subsets of larger ones.
        keep = np.sort(rng.permutation(len(df))[: int(len(df) * frac)])
        df = df.iloc[keep]

    u, user_ids = pd.factorize(df["userId"])
    i, item_ids = pd.factorize(df["movieId"])
    u, i = u.astype(np.int32), i.astype(np.int32)
    r = df["rating"].to_numpy(np.float32)

    test = rng.random(len(r)) < test_size
    train = ~test
    shape = (len(user_ids), len(item_ids))
    R = ssp.csr_matrix((r[train], (u[train], i[train])), shape=shape, dtype=np.float32)
    Rt = R.T.tocsr()

    # Drop test pairs whose user or item never appears in train (cold start).
    seen = (np.diff(R.indptr) > 0)[u[test]] & (np.diff(Rt.indptr) > 0)[i[test]]
    return Data(R, Rt, u[test][seen], i[test][seen], r[test][seen],
                np.asarray(user_ids), np.asarray(item_ids))


def to_gpu(d):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    return Data(csp.csr_matrix(d.R), csp.csr_matrix(d.Rt),
                cp.asarray(d.u_test), cp.asarray(d.i_test), cp.asarray(d.r_test),
                d.user_ids, d.item_ids)
