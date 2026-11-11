import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

from als import als, rmse
from load import prepare, read_ratings, to_gpu
from topk import top10_kernel, top10_xp

FIELDS = ["frac", "n_ratings", "n_users", "n_items", "etapa", "dispositivo",
          "reps", "tiempo_s", "tiempos", "rmse"]


def gpu_sync():
    import cupy as cp
    cp.cuda.Device().synchronize()


def gpu_free():
    import cupy as cp
    gpu_sync()
    cp.get_default_memory_pool().free_all_blocks()


def measure(fn, reps, sync=None):
    """Run fn reps times; return the last output and the list of times."""
    times, out = [], None
    for _ in range(reps):
        out = None
        if sync:
            gpu_free()
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return out, times


def env_info(args):
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "cpu_logical_cores": os.cpu_count(), "platform": platform.platform(),
            "args": vars(args)}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    if "gpu" in args.devices:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        info.update(cupy=cp.__version__, gpu=props["name"].decode(),
                    gpu_mem_gb=round(props["totalGlobalMem"] / 2**30, 1),
                    cuda_driver=cp.cuda.runtime.driverGetVersion(),
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    return info


def warmup_and_check(df, args):
    """Load CUDA libraries/kernels before timing and validate the top-10 kernel."""
    import cupy as cp
    g = to_gpu(prepare(df, 0.01))
    U, V = als(g.R, g.Rt, k=args.k, iters=1, lam=args.lam, device="gpu")
    top10_kernel(U, V)
    U = U[:2000]
    a = top10_kernel(U, V)
    b = top10_xp(cp, U, V, block=512)
    S = cp.asnumpy(U @ V.T)
    # Compare scores, not indices: tied scores may come back in any order.
    if not np.allclose(np.take_along_axis(S, a, 1), np.take_along_axis(S, b, 1)):
        sys.exit("top-10 kernel does not match cupy.argpartition")
    gpu_free()
    print("warm-up ok, top-10 kernel matches cupy")


def save_sample(path, data_dir, d, top, n_users=5):
    movies_csv = os.path.join(data_dir, "movies.csv")
    titles = pd.read_csv(movies_csv, index_col="movieId")["title"] if os.path.exists(movies_csv) else None
    rows = []
    for u in range(n_users):
        for rank, item in enumerate(top[u], 1):
            mid = int(d.item_ids[item])
            rows.append({"userId": int(d.user_ids[u]), "rank": rank, "movieId": mid,
                         "title": titles.get(mid, "") if titles is not None else ""})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description="ALS CPU vs GPU benchmark on MovieLens 32M")
    p.add_argument("--data", default="data/ml-32m")
    p.add_argument("--out", default="results")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cpu-reps-full", type=int, default=1, help="CPU reps when frac == 1.0")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--gpu-argpartition", action=argparse.BooleanOptionalAction, default=True,
                   help="also time cupy.argpartition top-10 to compare with the kernel")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "env.json"), "w") as f:
        json.dump(env_info(args), f, indent=2)

    df, t_read = read_ratings(args.data)
    print(f"lectura_csv: {t_read:.2f}s ({len(df):,} ratings)")
    if "gpu" in args.devices:
        warmup_and_check(df, args)

    out_csv = os.path.join(args.out, "results.csv")
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()

    for frac in args.fracs:
        t0 = time.perf_counter()
        d = prepare(df, frac)
        t_prep = time.perf_counter() - t0
        base = {"frac": frac, "n_ratings": d.n_ratings, "n_users": d.R.shape[0], "n_items": d.R.shape[1]}
        rows = []

        def add(etapa, dev, times, err=""):
            rows.append({**base, "etapa": etapa, "dispositivo": dev, "reps": len(times),
                         "tiempo_s": round(statistics.median(times), 4),
                         "tiempos": ";".join(f"{t:.4f}" for t in times),
                         "rmse": "" if err == "" else round(err, 6)})
            print(f"[{frac}] {etapa:<20} {dev}  {statistics.median(times):9.3f}s  {err}")

        # The CSV is read once; every run needs the full file, so the same time applies to all fractions.
        add("lectura_csv", "cpu", [t_read])
        add("construir_csr", "cpu", [t_prep])
        top_gpu = None

        if "cpu" in args.devices:
            reps = args.cpu_reps_full if frac >= 1.0 else args.reps
            (U, V), times = measure(lambda: als(d.R, d.Rt, args.k, args.iters, args.lam, "cpu"), reps)
            add(f"als_{args.iters}_iter", "cpu", times, rmse(np, U, V, d.u_test, d.i_test, d.r_test))
            _, times = measure(lambda: top10_xp(np, U, V), reps)
            add("top10", "cpu", times)
            del U, V

        if "gpu" in args.devices:
            import cupy as cp
            g, times = measure(lambda: to_gpu(d), args.reps, sync=gpu_sync)
            add("transferencia", "gpu", times)
            (U, V), times = measure(lambda: als(g.R, g.Rt, args.k, args.iters, args.lam, "gpu"),
                                    args.reps, sync=gpu_sync)
            add(f"als_{args.iters}_iter", "gpu", times, rmse(cp, U, V, g.u_test, g.i_test, g.r_test))
            top_gpu, times = measure(lambda: top10_kernel(U, V), args.reps, sync=gpu_sync)
            add("top10", "gpu", times)
            if args.gpu_argpartition:
                # Reference only and slow, so a single rep.
                _, times = measure(lambda: top10_xp(cp, U, V, block=512), 1, sync=gpu_sync)
                add("top10_argpartition", "gpu", times)
            del g, U, V
            gpu_free()

        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerows(rows)
        if frac >= 1.0 and top_gpu is not None:
            save_sample(os.path.join(args.out, "top10_muestra.csv"), args.data, d, top_gpu)

    print(f"results written to {out_csv}")


if __name__ == "__main__":
    main()
