import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"cpu": "#2a78d6", "gpu": "#eb6834"}
FRAC_SHADES = ["#9ec5f4", "#5a9be6", "#1f5fae"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def summarize(df):
    """Per-fraction speedups, end-to-end totals and Amdahl's law."""
    rows = []
    for frac, g in df.groupby("frac"):
        t = {(r.etapa, r.dispositivo): r.tiempo_s for r in g.itertuples()}
        als = next(e for e, _ in t if e.startswith("als_"))
        seq = t[("lectura_csv", "cpu")] + t[("construir_csr", "cpu")]
        par_cpu = t[(als, "cpu")] + t[("top10", "cpu")]
        par_gpu = t[(als, "gpu")] + t[("top10", "gpu")]
        total_cpu = seq + par_cpu
        total_gpu = seq + t[("transferencia", "gpu")] + par_gpu
        p = par_cpu / total_cpu
        s = par_cpu / par_gpu
        for etapa, tc, tg in [(als, t[(als, "cpu")], t[(als, "gpu")]),
                              ("top10", t[("top10", "cpu")], t[("top10", "gpu")]),
                              ("total", total_cpu, total_gpu)]:
            rows.append({"frac": frac, "etapa": etapa, "t_cpu_s": round(tc, 3),
                         "t_gpu_s": round(tg, 3), "speedup": round(tc / tg, 2)})
        rows.append({"frac": frac, "etapa": "amdahl", "p_paralelo": round(p, 4),
                     "speedup_paralelo": round(s, 2),
                     "amdahl_max": round(1 / (1 - p), 2),
                     "amdahl_predicho": round(1 / ((1 - p) + p / s), 2),
                     "speedup": round(total_cpu / total_gpu, 2)})
    return pd.DataFrame(rows)


def fmt_s(x):
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.1f} s" if x < 100 else f"{x / 60:.1f} min"


def plot_stages(df, path):
    frac = df["frac"].max()
    g = df[(df["frac"] == frac) & (df["etapa"] != "top10_argpartition")]
    stages = list(dict.fromkeys(g["etapa"]))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(stages) + 1.4))
    for dev, dy in (("cpu", 0.15), ("gpu", -0.15)):
        sub = g[g["dispositivo"] == dev].set_index("etapa")
        ys = [len(stages) - 1 - stages.index(e) + dy for e in sub.index]
        ax.scatter(sub["tiempo_s"], ys, s=60, color=COLORS[dev], label=dev.upper(),
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        for x, y in zip(sub["tiempo_s"], ys):
            ax.annotate(fmt_s(x), (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(stages)), stages[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("tiempo (s, escala log)")
    ax.set_title(f"Tiempo por etapa, {frac:.0%} de los datos")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scaling(df, path):
    als = next(e for e in df["etapa"] if e.startswith("als_"))
    stages = [als, "top10", "construir_csr"]
    fig, axes = plt.subplots(1, len(stages), figsize=(12, 3.8))
    for ax, etapa in zip(axes, stages):
        for dev in ("cpu", "gpu"):
            sub = df[(df["etapa"] == etapa) & (df["dispositivo"] == dev)].sort_values("n_ratings")
            if sub.empty:
                continue
            x = sub["n_ratings"] / 1e6
            ax.plot(x, sub["tiempo_s"], color=COLORS[dev], linewidth=2, marker="o",
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=dev.upper())
            ax.annotate(dev.upper(), (x.iloc[-1], sub["tiempo_s"].iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT2)
        ax.set_title(etapa)
        ax.set_xlabel("calificaciones (millones)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("tiempo (s)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Escalado con el tamaño de los datos", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_speedup(summary, path):
    s = summary[summary["etapa"] != "amdahl"]
    stages = list(dict.fromkeys(s["etapa"]))
    fracs = sorted(s["frac"].unique())
    h = 0.8 / len(fracs)
    fig, ax = plt.subplots(figsize=(8, 1.2 * len(stages) + 1.2))
    for j, frac in enumerate(fracs):
        sub = s[s["frac"] == frac].set_index("etapa").reindex(stages)
        ys = [len(stages) - 1 - i + (len(fracs) / 2 - j - 0.5) * h for i in range(len(stages))]
        ax.barh(ys, sub["speedup"], height=h - 0.03, color=FRAC_SHADES[j % len(FRAC_SHADES)],
                label=f"{frac:.0%}")
        for y, v in zip(ys, sub["speedup"]):
            ax.annotate(f"×{v:.1f}", (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT2)
    ax.axvline(1, color=TEXT2, linewidth=1)
    ax.set_yticks(range(len(stages)), stages[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("speedup (T_CPU / T_GPU)")
    ax.set_title("Speedup GPU vs CPU")
    ax.legend(title="datos", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    df = pd.read_csv(os.path.join(args.results, "results.csv"))
    summary = summarize(df)
    summary.to_csv(os.path.join(args.results, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    plot_stages(df, os.path.join(args.results, "tiempos_por_etapa.png"))
    plot_scaling(df, os.path.join(args.results, "escalado.png"))
    plot_speedup(summary, os.path.join(args.results, "speedup.png"))


if __name__ == "__main__":
    main()
