import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- Typography / style ---
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "Libertine", "Times"],
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# --- Data ---
num_workers = [1, 2, 4, 6, 8, 10]
labels = ["1", "2", "4", "6", "8", "10"]

times = {
    "NanoGPT (4L)": [1108.401158, 448.284717, 226.143169, 144.75947, 110.780916, 89.403294],
    "MobileNet":    [9639, 3405.093471, 1713.609643, 1154.744915, 914.102607, 761.712696],
}   

ezkl_times = {"NanoGPT (4L)": 6165, "MobileNet": 11231}

# Colors
zkinfer_color = "#008080"
ezkl_color = "#212121"
self_color = "#d62728"   # red for self-speedup

# --- Compute speedups ---
speedup_vs_ezkl = {}
speedup_vs_self = {}

for model, t in times.items():
    t = np.array(t, dtype=float)
    speedup_vs_ezkl[model] = ezkl_times[model] / t
    speedup_vs_self[model] = t[0] / t   # relative to 1 worker

# --- Plot ---
fig_width, fig_height = 3.2, 2.6
x = np.arange(len(num_workers))

for model_label in times.keys():
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # baseline (ezkl = 1)
    ax.plot(x, [1.0]*len(x), "--", color=ezkl_color, lw=2.2, label="ezkl baseline (1×)")

    # speedup vs ezkl
    sp_ezkl = speedup_vs_ezkl[model_label]
    ax.plot(x, sp_ezkl, marker="o", color=zkinfer_color, lw=3.6, label="zkInfer vs ezkl")

    # speedup vs self
    sp_self = speedup_vs_self[model_label]
    ax.plot(x, sp_self, marker="s", color=self_color, lw=3.6, label="zkInfer vs 1 worker")

    # annotate last points only (clearer)
    ax.text(x[-1], sp_ezkl[-1]*1.05, f"x{sp_ezkl[-1]:.1f}",
            ha="center", va="bottom", fontsize=14, fontweight="semibold", color=zkinfer_color)
    ax.text(x[-1], sp_self[-1]*1.05, f"x{sp_self[-1]:.1f}",
            ha="center", va="bottom", fontsize=14, fontweight="semibold", color=self_color)

    # labels/ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup")

    # style
    ax.grid(True, axis="y", linestyle="--", lw=0.6, alpha=0.4)
    ax.set_ylim(0, max(sp_ezkl.max(), sp_self.max()) * 1.4)

    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"figures/distributed/multi-node/multi_node_scaling_{model_label.replace(' ', '_')}_speedup.pdf",
        dpi=400, bbox_inches="tight"
    )
    plt.show()
