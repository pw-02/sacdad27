import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
# --- Typography / style (match your paper style) ---
matplotlib.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["Linux Libertine", "Libertine", "Times"],
    "axes.labelsize": 14,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
})
# ----------------------------
# Data
# ----------------------------
models = ["NanoGPT (4L)", "MNIST-GAN"]
devices = ["GPU", "CPU"]
components = ["NTT", "MSM", "Other"]

prover_times = {
    "NanoGPT (4L)": {"GPU": 543.1294833, "CPU": 928.2107211},
    "MNIST-GAN": {"GPU": 237.4696116, "CPU": 438.788715},
}
fft_times = {
    "NanoGPT (4L)": {"GPU": 80.74473667, "CPU": 239.1086189},
    "MNIST-GAN": {"GPU": 22.8851258, "CPU": 133.2966129},
}
msm_times = {
    "NanoGPT (4L)": {"GPU": 153.1854232, "CPU": 358.5777848},
    "MNIST-GAN": {"GPU": 47.3280334, "CPU": 136.6266694},
}

# Compute "Other" = total - FFT - MSM
other_times = {
    m: {d: prover_times[m][d] - fft_times[m][d] - msm_times[m][d] for d in devices}
    for m in models
}

times_by_component = {"NTT": fft_times, "MSM": msm_times, "Other": other_times}

# ----------------------------
# Plot style
# ----------------------------
colors = {"NTT": "#D3D3D3", "MSM": "#FFA500", "Other": "#A52A2A"}
hatches = {"NTT": "", "MSM": "xxx", "Other": "...."}

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, len(models), figsize=(4.5, 2.9), sharey=True)
if len(models) == 1:
    axes = [axes]

bar_width = 0.5
x = np.arange(len(devices))

for ax, model in zip(axes, models):
    for j, dev in enumerate(devices):
        bottom = 0.0
        for comp in components:
            val = times_by_component[comp][model][dev]
            ax.bar(
                j,
                val,
                width=bar_width,
                bottom=bottom,
                color=colors[comp],
                edgecolor="black",
                linewidth=1.5,
                hatch=hatches[comp],
            )
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=10)

    # Title below subplot
    # ax.text(
    #     0.5, -0.28, model,
    #     transform=ax.transAxes,
    #     ha="center", va="top",
    #     fontsize=11, 
    # )
    ax.set_title(model, fontsize=11)

# Y label on first subplot
axes[0].set_ylabel("Proving Time (s)", fontsize=11)

# Legend INSIDE the first subplot only
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[c], edgecolor="black", hatch=hatches[c])
    for c in components
]
axes[0].legend(
    legend_handles,
    components,
    loc="upper left",
    fontsize=9,
    frameon=True,
)

axes[1].legend(
    legend_handles,
    components,
    loc="upper left",
    fontsize=9,
    frameon=True,
)

fig.tight_layout()

# Save BEFORE show
out_path = Path("figures/gpu/gpu_impact.pdf")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, bbox_inches="tight", dpi=300)

plt.show()
