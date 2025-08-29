import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ----- data -----
r1_values = np.linspace(0.2, 1.8, 9)   # relaxivity values
T1_pre = 2.4
R1_pre = 1 / T1_pre
C = np.linspace(0, 30, 200)

# ----- style -----
plt.rcParams.update({
    "figure.dpi": 140,
    "axes.labelweight": "bold",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 14
})

# Color-blind–friendly palette
cmap = get_cmap("tab10")   # categorical, good separation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

# ----- T1(C) with the ONLY legend -----
legend_lines = []
legend_labels = []

for i, r1 in enumerate(r1_values):
    color = cmap(i % 10)  # cycle through tab10
    R1_post = r1 * C + R1_pre
    T1_post = 1 / R1_post
    line, = ax1.plot(C, T1_post,
                     lw=1.5,          # thinner lines
                     color=color,
                     linestyle="-",
                     label=f"{r1:.1f}")
    legend_lines.append(line)
    legend_labels.append(f"r1 = {r1:.1f}")

ax1.set_title("Blood T1 vs. Contrast Agent Concentration", pad=8)
ax1.set_ylabel("T1 of Blood (s)")
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.axhline(T1_pre, lw=1.2, linestyle=":", alpha=0.8)

# Legend INSIDE T1 axes, upper right
ax1.legend(legend_lines, legend_labels,
           title="Relaxivity r1 (s⁻¹·mM⁻¹)",
           frameon=True, framealpha=0.9, facecolor="white",
           loc="upper right", ncol=3, borderaxespad=0.8)

# ----- R1(C) (no legend here) -----
for i, r1 in enumerate(r1_values):
    color = cmap(i % 10)
    R1_post = r1 * C + R1_pre
    ax2.plot(C, R1_post,
             lw=1.5,
             color=color,
             linestyle="-",
             label="_nolegend_")

ax2.set_title("Blood R1 vs. Contrast Agent Concentration", pad=8)
ax2.set_xlabel("Concentration of Contrast Agent Injection (mg/kg)")
ax2.set_ylabel("R1 of Blood (s⁻¹)")
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_xlim(0, 30)

plt.show()
