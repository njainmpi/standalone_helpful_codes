import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# === FUNCTIONS ===
def compute_E(TR, T1):
    return np.exp(-TR / T1)

def compute_q(E, theta_deg):
    theta_rad = np.radians(theta_deg)
    return E * np.cos(theta_rad)

def compute_signal(TR, T1, theta_deg, slice_thickness_mm, velocity_mm_per_ms):
    n = slice_thickness_mm / (velocity_mm_per_ms * TR)
    E = compute_E(TR, T1)
    theta_rad = np.radians(theta_deg)
    q = compute_q(E, theta_deg)
    if q == 1:
        raise ValueError("q must not be equal to 1 to avoid division by zero.")
    coeff = (1 - E) / (1 - q)
    bracket_term = (1 - q**n) / (n * (1 - q))
    inner = coeff + (1 - coeff) * bracket_term
    return np.sin(theta_rad) * inner

# === PARAMETERS ===
TR = 4.968  # ms
T1_values = [2400, 450]  # ms
slice_thickness_mm = 0.5
baseline_velocity_um_per_ms = 5
num_increments = 5
theta_values = np.linspace(0, 90, 100)

# === PREPARE VELOCITIES ===
baseline_velocity_mm_per_ms = baseline_velocity_um_per_ms / 1000
velocity_factors = [1 + 0.2 * i for i in range(num_increments)]
velocity_mm_per_ms_list = [baseline_velocity_mm_per_ms * f for f in velocity_factors]
velocity_labels_um_per_ms = [v * 1000 for v in velocity_mm_per_ms_list]

# Percentage labels for x-axis: Baseline, 20%, 40%, ...
percent_labels = ["Baseline"] + [f"{(f - 1) * 100:.0f}%" for f in velocity_factors[1:]]

# === PLOT FRE vs FLIP ANGLE FOR EACH T1 ===
plt.figure(figsize=(10, 6))
linestyles = ['-', '--']
colors = ['blue', 'green']

for idx, T1 in enumerate(T1_values):
    for velocity in velocity_mm_per_ms_list:
        signal = [compute_signal(TR, T1, theta, slice_thickness_mm, velocity) for theta in theta_values]
        label = f'T1={T1} ms | v={velocity*1000:.1f} μm/ms'
        plt.plot(theta_values, signal, linestyle=linestyles[idx], color=colors[idx], label=label)

plt.title(f'FRE vs Flip Angle θ\n(Baseline Velocity = {baseline_velocity_um_per_ms} μm/ms)')
plt.xlabel('Flip Angle θ (degrees)')
plt.ylabel('Signal (FRE)')
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# === PLOT % CHANGE IN FRE vs RELATIVE VELOCITY FOR EACH T1 ===
plt.figure(figsize=(8, 5))

for idx, T1 in enumerate(T1_values):
    fre_at_20 = [compute_signal(TR, T1, 20, slice_thickness_mm, vel) for vel in velocity_mm_per_ms_list]
    baseline_fre = fre_at_20[0]
    percent_change_fre = [(f - baseline_fre) / baseline_fre * 100 for f in fre_at_20]
    plt.plot(
        velocity_labels_um_per_ms,
        percent_change_fre,
        marker='o',
        linestyle=linestyles[idx],
        color=colors[idx],
        label=f'T1 = {T1} ms'
    )

plt.title('Change in FRE vs Blood Velocity with baseline velocity of 5 mm/s', fontsize=30)
plt.xlabel('Blood Velocity (relative to baseline)', fontsize=30, fontweight='bold')
plt.ylabel('Δ FRE (%) from Baseline', fontsize=30, fontweight='bold')
plt.xticks(velocity_labels_um_per_ms, percent_labels, fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
# plt.grid(True)
# plt.box(False)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# === PRINT FRE TABLE AT θ = 20° INCLUDING RF PULSES (n) ===
theta_of_interest = 20
fre_table = []
headers = ["Velocity (μm/ms)", "RF Pulses (n)"] + [f"T1 = {T1} ms" for T1 in T1_values]

for velocity_mm_per_ms, velocity_um_per_ms in zip(velocity_mm_per_ms_list, velocity_labels_um_per_ms):
    n_rf = slice_thickness_mm / (velocity_mm_per_ms * TR)
    n_rf_rounded = round(n_rf)
    row = [f"{velocity_um_per_ms:.1f}", f"{n_rf_rounded}"]
    for T1 in T1_values:
        fre_val = compute_signal(TR, T1, theta_of_interest, slice_thickness_mm, velocity_mm_per_ms)
        row.append(f"{fre_val:.4f}")
    fre_table.append(row)

print(f"\nFRE values at θ = {theta_of_interest}°, including number of RF pulses experienced (rounded):\n")
print(tabulate(fre_table, headers=headers, tablefmt="grid"))
