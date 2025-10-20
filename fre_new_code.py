import numpy as np
import matplotlib.pyplot as plt

# === Functions ===
def compute_E(TR, T1):
    return np.exp(-TR / T1)

def compute_q(E, theta_deg):
    theta_rad = np.radians(theta_deg)
    return E * np.cos(theta_rad)

def custom_expression_theta_sin(TR, T1, theta_deg, n):
    E = compute_E(TR, T1)
    theta_rad = np.radians(theta_deg)
    q = compute_q(E, theta_deg)

    if q == 1:
        raise ValueError("q must not be equal to 1 to avoid division by zero.")

    coeff = (1 - E) / (1 - q)
    bracket_term = (1 - q**n) / (n * (1 - q))
    inner_expression = coeff + (1 - coeff) * bracket_term
    result = np.sin(theta_rad) * inner_expression
    return result

# === Parameters ===
TR = 20     # milliseconds
T1 = 1200   # milliseconds
theta_values = np.linspace(0, 90, 100)  # degrees
n_choices = [1, 3, 5, 7, 10]

# === Compute results ===
results_by_n_theta_sin = {}
for n in n_choices:
    results_by_n_theta_sin[n] = [custom_expression_theta_sin(TR, T1, theta, n) for theta in theta_values]

# === Plotting ===
plt.figure(figsize=(10, 6))
for n in n_choices:
    plt.plot(theta_values, results_by_n_theta_sin[n], label=f'n = {n}')

plt.title(f'Signal vs Flip Angle θ (TR={TR} ms, T1={T1} ms)')
plt.xlabel('Flip Angle θ (degrees)')
plt.ylabel('Signal')
plt.grid(True)
plt.legend(title='Number of Repeats (n)')
plt.tight_layout()
plt.show()
