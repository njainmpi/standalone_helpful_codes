import numpy as np
import matplotlib.pyplot as plt

# Constants
r1_values = np.linspace(0.2, 1.8, 9)  # Adjusted for 9 values from 0.2 to 1.8
T1_pre = 2.4
R1_pre = 1 / T1_pre

# Generate a range of C values
C = np.linspace(0, 30, 100)

# Plotting setup
plt.figure(figsize=(10, 6))

for r1 in r1_values:
    # Compute R1,post and T1,post for each r1
    R1_post = r1 * C + R1_pre
    T1_post = 1 / R1_post
    plt.plot(C, T1_post, label=f'r1 = {r1:.1f}')

# Final plot details
plt.title('Plot of Blood T1 as a function of C for various r1 values', fontsize=30)
plt.xlabel('Concentration of Contrast Agent Injection (in mg/Kg)', fontsize=30, fontweight='bold')
plt.ylabel('T1 of Blood (in sec)', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.legend(fontsize=30, frameon=False)
# plt.ylim(0, 0.12)  
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()