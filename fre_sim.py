import numpy as np
import matplotlib.pyplot as plt
import math

def FRESignalChange(FRE_matrix, angles, rf_counts):
    plt.figure(figsize=(10, 6))
    for i, rf in enumerate(rf_counts):
        plt.plot(angles, FRE_matrix[:, i], label=f'{rf} RF Pulses', linewidth=3)
    plt.xlabel('Flip Angle (degrees)', fontsize=30, fontweight='bold')
    plt.ylabel('Flow Related Enhancement (a.u.)', fontsize=30, fontweight='bold')
    plt.title('FRE at Different Blood Velocities without Contrast Agent', fontsize=30, fontweight='bold')
    plt.xticks(fontsize=30, fontweight='bold')
    plt.yticks(fontsize=30, fontweight='bold')
    plt.legend(fontsize=30, frameon=False)
    # plt.ylim(0, 0.12)  
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Parameters
slice_thickness_um = 1000            # Slice thickness in micrometers
pulse_interval_ms = 4.968           # Interval between RF pulses in milliseconds
base_velocity =50000                  # Base blood velocity in µm/s
percent_step = 20                    # Percentage increase per step
num_steps = 6                      # Number of velocity steps

T1Blood = 2400                   # ms
RepetitionTime = 10                # ms
Angle = np.arange(15, 25)           # Flip angle from 1° to 90°

# Convert interval to seconds
pulse_interval = pulse_interval_ms / 1000  # seconds

# Generate velocities: 5% increments from base velocity
# velocities = np.array([
#     base_velocity * (1 + percent_step / 100) ** i for i in range(num_steps)
# ])

velocities = np.array([])
for i in range(num_steps):
    val = 1+(percent_step*i)/100
    
    velocity = base_velocity * val
    print(velocity)
    velocities = np.append([velocity], velocities)
    

print(velocities)  

# Calculate time in slice and number of RF pulses
time_in_slice = slice_thickness_um / velocities  # seconds
num_rf_pulses = (time_in_slice / pulse_interval).astype(int)
unique_rf_counts = np.unique(num_rf_pulses)


print (velocities)

print(num_rf_pulses)
# Create FRE matrix
FRE = np.zeros((len(Angle), len(unique_rf_counts)))

for i, theta in enumerate(Angle):
    SinValue = math.sin(math.radians(theta))
    CosValue = math.cos(math.radians(theta))
    ExponentialValue = math.exp(-(RepetitionTime / T1Blood))
    Constant = 1 - ((1 - ExponentialValue) / (1 - (ExponentialValue * CosValue)))

    for j, rf in enumerate(unique_rf_counts):
        FRE[i, j] = SinValue * Constant * ((ExponentialValue * CosValue) ** (rf - 1))

# Plot FRE
FRESignalChange(FRE, Angle, unique_rf_counts)
