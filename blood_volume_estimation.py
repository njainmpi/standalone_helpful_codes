import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def solve_blood_volume(Sv_rest, Sv_act, Sb_rest, Sb_act, V_voxel=0.02):
    warnings = []
    numerator = (Sv_act - Sv_rest) * V_voxel
    denominator = (Sb_act - Sb_rest)

    if denominator == 0:
        warnings.append("⚠️  Blood signals for rest and active are equal. Cannot compute blood volume accurately.")
        return None, None, warnings

    V_blood_fraction = numerator / denominator
    V_blood_mm3 = V_voxel * V_blood_fraction
    V_tissue_mm3 = V_voxel - V_blood_mm3

    if V_blood_fraction < 0:
        warnings.append("⚠️  Computed blood volume is negative. Check input signal data.")
    if V_blood_mm3 > V_voxel:
        warnings.append("⚠️  Blood volume exceeds total voxel volume. Check input signal data.")
    if V_tissue_mm3 > V_voxel:
        warnings.append("⚠️  Tissue volume exceeds total voxel volume. Check input signal data.")

    try:
        S_tissue = (Sv_rest * V_voxel - Sb_rest * V_blood_fraction) / (V_voxel - V_blood_fraction)
    except ZeroDivisionError:
        warnings.append("⚠️  Division by zero while computing S_tissue.")
        return None, None, warnings

    return V_blood_fraction * 100, S_tissue, warnings

def process_roi_file(filepath):
    data = np.loadtxt(filepath)
    data_with = data[:147]
    data_without = data[147:]

    def extract_percentiles(segment):
        segment = segment[7:147]
        reshaped = segment.reshape((20, 7))
        averaged = np.mean(reshaped, axis=1)
        act = np.percentile(averaged[1:8], 95)
        rest = np.percentile(averaged[11:19], 95)
        return act, rest, averaged

    with_act, with_rest, avg_with = extract_percentiles(data_with)
    without_act, without_rest, avg_without = extract_percentiles(data_without)

    return {
        "With_Active": with_act,
        "With_Rest": with_rest,
        "Without_Active": without_act,
        "Without_Rest": without_rest,
        "Avg_With": avg_with,
        "Avg_Without": avg_without
    }

def print_signal_table(blood_data, voxel_data):
    print("\nSignal Intensities (95th percentile grouped by ROI)")
    print("-" * 105)
    print("| {:<10} | {:>12} {:>12} | {:>14} {:>14} | {:>14} {:>14} |".format(
        "ROI", "With-Rest", "With-Act", "Without-Rest", "Without-Act", "Δ-Rest", "Δ-Act"
    ))
    print("-" * 105)

    for roi_name, data in [("Blood", blood_data), ("Voxel", voxel_data)]:
        with_rest = data["With_Rest"]
        with_act = data["With_Active"]
        without_rest = data["Without_Rest"]
        without_act = data["Without_Active"]

        delta_rest = with_act - with_rest
        delta_act = without_act - without_rest

        print("| {:<10} | {:>12.2f} {:>12.2f} | {:>14.2f} {:>14.2f} | {:>14.2f} {:>14.2f} |".format(
            roi_name,
            with_rest, with_act,
            without_rest, without_act,
            delta_rest, delta_act
        ))
    print("-" * 105)

def plot_volume_summary(V_blood_rest_mm3, V_tissue_rest_mm3, V_blood_act_mm3, V_tissue_act_mm3):
    labels = ['Rest', 'Active']
    blood_volumes = [V_blood_rest_mm3, V_blood_act_mm3]
    tissue_volumes = [V_tissue_rest_mm3, V_tissue_act_mm3]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    ax.bar(x, blood_volumes, width, label='Blood Volume (mm³)', bottom=0)
    ax.bar(x, tissue_volumes, width, label='Tissue Volume (mm³)', bottom=blood_volumes)

    ax.set_ylabel('Volume per Voxel (mm³)')
    ax.set_title('Blood vs Tissue Volume per Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_avg_signals(blood_data, voxel_data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(blood_data["Avg_With"], label='Blood With')
    axs[0, 0].set_title("Blood (With Contrast)")
    axs[0, 1].plot(blood_data["Avg_Without"], label='Blood Without')
    axs[0, 1].set_title("Blood (Without Contrast)")
    axs[1, 0].plot(voxel_data["Avg_With"], label='Voxel With')
    axs[1, 0].set_title("Voxel (With Contrast)")
    axs[1, 1].plot(voxel_data["Avg_Without"], label='Voxel Without')
    axs[1, 1].set_title("Voxel (Without Contrast)")

    for ax in axs.flat:
        ax.set_xlabel("Block")
        ax.set_ylabel("Signal Intensity")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_blood_voxel.py blood.txt voxel.txt")
        sys.exit(1)

    blood_file = sys.argv[1]
    voxel_file = sys.argv[2]

    blood_data = process_roi_file(blood_file)
    voxel_data = process_roi_file(voxel_file)

    warnings_list = []

    V_blood_rest, S_tissue_rest, rest_warnings = solve_blood_volume(
        voxel_data["Without_Rest"], voxel_data["With_Rest"],
        blood_data["Without_Rest"], blood_data["With_Rest"]
    )
    warnings_list.extend(["Rest: " + w for w in rest_warnings])

    V_blood_act, S_tissue_act, act_warnings = solve_blood_volume(
        voxel_data["Without_Active"], voxel_data["With_Active"],
        blood_data["Without_Active"], blood_data["With_Active"]
    )
    warnings_list.extend(["Active: " + w for w in act_warnings])

    print_signal_table(blood_data, voxel_data)

    if V_blood_rest is not None and V_blood_act is not None:
        print("\nFormulas:")
        print("--------------------------------------------------------------------------")
        print("Change in CBV (%)           = ((V_blood_act - V_blood_rest) / V_blood_rest) * 100")
        print("Change in relative CBV (%)  = ((ml/100ml_act - ml/100ml_rest) / ml/100ml_act) * 100")
        
        print("\nModel Output (absolute volumes per voxel and derived metrics)")
        print("--------------------------------------------------------------------------")
        

        V_voxel_mm3 = 0.2
        V_blood_rest_mm3 = (V_blood_rest / 100) * V_voxel_mm3
        V_tissue_rest_mm3 = V_voxel_mm3 - V_blood_rest_mm3

        V_blood_act_mm3 = (V_blood_act / 100) * V_voxel_mm3
        V_tissue_act_mm3 = V_voxel_mm3 - V_blood_act_mm3

        # Conversion to mL/100 mL
        def to_ml_per_100ml(mm3_value):
            return (mm3_value / V_voxel_mm3) * 100

        rest_ml_per_100ml = to_ml_per_100ml(V_blood_rest_mm3)
        act_ml_per_100ml = to_ml_per_100ml(V_blood_act_mm3)

        print(f"{'Condition':<10} | {'V_blood (mm³)':>15} | {'V_tissue (mm³)':>15} | {'Blood Vol %':>12} | {'ml/100ml':>10} | {'S_tissue':>10}")
        print("-" * 90)
        print(f"{'Rest':<10} | {V_blood_rest_mm3:15.6f} | {V_tissue_rest_mm3:15.6f} | {V_blood_rest:12.2f} | {rest_ml_per_100ml:10.2f} | {S_tissue_rest:10.2f}")
        print(f"{'Active':<10} | {V_blood_act_mm3:15.6f} | {V_tissue_act_mm3:15.6f} | {V_blood_act:12.2f} | {act_ml_per_100ml:10.2f} | {S_tissue_act:10.2f}")

        if V_blood_rest_mm3 > V_blood_act_mm3:
            warnings_list.append("⚠️  Blood volume during rest is greater than during activation. This may indicate incorrect data or model mismatch.")

        # Derived outputs
        cbv_change_percent = ((V_blood_act_mm3 - V_blood_rest_mm3) / V_blood_rest_mm3) * 100
        relative_cbv_change_percent = ((act_ml_per_100ml - rest_ml_per_100ml) / act_ml_per_100ml) * 100

        print("\nDerived CBV Changes")
        print("---------------------------")
        print(f"Change in CBV (absolute, %):       {cbv_change_percent:.2f} %")
        print(f"Change in relative CBV (ml/100ml): {relative_cbv_change_percent:.2f} %")

    if warnings_list:
        print("\n⚠️  Volume Validation Warnings:")
        print("--------------------------------------------------------------")
        for w in warnings_list:
            print(w)
