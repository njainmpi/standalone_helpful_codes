#!/bin/bash

fmri_file="cleaned_mc_func.nii.gz"
n_components=100
base_dir="results"
voxel="32 22 21"   # <-- replace with your voxel coordinates (x y z)

for freq in 0.005 0.01 0.02 0.05 0.1 0.2; do
    out_dir="${base_dir}_freq${freq}"
    echo "Running PCA with freq-thr=${freq}, saving to ${out_dir}"

    # Run PCA
    python3 ~/Desktop/pca.py "$fmri_file" "$n_components" "$out_dir" \
        --mode temporal \
        --detrend 1 \
        --auto-drop \
        --freq-thr "$freq" \
        --pow-frac 0.5 \
        --tr 1.0

    # Extract voxel timecourse from func_pca_denoised.nii.gz
    echo "Extracting timecourse for voxel $voxel"
    fslmeants -i ${out_dir}/func_pca_denoised.nii.gz -c $voxel > ${out_dir}/voxel_timeseries.txt
done
