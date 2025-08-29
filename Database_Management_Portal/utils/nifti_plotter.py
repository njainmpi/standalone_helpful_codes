# utils/nifti_plotter.py
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import os

def generate_coronal_slice_html(nifti_path, output_folder, max_slices=100):
    img = nib.load(nifti_path)
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 0]  # Use first timepoint

    ny = data.shape[1]  # Coronal axis

    os.makedirs(output_folder, exist_ok=True)
    html_paths = []

    for i in range(ny):
        slice_img = np.rot90(data[:, i, :])  # coronal slice
        fig = go.Figure(data=go.Heatmap(z=slice_img, colorscale='Viridis', colorbar=None))
        fig.update_layout(width=300, height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))

        slice_path = os.path.join(output_folder, f"coronal_slice_{i}.html")
        fig.write_html(slice_path, include_plotlyjs=False, full_html=False)
        html_paths.append(slice_path)

        if len(html_paths) >= max_slices:
            break

    return html_paths
