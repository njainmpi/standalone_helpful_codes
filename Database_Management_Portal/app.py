from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
from nilearn import plotting, image
import matplotlib.pyplot as plt
import base64
import io
import nibabel as nib
import sys

# Add this line:
ROOT_LOCATION = sys.argv[2] if len(sys.argv) > 2 else '/Users/njain/Desktop/Animal_Experiments_Sequences_v1'


app = Flask(__name__)
app.secret_key = 'your_secret_key'

EXCEL_PATH = sys.argv[1]
USERNAME = 'admin'
PASSWORD = 'password'


def find_run_folder(base_path, run_number):
    try:
        run_str = str(int(run_number)) if isinstance(run_number, float) else str(run_number).strip()
        print("🔎 Looking for folders in:", base_path)
        print("🔎 Looking for run string:", run_str)

        for folder in os.listdir(base_path):
            print("  ↪ Found folder:", folder)
            if os.path.isdir(os.path.join(base_path, folder)):
                if run_str in folder:
                    print("✅ Match found:", folder)
                    return os.path.join(base_path, folder)

        print("❌ No match found.")
    except Exception as e:
        print(f"[find_run_folder error] {e}")
    return None



EXPECTED_FILES = {
    "G1_cp.nii.gz": "Initial Raw 4D data",
    "mc_func.nii.gz": "Motion Corrected 4D data",
    "rest_rotation.jpg": "Rotational Movement",
    "rest_translation.jpg": "Translational Movement",
    "mean_mc_func.nii.gz": "Mean of Motion Corrected 4D Data",
    "initial_mean_mc_func.nii.gz": "Initial Mask on Mean Functional Image",
    "mask_mean_mc_func.nii.gz": "Final Mask on Mean Functional Image",
    "cleaned_mc_func.nii.gz": "Cleaned Functional Image (pre-Bias Correction)",
    "cleaned_N4_mean_mc_func.nii.gz": "Cleaned Mean Functional Image",
    "before_despiking_spikecountTC.png": "No of spikes before despiking data",
    "after_despiking_spikecountTC.png": "No of spikes after despiking data",
    "despike_cleaned_mc_func.nii.gz": "Despiked Functional Data",
    "cleaned_sm_despike_cleaned_mc_func.nii.gz": "1 Voxel Smoothing after Despiking",
    "anatomy_to_func.txt": "Translation Matrix",
    "Coregistered_SCM.nii.gz": "Coregistered Signal Change Map"
}

def check_run_files(path):
    return {
        fname: {
            "label": label,
            "exists": os.path.exists(os.path.join(path, fname))
        }
        for fname, label in EXPECTED_FILES.items()
    }



def load_data():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip()

    # ✅ Rename your actual Excel headers to expected ones
    df.rename(columns={
        "Animal Id": "Dataset",
        "Project": "Project",
        "SubProject": "Subproject",
        "Functional": "Run"
    }, inplace=True)

    return df




# ✅ INSERT THIS HERE
def extract_grouped_from_excel(file_path, row_index):
    df = pd.read_excel(file_path, header=None)

    field_names = df.iloc[0]
    group_names = df.iloc[1]
    data_row = df.iloc[row_index + 1]

    # ✅ Build full dataset_info dictionary using all fields
    dataset_info = {}
    for i in range(len(field_names)):
        key = field_names[i]
        val = data_row[i]
        if pd.notna(key):
            dataset_info[key] = val
    # Patch for Excel headers
        # ✅ Fix for mismatched column headers
    if "Animal Id" in dataset_info and "Dataset" not in dataset_info:
        dataset_info["Dataset"] = dataset_info["Animal Id"]

    if "Functional" in dataset_info and "Run" not in dataset_info:
        dataset_info["Run"] = dataset_info["Functional"]

    if "SubProject" in dataset_info and "Subproject" not in dataset_info:
        dataset_info["Subproject"] = dataset_info["SubProject"]

    # ✅ Group fields based on row 2 values (starting at column 5)
    grouped = {}
    current_group = None
    for i in range(4, len(field_names)):  # Start from column 5
        group = group_names[i]
        key = field_names[i]
        val = data_row[i]

        if pd.notna(group):
            current_group = group

        if pd.notna(key) and current_group:
            if current_group not in grouped:
                grouped[current_group] = {}
            grouped[current_group][key] = val

    return dataset_info, grouped


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == USERNAME and request.form['password'] == PASSWORD:
            session['user'] = USERNAME
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    # return render_template('login.html')
    return render_template('login.html', year=datetime.now().year)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    df = load_data()
    projects = sorted(df['Project'].dropna().astype(str).unique())
    return render_template('projects.html', projects=projects)



@app.route('/status/<dataset>')
def file_status(dataset):
    df = load_data()

    row = df[df['Dataset'] == dataset].iloc[0]
    project = row[2]       # Column 3
    subproject = row[3]    # Column 4
    dataset_name = row[1]  # Column 2
    run_number = int(row[5])  # Column 6

    dataset_base = os.path.join(ROOT_LOCATION, project, subproject, dataset_name)
    run_path = find_run_folder(dataset_base, run_number)

    if run_path is None:
        return f"No folder found for run number {run_number} in {dataset_base}", 404

    file_status = check_run_files(run_path)

    return render_template("run_status.html",
                           dataset=dataset_name,
                           project=project,
                           subproject=subproject,
                           run_number=run_number,
                           status=file_status)



@app.route('/project/<project_name>')
def show_subprojects(project_name):
    if 'user' not in session:
        return redirect(url_for('login'))

    df = load_data()
    subprojects = sorted(df[df['Project'] == project_name]['Subproject'].unique())
    return render_template(
        'subprojects.html',
        project=project_name,
        subprojects=subprojects,
        subprojects_menu=subprojects  # for dropdown
)



@app.route('/project/<project_name>/<subproject_name>')
def show_datasets(project_name, subproject_name):
    if 'user' not in session:
        return redirect(url_for('login'))

    df = load_data()
    filtered = df[(df['Project'] == project_name) & (df['Subproject'] == subproject_name)].reset_index()

    # Group datasets by scan month
    monthly_groups = defaultdict(list)

    for _, row in filtered.iterrows():
        dataset_name = row['Dataset']
        index = row['index']

        try:
            # Extract YYMMDD (e.g., 250407) from name
            yymmdd = dataset_name.split('_')[1]
            scan_date = datetime.strptime(yymmdd, "%y%m%d")
            month_key = scan_date.strftime("%B %Y")  # e.g., "April 2025"
        except Exception:
            month_key = "Unknown"

        monthly_groups[month_key].append((dataset_name, index))

    # Sort the months by date
    sorted_groups = dict(sorted(monthly_groups.items(),
                                key=lambda x: datetime.strptime(x[0], "%B %Y") if x[0] != "Unknown" else datetime.min,
                                reverse=True))

    subprojects_menu = sorted(df[df['Project'] == project_name]['Subproject'].dropna().unique())
    dataset_names_menu = {
        name: 1 for name in filtered['Dataset'].unique()
    }

    return render_template(
        'datasets.html',
        project=project_name,
        subproject=subproject_name,
        grouped_datasets=sorted_groups,
        subprojects_menu=subprojects_menu,
        dataset_names_menu=dataset_names_menu
    )

@app.route('/dataset/<dataset_name>')
def dataset_detail(dataset_name):
    if 'user' not in session:
        return redirect(url_for('login'))

    df = load_data()
    row = df[df["Dataset"] == dataset_name]
    if row.empty:
        return f"No dataset found with name {dataset_name}"

    dataset_info = row.iloc[0].to_dict()
    return render_template('dataset_detail.html', dataset=dataset_info, grouped=grouped)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

from flask import request

@app.route('/dataset_by_index/<int:row_id>')
def dataset_detail_by_index(row_id):
    if 'user' not in session:
        return redirect(url_for('login'))

    tab = request.args.get("tab", "metadata")

    try:
        dataset_info, grouped = extract_grouped_from_excel(EXCEL_PATH, row_id)
    except Exception as e:
        return f"Failed to parse dataset row {row_id}: {e}"

    df = load_data()
    project_name = dataset_info.get("Project")
    subproject_name = dataset_info.get("Subproject")
    run_number = str(dataset_info.get("Run")).strip()

    # Fallback if dataset field is missing
    if "Animal Id" in dataset_info and "Dataset" not in dataset_info:
        dataset_info["Dataset"] = dataset_info["Animal Id"]

    df_same = df[df["Project"] == project_name]
    subprojects_menu = sorted(df_same["Subproject"].dropna().unique())
    dataset_names_menu = {
        name: 1 for name in df[df["Subproject"] == subproject_name]["Dataset"].unique()
    }

    run_path = None
    file_status = None
    fmri_slices = []
    fmri_images= []
    fmri_images_by_label = {}

    def get_coronal_previews(nifti_path):
        """Generate coronal slice previews from a NIfTI file using Nilearn."""
        try:
            img = nib.load(nifti_path)
            n_slices = img.shape[1]  # Coronal axis
            previews = []

            for y in range(n_slices):
                display = plotting.plot_anat(img, display_mode='y', cut_coords=[y], title=None)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                previews.append(img_str)

            return previews
        except Exception as e:
            print(f"[Error generating coronal slices]: {e}")
            return []

    
    # === Handle processing and fMRI tabs ===

    def sanitize_path_component(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)

    if tab in ["processing", "fmri"]:
        dataset_name = sanitize_path_component(dataset_info.get("Dataset"))
        project_name = sanitize_path_component(dataset_info.get("Project"))
        subproject_name = sanitize_path_component(dataset_info.get("Subproject"))
        run_number = sanitize_path_component(dataset_info.get("Run"))

        dataset_base = os.path.join(ROOT_LOCATION, project_name, subproject_name, dataset_name)
        
        print("📁 ROOT_LOCATION =", ROOT_LOCATION)
        print("📁 Project =", project_name)
        print("📁 Subproject =", subproject_name)
        print("📁 Dataset =", dataset_name)
        print("📁 Run Number (raw) =", dataset_info.get("Run"))
        print("📁 Run Number (used) =", run_number)


        run_path = find_run_folder(dataset_base, run_number)

    if run_path:
        if tab == "processing":
            file_status = check_run_files(run_path)

        elif tab == "fmri":
        

            temp_img_dir = os.path.join(run_path, "temp_imgs")
            print("🔍 Looking for PNGs in:", temp_img_dir)
            
            if os.path.exists(temp_img_dir):
                for fname in sorted(os.listdir(temp_img_dir)):
                    if fname.lower().endswith(".png"):
                        full_path = os.path.join(temp_img_dir, fname)
                        with open(full_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                            fmri_slices.append({
                                "name": fname,
                                "data": img_data
                            })
            

            if run_path:

                temp_dir = os.path.join(run_path, "temp_imgs")


                if os.path.exists(temp_dir):
                    all_pngs = [f for f in os.listdir(temp_dir) if f.endswith(".png")]

                    for filename, label in EXPECTED_FILES.items():
                        prefix = filename.split(".")[0]
                        matched_files = sorted([
                            f for f in all_pngs if f.startswith(prefix + "_slice")
                        ])

                        img_data_list = []
                        for f in matched_files:
                            img_path = os.path.join(temp_dir, f)
                            try:
                                with open(img_path, "rb") as img_f:
                                    b64 = base64.b64encode(img_f.read()).decode("utf-8")
                                    img_data_list.append(f"data:image/png;base64,{b64}")
                            except Exception as e:
                                print(f"[Error reading image {f}]: {e}")

                        if img_data_list:
                            fmri_images_by_label[label] = img_data_list





                    # for filename, label in EXPECTED_FILES.items():
                    #     prefix = filename.split(".")[0]
                    #     matched_imgs = sorted([
                    #         os.path.join("static_proxy", "data", project_name, subproject_name, dataset_name, os.path.basename(run_path), "temp_imgs", f)
                    #         for f in all_pngs if f.startswith(prefix + "_slice")
                    #     ])
                    #     if matched_imgs:
                    #         fmri_images_by_label[label] = matched_imgs
    
    return render_template(
        'dataset_detail.html',
        dataset=dataset_info,
        grouped=grouped,
        project=project_name,
        subproject=subproject_name,
        subprojects_menu=subprojects_menu,
        dataset_names_menu=dataset_names_menu,
        tab=tab,
        run_path=run_path,
        file_status=file_status,
        fmri_images=fmri_images,
        fmri_slices=fmri_slices,
        fmri_images_by_label=fmri_images_by_label
    )


    grouped = {}
    remaining = dataset_info.copy()
    for group, fields in CATEGORY_MAP.items():
        group_data = {k: v for k, v in dataset_info.items() if k in fields and k not in ['Project', 'Subproject', 'Dataset']}
        if group_data:
            grouped[group] = group_data
            for k in group_data:
                remaining.pop(k, None)

    # Add any ungrouped fields under "Other"
    other = {k: v for k, v in remaining.items() if k not in ['Project', 'Subproject', 'Dataset']}
    if other:
        grouped["Other"] = other

    project_name = dataset_info.get("Project")
    subproject_name = dataset_info.get("Subproject")

    df_same = df[df["Project"] == project_name]
    subprojects_menu = sorted(df_same["Subproject"].dropna().unique())
    dataset_names_menu = {
        name: 1 for name in df[df["Subproject"] == subproject_name]["Dataset"].unique()
    }

    return render_template(
        'dataset_detail.html',
        dataset=dataset_info,
        grouped=grouped,
        project=project_name,
        subproject=subproject_name,
        subprojects_menu=subprojects_menu,
        dataset_names_menu=dataset_names_menu
    )   



@app.template_filter('format_value')
def format_value(value):
    try:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        elif isinstance(value, (float, int)):
            return f"{value}"
        elif pd.isna(value):
            return "-"
        return str(value)
    except:
        return str(value)



if __name__ == '__main__':
    app.run(debug=True)
