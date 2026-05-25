"""
Part A: Read DICOM files and build 3D volumes.
Part B: Filter, rename (Hebrew folder names -> English), and save as NIfTI.

Output: Nifti_Volumes/<patient>_s01_CT.nii.gz  (one file per series, all series saved)
"""

import os
import pydicom
import numpy as np
from collections import defaultdict
import SimpleITK as sitk

base_dir = r"C:\Users\ronin\PycharmProjects\PFinalproject"
all_patients_data = []

print("Scanning patient folders (this may take a moment)...")

for folder_name in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, folder_name)

    if (not os.path.isdir(patient_path)
            or folder_name.startswith('.')
            or folder_name in ('venv', '.venv', 'Nifti_Volumes', 'Liver_Masks',
                               'Cropped_Data', '__pycache__')):
        continue

    print(f"  Reading: {folder_name}")
    series_dict = defaultdict(list)

    for root, dirs, files in os.walk(patient_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                dcm = pydicom.dcmread(file_path, force=True)
                if hasattr(dcm, 'pixel_array'):
                    series_uid = dcm.get('SeriesInstanceUID', 'unknown_series')
                    series_dict[series_uid].append(dcm)
            except Exception:
                pass

    if not series_dict:
        continue

    for uid, slices in series_dict.items():
        if len(slices) < 10:
            continue

        shapes = {s.pixel_array.shape for s in slices}
        if len(shapes) > 1:
            continue

        try:
            slices.sort(key=lambda x: x.ImagePositionPatient[2])
            volume_3d = np.stack([s.pixel_array for s in slices])
            all_patients_data.append({
                "patient_id": folder_name,
                "series_uid": uid,
                "volume_3d":  volume_3d,
                "metadata":   slices[0],
            })
        except Exception:
            pass

print(f"\nSuccessfully loaded {len(all_patients_data)} 3D volumes.")

# ── Part B: filter, translate names, save as NIfTI ───────────────────────────

print("\nTranslating patient names and saving to NIfTI format...")

output_dir = os.path.join(base_dir, "Nifti_Volumes")
os.makedirs(output_dir, exist_ok=True)

translation_dict = {
    "פציינט 1 אחרי סרטן":          "Patient_1_After",
    "מטופל G-A ללא סרטן":                     "Patient_GA_Healthy",
    "מטופל K-B לפני סרטן":               "Patient_KB_Before",
    "מטופל K-B עם סרטן":                            "Patient_KB_Cancer",
    "מטופל V-T ללא סרטן":                     "Patient_VT_Healthy",
    "מטופל 1 לפני 3 פאזות":         "Patient_1_Before",
    "מטופל 2 אחרי סרטן":                 "Patient_2_After",
    "מטופל 2 לפני סרטן":                 "Patient_2_Before",
}

patient_names = {d["patient_id"] for d in all_patients_data}

for patient in patient_names:
    patient_volumes = [d for d in all_patients_data if d["patient_id"] == patient]

    valid_volumes = [
        v for v in patient_volumes
        if len(v["volume_3d"].shape) == 3 and v["volume_3d"].shape[1] == 512
    ]
    if not valid_volumes:
        continue

    # Sort by number of slices descending so s01 is always the largest series
    valid_volumes.sort(key=lambda v: v["volume_3d"].shape[0], reverse=True)

    clean_name = translation_dict.get(patient, f"Unknown_{patient[:20]}")

    for s_idx, vol in enumerate(valid_volumes, start=1):
        sitk_img = sitk.GetImageFromArray(vol["volume_3d"])
        try:
            xy_sp = vol["metadata"].PixelSpacing
            z_sp  = vol["metadata"].SliceThickness
            sitk_img.SetSpacing((float(xy_sp[0]), float(xy_sp[1]), float(z_sp)))
        except Exception:
            pass
        file_name   = f"{clean_name}_s{s_idx:02d}_CT.nii.gz"
        output_path = os.path.join(output_dir, file_name)
        sitk.WriteImage(sitk_img, output_path)
        n_slices = vol["volume_3d"].shape[0]
        print(f"  Saved: {file_name}  ({n_slices} slices)")

print("\nDone! NIfTI files ready in Nifti_Volumes/")
