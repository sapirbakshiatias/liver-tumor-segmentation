"""
Crop liver region from CT volumes using liver masks.
Loads from Nifti_Volumes/ + Liver_Masks/, saves cropped volumes to Cropped_Data/CT/ and Cropped_Data/Masks/.

Note: np.clip(-100, 400) below is preserved from the original script.
extract_radiomics.py bypasses Cropped_Data/CT/ and loads from Nifti_Volumes/ directly
to avoid this clipping affecting raw uint16 HU+1024 values.
"""

import os
import nibabel as nib
import numpy as np

base_dir = r"C:\Users\ronin\PycharmProjects\PFinalproject"
ct_dir   = os.path.join(base_dir, "Nifti_Volumes")
mask_dir = os.path.join(base_dir, "Liver_Masks")
output_dir = os.path.join(base_dir, "Cropped_Data")

os.makedirs(os.path.join(output_dir, "CT"),    exist_ok=True)
os.makedirs(os.path.join(output_dir, "Masks"), exist_ok=True)


def crop_roi():
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")]
    for mask_name in mask_files:
        ct_name  = mask_name.replace("_mask.nii.gz", ".nii.gz")
        ct_path  = os.path.join(ct_dir,   ct_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(ct_path):
            print(f"  SKIP  {ct_name} (CT not found)")
            continue

        ct_nii   = nib.load(ct_path)
        mask_nii = nib.load(mask_path)
        ct_data   = ct_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        coords = np.array(np.where(mask_data > 0))
        if coords.size == 0:
            print(f"  SKIP  {ct_name} (empty mask)")
            continue

        min_c  = coords.min(axis=1)
        max_c  = coords.max(axis=1)
        margin = 5
        min_c  = np.maximum(min_c - margin, 0)
        max_c  = np.minimum(max_c + margin, np.array(ct_data.shape))

        cropped_ct   = ct_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
        cropped_mask = mask_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

        cropped_ct = np.clip(cropped_ct, -100, 400)

        new_ct_nii   = nib.Nifti1Image(cropped_ct,                  ct_nii.affine)
        new_mask_nii = nib.Nifti1Image(cropped_mask.astype(np.uint8), ct_nii.affine)

        nib.save(new_ct_nii,   os.path.join(output_dir, "CT",    f"cropped_{ct_name}"))
        nib.save(new_mask_nii, os.path.join(output_dir, "Masks", f"cropped_{mask_name}"))

        print(f"  OK  {ct_name} -> {cropped_ct.shape}")

    print("All files cropped and saved to Cropped_Data.")


if __name__ == "__main__":
    print("Cropping liver ROI from CT volumes...")
    crop_roi()
