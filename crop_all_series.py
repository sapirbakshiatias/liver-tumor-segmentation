"""
Crop liver ROI from ALL series in All_Series_Nifti/ using masks in All_Series_Masks/.
Only keeps series where liver (label 5) exists in the mask.
Output: Cropped_Data/All_Series_CT/ and Cropped_Data/All_Series_Masks/
"""

import os
import numpy as np
import nibabel as nib

BASE_DIR     = r"C:\Users\ronin\PycharmProjects\PFinalproject"
NIFTI_DIR    = os.path.join(BASE_DIR, "All_Series_Nifti")
MASK_DIR     = os.path.join(BASE_DIR, "All_Series_Masks")
OUT_CT_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
OUT_MASK_DIR = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")

HU_OFFSET   = 1024
LIVER_LABEL = 5
MARGIN      = 5

os.makedirs(OUT_CT_DIR,   exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

mask_files = sorted(f for f in os.listdir(MASK_DIR) if f.endswith(".nii.gz"))
kept = 0

for mask_name in mask_files:
    stem    = mask_name.replace("_mask.nii.gz", "")
    ct_name = f"{stem}.nii.gz"
    ct_path = os.path.join(NIFTI_DIR, ct_name)

    if not os.path.exists(ct_path):
        print(f"  SKIP  {ct_name}  (CT not found)")
        continue

    seg   = nib.load(os.path.join(MASK_DIR, mask_name)).get_fdata()
    liver = (seg == LIVER_LABEL).astype(np.uint8)

    if liver.sum() == 0:
        print(f"  SKIP  {stem}  (no liver in mask)")
        continue

    ct_nii  = nib.load(ct_path)
    ct_data = ct_nii.get_fdata().astype(float) - HU_OFFSET

    coords = np.array(np.where(liver > 0))
    min_c  = np.maximum(coords.min(axis=1) - MARGIN, 0)
    max_c  = np.minimum(coords.max(axis=1) + MARGIN, np.array(ct_data.shape))

    cropped_ct   = np.clip(ct_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]], -100, 400)
    cropped_mask = liver[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

    nib.save(nib.Nifti1Image(cropped_ct,                   ct_nii.affine),
             os.path.join(OUT_CT_DIR,   f"cropped_{ct_name}"))
    nib.save(nib.Nifti1Image(cropped_mask.astype(np.uint8), ct_nii.affine),
             os.path.join(OUT_MASK_DIR, f"cropped_{mask_name}"))

    print(f"  OK  {stem}  shape={cropped_ct.shape}")
    kept += 1

print(f"\nDone. {kept} series cropped and saved.")
