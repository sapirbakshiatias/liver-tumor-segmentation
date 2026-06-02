"""
Liver segmentation using TotalSegmentator.
Processes all NIfTI volumes in Nifti_Volumes/ and saves liver masks to Liver_Masks/.
Runs one patient at a time to save RAM.
"""

import sys, os; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")


import os
import gc
from totalsegmentator.python_api import totalsegmentator

base_dir   = r"C:\Users\ronin\PycharmProjects\PFinalproject"
input_dir  = os.path.join(base_dir, "Nifti_Volumes")
output_dir = os.path.join(base_dir, "Liver_Masks")
os.makedirs(output_dir, exist_ok=True)

print("Running liver segmentation (one patient at a time to save RAM)...")

for file_name in os.listdir(input_dir):
    if file_name.endswith(".nii.gz"):
        input_path       = os.path.join(input_dir, file_name)
        output_mask_path = os.path.join(
            output_dir, file_name.replace(".nii.gz", "_mask.nii.gz")
        )

        # Skip already-processed files
        if os.path.exists(output_mask_path) and os.path.getsize(output_mask_path) > 0:
            print(f"  SKIP  {file_name} (already exists)")
            continue

        print(f"\n  Segmenting: {file_name} ...")

        try:
            totalsegmentator(
                input_path,
                output_mask_path,
                task="total",
                ml=True,
                fast=True,
                roi_subset=["liver"]
            )
            print(f"  OK    {file_name}")

        except Exception as e:
            print(f"  FAIL  {file_name} — {e}")

        # Free memory before next patient
        gc.collect()

print("\nSegmentation complete.")
