#!/usr/bin/env python3
"""
Test the cleaned preprocessing pipeline:
DICOM → NIfTI → N4 Bias Correction → MNI Registration → Skull Stripping
"""

import os
import sys
import logging

# Add preprocessing to path
sys.path.insert(0, 'preprocessing')

from dicom_to_nifti import convert_patient_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("CLEAN PREPROCESSING PIPELINE TEST")
    print("="*60)
    print("Steps: DICOM → NIfTI → N4 + MNI Registration → Skull Stripping")
    print("="*60)
    
    # Configuration
    input_dir = "ADNI"
    output_dir = "ADNI_clean_processed"
    mni_template = "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Template: {mni_template}")
    
    # Test Step 1: DICOM to NIfTI conversion
    print("\n" + "="*40)
    print("STEP 1: DICOM TO NIFTI CONVERSION")
    print("="*40)
    
    nifti_dir = os.path.join(output_dir, "01_nifti")
    
    nifti_files = convert_patient_directory(input_dir, nifti_dir)
    print(f"✓ Converted {len(nifti_files)} DICOM series to NIfTI")
    
    # Show output structure
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    01_nifti/     # {len(nifti_files)} NIfTI files")
    print(f"    02_processed/ # N4 + MNI registration (next step)")
    print(f"    03_skull_stripped/ # Final brain images (final step)")

if __name__ == "__main__":
    main()