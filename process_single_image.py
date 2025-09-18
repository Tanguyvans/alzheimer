#!/usr/bin/env python3
"""
Process a single NIfTI image through the pipeline steps:
- N4 bias correction
- MNI template registration

Usage:
    python process_single_image.py <input_nifti> <output_nifti> <mni_template>
"""

import sys
import os
import logging

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))

from image_enhancement import process_nifti_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python process_single_image.py <input_nifti> <output_nifti> [mni_template]")
        print("\nExample:")
        print("python process_single_image.py \\")
        print("  /Volumes/KINGSTON/ADNI_nifti/002_S_0295/MP-RAGE_2009-05-22_07_00_57.0_I144446_144446.nii.gz \\")
        print("  /tmp/processed_output.nii.gz")
        print("\nTemplate is optional - default MNI template will be used if not specified")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Use default MNI template if not provided
    if len(sys.argv) == 4:
        template_file = sys.argv[3]
    else:
        template_file = "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
    
    # Validate inputs
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not os.path.exists(template_file):
        print(f"Error: Template file not found: {template_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract subject ID from filename
    subject_id = os.path.basename(input_file).replace('.nii.gz', '')
    
    print(f"Processing single image...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Template: {template_file}")
    print(f"Subject ID: {subject_id}")
    print()
    
    # Process the image
    success = process_nifti_file(input_file, output_file, template_file, subject_id)
    
    if success:
        print(f"\n✅ Processing completed successfully!")
        print(f"Output saved to: {output_file}")
    else:
        print(f"\n❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()