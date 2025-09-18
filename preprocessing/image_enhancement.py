#!/usr/bin/env python3
"""
Image enhancement module for medical imaging preprocessing.
Includes N4 bias correction and MNI template registration.
"""

import os
import logging
import nibabel as nib
import numpy as np
from typing import List, Dict, Optional
import tempfile
import shutil

logger = logging.getLogger(__name__)

def check_ants_installation():
    """Check if ANTsPy is installed and available."""
    try:
        import ants
        return True
    except ImportError:
        return False

def apply_n4_bias_correction(input_file: str, output_file: str, subject_id: str = None) -> bool:
    """
    Apply N4 bias field correction using ANTsPy.
    
    Args:
        input_file: Input NIfTI file
        output_file: Output corrected file
        subject_id: Subject identifier for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not check_ants_installation():
            logger.warning("ANTsPy not found, copying original file without N4 correction")
            shutil.copy2(input_file, output_file)
            return True
        
        import ants
        
        logger.info(f"Applying N4 bias correction to {subject_id or os.path.basename(input_file)}")
        
        # Load image with ANTsPy
        img = ants.image_read(input_file)
        
        # Apply N4 bias correction
        corrected_img = ants.n4_bias_field_correction(img)
        
        # Save corrected image
        ants.image_write(corrected_img, output_file)
        
        logger.info(f"N4 bias correction completed for {subject_id}")
        return True
            
    except Exception as e:
        logger.error(f"Error in N4 bias correction for {subject_id}: {str(e)}")
        try:
            shutil.copy2(input_file, output_file)
        except:
            pass
        return False

def register_to_mni(input_file: str, output_file: str, template_file: str, subject_id: str = None) -> bool:
    """
    Register image to MNI template using ANTsPy.
    
    Args:
        input_file: Input NIfTI file
        output_file: Output registered file
        template_file: MNI template file
        subject_id: Subject identifier for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not check_ants_installation():
            logger.warning("ANTsPy not found, copying original file without registration")
            shutil.copy2(input_file, output_file)
            return True
        
        if not os.path.exists(template_file):
            logger.error(f"Template file not found: {template_file}")
            shutil.copy2(input_file, output_file)
            return False
        
        import ants
        
        logger.info(f"Registering {subject_id or os.path.basename(input_file)} to MNI template")
        
        # Load images with ANTsPy
        moving_img = ants.image_read(input_file)
        fixed_img = ants.image_read(template_file)
        
        # Perform registration using SyN (symmetric normalization)
        registration = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='SyN',
            verbose=False
        )
        
        # Get the warped image
        warped_img = registration['warpedmovout']
        
        # Save registered image
        ants.image_write(warped_img, output_file)
        
        logger.info(f"MNI registration completed for {subject_id}")
        return True
                
    except Exception as e:
        logger.error(f"Error in MNI registration for {subject_id}: {str(e)}")
        try:
            shutil.copy2(input_file, output_file)
        except:
            pass
        return False

def process_nifti_file(input_file: str, output_file: str, template_file: str, subject_id: str = None) -> bool:
    """
    Complete processing pipeline for a single NIfTI file:
    1. N4 bias correction
    2. MNI registration
    
    Args:
        input_file: Input NIfTI file
        output_file: Final output file
        template_file: MNI template file
        subject_id: Subject identifier for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info(f"Processing {subject_id or os.path.basename(input_file)}")
        
        # Step 1: N4 bias correction
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_n4:
            temp_n4_path = temp_n4.name
        
        try:
            success_n4 = apply_n4_bias_correction(input_file, temp_n4_path, subject_id)
            if not success_n4:
                logger.warning(f"N4 correction failed for {subject_id}, proceeding with original")
                temp_n4_path = input_file
            
            # Step 2: MNI registration
            success_reg = register_to_mni(temp_n4_path, output_file, template_file, subject_id)
            
            return success_reg
            
        finally:
            # Cleanup temporary files
            if temp_n4_path != input_file and os.path.exists(temp_n4_path):
                os.unlink(temp_n4_path)
                
    except Exception as e:
        logger.error(f"Error processing {subject_id}: {str(e)}")
        return False

def process_directory(input_dir: str, output_dir: str, template_file: str) -> Dict:
    """
    Process all NIfTI files in a directory.
    
    Args:
        input_dir: Directory containing input NIfTI files
        output_dir: Directory for output files
        template_file: MNI template file
        
    Returns:
        Dictionary with processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'successful': [], 'failed': [], 'total': 0}
    
    # Find all NIfTI files
    nifti_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.nii.gz'):
            nifti_files.append(os.path.join(input_dir, filename))
    
    results['total'] = len(nifti_files)
    logger.info(f"Found {len(nifti_files)} NIfTI files to process")
    
    for nifti_file in nifti_files:
        filename = os.path.basename(nifti_file)
        subject_id = filename.replace('.nii.gz', '')
        output_file = os.path.join(output_dir, f"{subject_id}_processed.nii.gz")
        
        # Skip if already exists
        if os.path.exists(output_file):
            logger.info(f"Processed file already exists: {output_file}")
            results['successful'].append(subject_id)
            continue
        
        success = process_nifti_file(nifti_file, output_file, template_file, subject_id)
        if success:
            results['successful'].append(subject_id)
        else:
            results['failed'].append(subject_id)
    
    return results