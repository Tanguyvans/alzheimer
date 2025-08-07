import os
import logging
import SimpleITK as sitk
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def extract_series_info_from_adni_filename(filename: str) -> str:
    """Extract series info from ADNI DICOM filename."""
    # For ADNI files like: ADNI_137_S_1414_MR_MP-RAGE__br_raw_20080226130530486_1_S46193_I92752.dcm
    # Extract the series ID (I92752) at the end
    match = re.search(r'_I(\d+)\.dcm$', filename)
    return match.group(1) if match else None

def extract_id_from_filename(filename: str) -> str:
    """Extract the ID from the DICOM filename (supports both original and ADNI formats)."""
    # Try ADNI format first
    adni_id = extract_series_info_from_adni_filename(filename)
    if adni_id:
        return adni_id
    
    # Try original format
    match = re.search(r'\.hap\.([a-f0-9]+)\.be\.hap', filename)
    return match.group(1) if match else None

def convert_dicom_to_nifti(input_dir: str, output_dir: str, patient_id: str, timepoint: str) -> List[str]:
    """
    Convert DICOM files to NIfTI format.
    
    Args:
        input_dir: Directory containing DICOM files
        output_dir: Directory to save NIfTI files
        patient_id: Patient identifier
        timepoint: Timepoint identifier
        
    Returns:
        List of output file paths created
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DICOM files (both .dcm and .DCM extensions)
    dicom_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.dcm'):
            dicom_files.append(os.path.join(input_dir, filename))
    
    if not dicom_files:
        logger.warning(f"No DICOM files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(dicom_files)} DICOM files")
    
    # For ADNI data, typically all files in a folder belong to the same series
    # Sort files to ensure correct order
    dicom_files.sort()
    
    # Create output filename
    series_info = extract_id_from_filename(os.path.basename(dicom_files[0]))
    if series_info:
        output_filename = f"{patient_id}_{timepoint}_{series_info}.nii.gz"
    else:
        output_filename = f"{patient_id}_{timepoint}.nii.gz"
    
    output_path = os.path.join(output_dir, output_filename)
    
    output_files = []
    
    try:
        # Create a reader and set the file names
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        
        # Read the image
        image = reader.Execute()
        
        # Write the NIfTI file (no axis permutation for standard DICOM)
        sitk.WriteImage(image, output_path)
        logger.info(f"Successfully converted {len(dicom_files)} DICOM files to {output_path}")
        output_files.append(output_path)
        
    except Exception as e:
        logger.error(f"Error converting DICOM files in {input_dir}: {str(e)}")
    
    return output_files

def convert_patient_directory(input_root: str, output_root: str, patient_id: str = None) -> List[str]:
    """
    Convert all DICOM files in a patient directory structure to NIfTI.
    Handles both original structure and ADNI structure (patient/sequence/date/series/).
    
    Args:
        input_root: Root directory containing patient folders
        output_root: Root directory to save NIfTI files
        patient_id: Specific patient ID to process (if None, process all)
        
    Returns:
        List of all output file paths created
    """
    os.makedirs(output_root, exist_ok=True)
    
    all_output_files = []
    
    # Get list of patient directories
    if patient_id:
        patient_dirs = [patient_id] if os.path.exists(os.path.join(input_root, patient_id)) else []
    else:
        patient_dirs = [d for d in os.listdir(input_root) 
                       if os.path.isdir(os.path.join(input_root, d))]
    
    logger.info(f"Processing {len(patient_dirs)} patient directories")
    
    for patient in patient_dirs:
        patient_path = os.path.join(input_root, patient)
        
        try:
            sequence_dirs = [d for d in os.listdir(patient_path) 
                           if os.path.isdir(os.path.join(patient_path, d))]
        except FileNotFoundError:
            logger.warning(f"Patient directory not accessible: {patient_path}")
            continue
        
        for sequence in sequence_dirs:
            sequence_path = os.path.join(patient_path, sequence)
            
            try:
                date_dirs = [d for d in os.listdir(sequence_path) 
                           if os.path.isdir(os.path.join(sequence_path, d))]
            except FileNotFoundError:
                logger.warning(f"Sequence directory not accessible: {sequence_path}")
                continue
            
            for date_dir in date_dirs:
                date_path = os.path.join(sequence_path, date_dir)
                
                # Check if this directory contains DICOM files directly
                dicom_files = [f for f in os.listdir(date_path) if f.lower().endswith('.dcm')]
                
                if dicom_files:
                    # DICOM files found directly in date directory
                    timepoint = f"{sequence}_{date_dir}"
                    output_files = convert_dicom_to_nifti(date_path, output_root, patient, timepoint)
                    all_output_files.extend(output_files)
                else:
                    # Check for subdirectories (like ADNI's I##### folders)
                    try:
                        series_dirs = [d for d in os.listdir(date_path) 
                                     if os.path.isdir(os.path.join(date_path, d))]
                        
                        for series_dir in series_dirs:
                            series_path = os.path.join(date_path, series_dir)
                            series_dicom_files = [f for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
                            
                            if series_dicom_files:
                                timepoint = f"{sequence}_{date_dir}_{series_dir}"
                                output_files = convert_dicom_to_nifti(series_path, output_root, patient, timepoint)
                                all_output_files.extend(output_files)
                                
                    except FileNotFoundError:
                        logger.warning(f"Date directory not accessible: {date_path}")
                        continue
    
    return all_output_files