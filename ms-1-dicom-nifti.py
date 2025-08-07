import os
import pandas as pd
import SimpleITK as sitk
import logging
from tqdm import tqdm
import warnings
import re

# Disable all warnings from dicom2nifti and other libraries
warnings.filterwarnings('ignore')

# Configuration du logging
# THIS MUST BE DONE BEFORE ANY LOGGER CALLS
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define base directory
BASE_DIR = ""
INPUT_ROOT_DIR = os.path.join(BASE_DIR, "irm_sep_etude")
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "new_irm_output")
CSV_FILENAME = "conversion_summary.csv"

def extract_id_from_filename(filename):
    # Extract the ID (like 2d99 or 6f1) from the filename
    match = re.search(r'\.hap\.([a-f0-9]+)\.be\.hap', filename)
    if match:
        return match.group(1)
    return None

def convert_dicom_to_nifti(input_dir, output_dir, patient_id, timepoint):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group DICOM files by their ID
    dicom_groups = {}
    
    # First, scan all DICOM files and group them by ID
    for filename in os.listdir(input_dir):
        if filename.endswith('.DCM'):
            file_id = extract_id_from_filename(filename)
            if file_id:
                if file_id not in dicom_groups:
                    dicom_groups[file_id] = []
                dicom_groups[file_id].append(os.path.join(input_dir, filename))
    
    logger.info(f"Found {len(dicom_groups)} different series IDs")
    
    # Convert each group to NIfTI
    for series_id, dicom_files in dicom_groups.items():
        logger.info(f"Processing series ID: {series_id}")
        
        # Sort files to ensure correct order
        dicom_files.sort()
        
        # Include series_id in the output filename
        output_path = os.path.join(output_dir, f"{patient_id}_{timepoint}_{series_id}.nii.gz")
        
        try:
            # Create a reader and set the file names
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            
            # Read the image
            image = reader.Execute()
            
            # Permute axes if needed
            image = sitk.PermuteAxes(image, [2, 1, 0])
            
            # Write the NIfTI file
            sitk.WriteImage(image, output_path)
            logger.info(f"Successfully converted series {series_id} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error converting series {series_id}: {str(e)}")

def convert_patient_flair_to_nifti():
    """
    Converts DICOM series in a structured input directory to NIfTI files
    in an output directory and generates a CSV summary.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    logger.info(f"Output directory set to: {OUTPUT_ROOT_DIR}")

    csv_data = []
    
    # Get list of patient directories
    try:
        patient_dirs = [d for d in os.listdir(INPUT_ROOT_DIR) if os.path.isdir(os.path.join(INPUT_ROOT_DIR, d))]
    except FileNotFoundError:
        logger.error(f"Input directory not found: {INPUT_ROOT_DIR}")
        return

    if not patient_dirs:
        logger.warning(f"No patient directories found in {INPUT_ROOT_DIR}")
        return

    logger.info(f"Found {len(patient_dirs)} potential patient directories.")

    with tqdm(total=sum(len(os.listdir(os.path.join(INPUT_ROOT_DIR, p_dir))) for p_dir in patient_dirs if os.path.isdir(os.path.join(INPUT_ROOT_DIR, p_dir))), desc="Processing series") as pbar:
        for patient_id in patient_dirs:
            patient_input_path = os.path.join(INPUT_ROOT_DIR, patient_id)
            
            try:
                timepoint_dirs = [d for d in os.listdir(patient_input_path) if os.path.isdir(os.path.join(patient_input_path, d))]
            except FileNotFoundError:
                logger.warning(f"Patient directory not found or inaccessible: {patient_input_path}. Skipping.")
                continue

            for timepoint in timepoint_dirs:
                pbar.set_description(f"Processing {patient_id}/{timepoint}")
                dicom_folder = os.path.join(patient_input_path, timepoint)
                
                # Convert DICOM files to NIfTI
                convert_dicom_to_nifti(dicom_folder, OUTPUT_ROOT_DIR, patient_id, timepoint)
                pbar.update(1)

    logger.info("Processing complete.")

if __name__ == "__main__":
    convert_patient_flair_to_nifti() 