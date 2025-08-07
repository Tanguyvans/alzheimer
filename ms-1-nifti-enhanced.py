import os
import numpy as np
import pandas as pd
import ants
import logging
from tqdm import tqdm
import time
import warnings
import traceback

# Disable all warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration du logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define base directory and other paths
# IMPORTANT: Set your BASE_DIR here
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"
INPUT_NIFTI_DIR = os.path.join(BASE_DIR, "ms-1-nifti") # Updated input directory
OUTPUT_PROCESSED_DIR = os.path.join(BASE_DIR, "ms-1-register") # Updated output directory
# IMPORTANT: Ensure this template path is correct
TEMPLATE_PATH = os.path.join(BASE_DIR, "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii")
CSV_FILENAME = "registration_n4_summary.csv" # Updated CSV filename

# List of specific NIfTI files to process
TARGET_NIFTI_FILES = [
    "SEP-MRI-001_T0_6f1.nii.gz",
    "SEP-MRI-001_T1_be7d.nii.gz",
    "SEP-MRI-014_T0_a594.nii.gz",
    "SEP-MRI-014_T1_999d.nii.gz",
    "SEP-MRI-014_T2_37fe.nii.gz",
    "SEP-MRI-014_T3_7358.nii.gz",
    "SEP-MRI-014_T4_2000.nii.gz",
    "SEP-MRI-028_T0_59b4.nii.gz",
    "SEP-MRI-028_T1_8466.nii.gz",
    "SEP-MRI-028_T2_8615.nii.gz",
    "SEP-MRI-028_T3_e6ff.nii.gz",
    "SEP-MRI-038_T0_954b.nii.gz",
    "SEP-MRI-038_T1_6274.nii.gz",
    "SEP-MRI-038_T2_93ef.nii.gz",
    "SEP-MRI-038_T3_7007.nii.gz",
    "SEP-MRI-038_T4_8434.nii.gz",
    "SEP-MRI-043_T0_9bb0.nii.gz",
    "SEP-MRI-043_T1_ba2b.nii.gz",
    "SEP-MRI-043_T2_6194.nii.gz",
    "SEP-MRI-043_T3_bc65.nii.gz",
    "SEP-MRI-043_T4_d425.nii.gz"
]

# --- Functions ---

def register_to_mni_ants(input_file_path, mni_template_path, subject_id):
    """Registers an input NIfTI image to the MNI template using ANTs."""
    logger.info(f"Starting registration for {subject_id} from {input_file_path}")
    try:
        moving_image = ants.image_read(input_file_path)
        fixed_image = ants.image_read(mni_template_path)
        
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='SyN'  # Using SyN for non-linear registration
        )
        registered_image = registration['warpedmovout']
        logger.info(f"Registration successful for {subject_id}")
        return registered_image
    except Exception as e:
        logger.error(f"Registration failed for {subject_id} from {input_file_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def apply_n4_bias_correction(image, subject_id):
    """
    Applies N4 bias field correction to an ANTs image.
    """
    logger.info(f"Starting N4 bias field correction for image {subject_id}...")
    try:
        if image is None:
            logger.error(f"Input image for N4 correction is None for {subject_id}")
            return None

        logger.info(f"Performing N4 bias field correction for {subject_id}...")
        n4_corrected_image = ants.n4_bias_field_correction(
            image,
            shrink_factor=1,
            convergence={'iters': [50, 40, 30], 'tol': 1e-6}, # Standard convergence criteria
            spline_param=200 # Common spline parameter
        )
        logger.info(f"N4 bias field correction completed for {subject_id}")
        return n4_corrected_image
        
    except Exception as e:
        logger.error(f"Error during N4 bias field correction for {subject_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback: return the original image if N4 correction fails
        return image

# --- Main processing function ---

def process_nifti_files():
    """
    Processes specified NIfTI files: performs registration and N4 bias correction,
    and saves them to the output directory.
    """
    if not BASE_DIR:
        logger.error("BASE_DIR is not set. Please edit the script to define BASE_DIR.")
        return

    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)
    logger.info(f"Processed output directory set to: {OUTPUT_PROCESSED_DIR}")

    if not os.path.exists(TEMPLATE_PATH):
        logger.error(f"MNI Template not found at: {TEMPLATE_PATH}. Please check the path and BASE_DIR.")
        return

    if not os.path.exists(INPUT_NIFTI_DIR):
        logger.error(f"Input NIfTI directory not found: {INPUT_NIFTI_DIR}. Please check the path and BASE_DIR.")
        return
        
    if not TARGET_NIFTI_FILES:
        logger.warning(f"No target NIfTI files specified in TARGET_NIFTI_FILES list.")
        return

    logger.info(f"Found {len(TARGET_NIFTI_FILES)} NIfTI files to process from the target list.")
    
    csv_data = []
    processed_count = 0 
    successful_processing_count = 0
    start_time_total = time.time()

    with tqdm(total=len(TARGET_NIFTI_FILES), desc="Processing NIfTI files") as pbar:
        for nifti_filename in TARGET_NIFTI_FILES:
            subject_id = nifti_filename.replace('.nii.gz', '')
            pbar.set_description(f"Processing {subject_id}")

            input_file_path = os.path.join(INPUT_NIFTI_DIR, nifti_filename)
            # Updated output filename convention
            output_processed_file_path = os.path.join(OUTPUT_PROCESSED_DIR, f"{subject_id}_registered_n4.nii.gz")
            current_status = 'pending'

            if not os.path.exists(input_file_path):
                logger.warning(f"Input file not found: {input_file_path}. Skipping.")
                current_status = 'skipped_missing_input'
                csv_data.append({
                    'subject_id': subject_id,
                    'original_nifti_path': input_file_path,
                    'processed_nifti_path': '', # Updated CSV column name
                    'status': current_status
                })
                processed_count += 1
                pbar.update(1)
                continue

            if os.path.exists(output_processed_file_path):
                logger.info(f"Processed file already exists: {output_processed_file_path}. Skipping.")
                current_status = 'skipped_exists'
                csv_data.append({
                    'subject_id': subject_id,
                    'original_nifti_path': input_file_path,
                    'processed_nifti_path': output_processed_file_path, # Updated CSV column name
                    'status': current_status
                })
                processed_count += 1
                pbar.update(1)
                continue
            
            # Step 1: Registration
            registered_image = register_to_mni_ants(input_file_path, TEMPLATE_PATH, subject_id)
            
            if registered_image is None:
                logger.error(f"Skipping N4 correction for {subject_id} due to registration failure.")
                current_status = 'registration_failed'
                csv_data.append({
                    'subject_id': subject_id,
                    'original_nifti_path': input_file_path,
                    'processed_nifti_path': '', # Updated CSV column name
                    'status': current_status
                })
                processed_count += 1
                pbar.update(1)
                continue

            # Step 2: N4 Bias Field Correction (on registered image)
            n4_corrected_image = apply_n4_bias_correction(registered_image, subject_id)
            
            if n4_corrected_image is None or n4_corrected_image == registered_image: 
                logger.warning(f"N4 bias correction might have failed or returned original registered image for {subject_id}.")
                # Decide if this is an error state or if we proceed with saving the registered_image
                # For now, let's assume we try to save what we have if n4_corrected_image is not None
                if n4_corrected_image is None: 
                    logger.error(f"Critical error: n4_corrected_image is None for {subject_id} after N4 correction step.")
                    current_status = 'n4_correction_critical_failure'
                    csv_data.append({
                        'subject_id': subject_id,
                        'original_nifti_path': input_file_path,
                        'processed_nifti_path': '', # Updated CSV column name
                        'status': current_status
                    })
                    processed_count +=1
                    pbar.update(1)
                    continue
            
            # Step 3: Save processed image
            try:
                ants.image_write(n4_corrected_image, output_processed_file_path)
                logger.info(f"Successfully saved processed image to {output_processed_file_path}")
                current_status = 'success'
                successful_processing_count += 1
            except Exception as e:
                logger.error(f"Failed to save processed image for {subject_id} to {output_processed_file_path}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                current_status = 'save_failed'
            
            csv_data.append({
                'subject_id': subject_id,
                'original_nifti_path': input_file_path,
                'processed_nifti_path': output_processed_file_path if current_status == 'success' else '', # Updated CSV column name
                'status': current_status
            })
            processed_count += 1
            
            elapsed_time_total = time.time() - start_time_total
            avg_time_per_file = elapsed_time_total / max(processed_count, 1)
            remaining_files = len(TARGET_NIFTI_FILES) - processed_count
            estimated_time_remaining = avg_time_per_file * remaining_files

            pbar.set_postfix({
                'Processed': f"{processed_count}/{len(TARGET_NIFTI_FILES)}",
                'ETA': f"{estimated_time_remaining/3600:.1f}h" if estimated_time_remaining > 0 else "0.0h",
                'Avg Time/file': f"{avg_time_per_file/60:.1f}min"
            })
            pbar.update(1)

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_output_path = os.path.join(OUTPUT_PROCESSED_DIR, CSV_FILENAME)
        try:
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Processing summary saved to: {csv_output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {str(e)}")
    else:
        logger.info("No data to save to CSV.")
        
    logger.info(f"\nProcessing completed:")
    logger.info(f"Total successfully processed and saved: {successful_processing_count}")
    logger.info(f"Total skipped (already exist or missing input): {sum(1 for item in csv_data if item['status'].startswith('skipped_'))}")
    total_failed = sum(1 for item in csv_data if item['status'] not in ['success'] and not item['status'].startswith('skipped_'))
    logger.info(f"Total failed (registration, N4 correction, or save): {total_failed}")
    logger.info(f"Total time: {(time.time() - start_time_total)/3600:.2f} hours")

if __name__ == "__main__":
    process_nifti_files()