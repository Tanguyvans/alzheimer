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

# Define base directory and other paths
# IMPORTANT: Set your BASE_DIR here
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"  # e.g., "/Users/yourusername/projects/my_project"
INPUT_NIFTI_DIR = os.path.join(BASE_DIR, "irm_output")
OUTPUT_ENHANCED_DIR = os.path.join(BASE_DIR, "irm_enhanced_output")
# IMPORTANT: Ensure this template path is correct
TEMPLATE_PATH = os.path.join(BASE_DIR, "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii")
CSV_FILENAME = "enhancement_summary.csv"

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
            type_of_transform='SyN'
        )
        registered_image = registration['warpedmovout']
        logger.info(f"Registration successful for {subject_id}")
        return registered_image
    except Exception as e:
        logger.error(f"Registration failed for {subject_id} from {input_file_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def enhance_registered_image(image, subject_id):
    """
    Enhances the quality of a registered ANTs image by applying N4 bias field correction
    and then intensity normalization.
    """
    logger.info(f"Starting N4 correction and normalization for registered image {subject_id}...")
    try:
        if image is None:
            logger.error(f"Input image for enhancement is None for {subject_id}")
            return None

        # 1. N4 Bias Field Correction (on the registered image)
        logger.info(f"Performing N4 bias field correction for {subject_id}...")
        n4_image = ants.n4_bias_field_correction(
            image,
            shrink_factor=2,
            convergence={'iters': [50, 40, 30], 'tol': 1e-6},
            spline_param=200 # A common spline parameter value
        )
        logger.info(f"N4 correction completed for {subject_id}")

        # 2. Normalization (on the N4 corrected, registered image)
        logger.info(f"Performing normalization for {subject_id}...")
        normalized_image = ants.iMath(n4_image, "Normalize") # Normalizes intensities to [0,1]
        logger.info(f"Normalization completed for {subject_id}")
        
        logger.info(f"N4 correction and normalization completed for {subject_id}")
        return normalized_image
        
    except Exception as e:
        logger.error(f"Error during N4 correction or normalization for {subject_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback: return the original registered image if enhancement fails
        return image

# --- Main processing function ---

def process_nifti_files():
    """
    Processes NIfTI files from input_dir, performs registration and enhancement,
    and saves them to output_dir.
    """
    if not BASE_DIR:
        logger.error("BASE_DIR is not set. Please edit the script to define BASE_DIR.")
        return

    os.makedirs(OUTPUT_ENHANCED_DIR, exist_ok=True)
    logger.info(f"Enhanced output directory set to: {OUTPUT_ENHANCED_DIR}")

    if not os.path.exists(TEMPLATE_PATH):
        logger.error(f"MNI Template not found at: {TEMPLATE_PATH}. Please check the path and BASE_DIR.")
        return

    try:
        nifti_files = [f for f in os.listdir(INPUT_NIFTI_DIR) if f.endswith('.nii.gz')]
    except FileNotFoundError:
        logger.error(f"Input NIfTI directory not found: {INPUT_NIFTI_DIR}. Please check the path and BASE_DIR.")
        return

    if not nifti_files:
        logger.warning(f"No .nii.gz files found in {INPUT_NIFTI_DIR}")
        return

    logger.info(f"Found {len(nifti_files)} NIfTI files to process.")
    
    csv_data = []
    processed_count = 0 
    successful_enhancements = 0
    start_time_total = time.time()

    with tqdm(total=len(nifti_files), desc="Processing NIfTI files") as pbar:
        for nifti_filename in nifti_files:
            subject_id = nifti_filename.replace('.nii.gz', '')
            pbar.set_description(f"Processing {subject_id}")

            input_file_path = os.path.join(INPUT_NIFTI_DIR, nifti_filename)
            output_enhanced_file_path = os.path.join(OUTPUT_ENHANCED_DIR, f"{subject_id}_enhanced.nii.gz")
            current_status = 'pending'

            if os.path.exists(output_enhanced_file_path):
                logger.info(f"Enhanced file already exists: {output_enhanced_file_path}. Skipping.")
                current_status = 'skipped_exists'
                csv_data.append({
                    'subject_id': subject_id,
                    'original_nifti_path': input_file_path,
                    'enhanced_nifti_path': output_enhanced_file_path,
                    'status': current_status
                })
                processed_count += 1
                pbar.update(1)
                continue
            
            # Step 1: Registration
            registered_image = register_to_mni_ants(input_file_path, TEMPLATE_PATH, subject_id)
            
            if registered_image is None:
                logger.error(f"Skipping enhancement for {subject_id} due to registration failure.")
                current_status = 'registration_failed'
                csv_data.append({
                    'subject_id': subject_id,
                    'original_nifti_path': input_file_path,
                    'enhanced_nifti_path': '',
                    'status': current_status
                })
                processed_count += 1
                pbar.update(1)
                continue

            # Step 2: Enhancement (N4 Correction and Normalization on registered image)
            final_enhanced_image = enhance_registered_image(registered_image, subject_id)
            
            if final_enhanced_image is None or final_enhanced_image == registered_image: # Check if enhancement effectively failed
                logger.warning(f"Enhancement (N4/Normalize) might have failed or returned original registered image for {subject_id}.")
                # Decide if this is an error state or if we proceed with saving the registered_image
                # For now, let's assume we try to save what we have if final_enhanced_image is not None
                if final_enhanced_image is None: # Should not happen if registered_image was not None
                    logger.error(f"Critical error: final_enhanced_image is None for {subject_id} after enhancement step.")
                    current_status = 'enhancement_critical_failure'
                    csv_data.append({
                        'subject_id': subject_id,
                        'original_nifti_path': input_file_path,
                        'enhanced_nifti_path': '',
                        'status': current_status
                    })
                    processed_count +=1
                    pbar.update(1)
                    continue
            
            # Step 3: Save enhanced image
            try:
                ants.image_write(final_enhanced_image, output_enhanced_file_path)
                logger.info(f"Successfully saved enhanced image to {output_enhanced_file_path}")
                current_status = 'success'
                successful_enhancements += 1
            except Exception as e:
                logger.error(f"Failed to save enhanced image for {subject_id} to {output_enhanced_file_path}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                current_status = 'save_failed'
            
            csv_data.append({
                'subject_id': subject_id,
                'original_nifti_path': input_file_path,
                'enhanced_nifti_path': output_enhanced_file_path if current_status == 'success' else '',
                'status': current_status
            })
            processed_count += 1
            
            elapsed_time_total = time.time() - start_time_total
            avg_time_per_file = elapsed_time_total / max(processed_count, 1)
            remaining_files = len(nifti_files) - processed_count
            estimated_time_remaining = avg_time_per_file * remaining_files

            pbar.set_postfix({
                'Processed': f"{processed_count}/{len(nifti_files)}",
                'ETA': f"{estimated_time_remaining/3600:.1f}h" if estimated_time_remaining > 0 else "0.0h",
                'Avg Time/file': f"{avg_time_per_file/60:.1f}min"
            })
            pbar.update(1)

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_output_path = os.path.join(OUTPUT_ENHANCED_DIR, CSV_FILENAME)
        try:
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Enhancement summary saved to: {csv_output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {str(e)}")
    else:
        logger.info("No data to save to CSV.")
        
    logger.info(f"\nProcessing completed:")
    logger.info(f"Total successfully processed and saved: {successful_enhancements}")
    logger.info(f"Total skipped (already exist): {sum(1 for item in csv_data if item['status'] == 'skipped_exists')}")
    total_failed = sum(1 for item in csv_data if item['status'] not in ['success', 'skipped_exists'])
    logger.info(f"Total failed (registration, enhancement, or save): {total_failed}")
    logger.info(f"Total time: {(time.time() - start_time_total)/3600:.2f} hours")

if __name__ == "__main__":
    process_nifti_files() 