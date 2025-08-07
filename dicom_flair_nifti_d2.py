import os
import pandas as pd
import dicom2nifti
import logging
from tqdm import tqdm
import warnings

# Disable all warnings from dicom2nifti and other libraries
warnings.filterwarnings('ignore')

# Configuration du logging
# THIS MUST BE DONE BEFORE ANY LOGGER CALLS
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Attempt to make dicom2nifti less strict
try:
    logger.info("Attempting to apply relaxed validation settings for dicom2nifti...")
    dicom2nifti.settings.disable_validate_orientation()
    dicom2nifti.settings.disable_validate_slice_increment()
    dicom2nifti.settings.disable_validate_slice_positions()
    # dicom2nifti.settings.disable_validate_slice_thickness() # Keep this commented for now unless specifically needed
    
    # Set resampling options - order 1 is linear, 0 is nearest neighbor
    dicom2nifti.settings.set_resample_spline_interpolation_order(1) 
    dicom2nifti.settings.set_resample_padding(-1000) 
    logger.info("Applied relaxed validation and resampling settings.")

except AttributeError as e:
    logger.warning(f"Could not apply all dicom2nifti settings. Error: {e}. This might be due to the version of dicom2nifti.")
except Exception as e:
    logger.warning(f"An unexpected error occurred while applying dicom2nifti settings: {e}")

# Define base directory
BASE_DIR = ""
INPUT_ROOT_DIR = os.path.join(BASE_DIR, "irm_sep_etude")
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "irm2_output")
CSV_FILENAME = "conversion_summary.csv"

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

    with tqdm(total=sum(len(os.listdir(os.path.join(INPUT_ROOT_DIR, p_dir))) for p_dir in patient_dirs if os.path.isdir(os.path.join(INPUT_ROOT_DIR, p_dir))), desc="Processing Flair series") as pbar:
        for patient_id in patient_dirs:
            patient_input_path = os.path.join(INPUT_ROOT_DIR, patient_id)
            
            try:
                flair_dirs = [d for d in os.listdir(patient_input_path) if os.path.isdir(os.path.join(patient_input_path, d))]
            except FileNotFoundError:
                logger.warning(f"Patient directory not found or inaccessible: {patient_input_path}. Skipping.")
                # Update pbar for potentially uncounted items if this was the only way to count
                # This is tricky as total is precalculated. For now, errors will just mean pbar finishes early.
                continue


            for flair_id in flair_dirs:
                pbar.set_description(f"Processing {patient_id}/{flair_id}")
                dicom_folder = os.path.join(patient_input_path, flair_id)
                
                # Check if the dicom_folder actually contains .dcm files (case-insensitive)
                try:
                    if not any(f.lower().endswith('.dcm') for f in os.listdir(dicom_folder)):
                        logger.warning(f"No .dcm files found in {dicom_folder}. Skipping.")
                        pbar.update(1)
                        continue
                except FileNotFoundError:
                    logger.warning(f"Flair directory not found: {dicom_folder}. Skipping.")
                    pbar.update(1)
                    continue
                except Exception as e:
                    logger.error(f"Error listing files in {dicom_folder}: {str(e)}. Skipping.")
                    pbar.update(1)
                    continue


                output_nifti_filename = f"{patient_id}_{flair_id}.nii.gz"
                output_nifti_path = os.path.join(OUTPUT_ROOT_DIR, output_nifti_filename)

                if os.path.exists(output_nifti_path):
                    logger.info(f"NIfTI file already exists: {output_nifti_path}. Skipping.")
                    # Add to CSV data even if skipped, assuming it was processed correctly before
                    csv_data.append({
                        'patient_id': patient_id,
                        'flair_id': flair_id,
                        'nifti_path': output_nifti_path
                    })
                    pbar.update(1)
                    continue

                try:
                    logger.info(f"Converting {dicom_folder} to {output_nifti_path}")
                    dicom2nifti.dicom_series_to_nifti(dicom_folder, output_nifti_path, reorient_nifti=True)
                    
                    if os.path.exists(output_nifti_path):
                        logger.info(f"Successfully converted to {output_nifti_path}")
                        csv_data.append({
                            'patient_id': patient_id,
                            'flair_id': flair_id,
                            'nifti_path': output_nifti_path
                        })
                    else:
                        # This case should ideally not happen if dicom_series_to_nifti doesn't error
                        logger.error(f"Conversion reported success but output file not found: {output_nifti_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to convert {dicom_folder}: {str(e)}")
                finally:
                    pbar.update(1)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_output_path = os.path.join(OUTPUT_ROOT_DIR, CSV_FILENAME)
        try:
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Conversion summary saved to: {csv_output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {str(e)}")
    else:
        logger.info("No data to save to CSV.")

    logger.info("Processing complete.")

if __name__ == "__main__":
    convert_patient_flair_to_nifti() 