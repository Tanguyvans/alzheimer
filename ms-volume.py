import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from collections import defaultdict

# --- Configuration ---
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer/"  # Adjust if your script is elsewhere
MASKS_INPUT_DIR = os.path.join(BASE_DIR, "ms-1-mask")
BRAIN_MASKS_DIR = os.path.join(BASE_DIR, "ms-1-mask-full")
OUTPUT_CSV_FILE = os.path.join(BASE_DIR, "ms_lesion_volumes.csv")

# --- Logging Setup (Optional, but good practice) ---
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Volume Calculation Functions ---

def calculate_volume_from_mask(nifti_path):
    """
    Calculate volume from a binary mask in cubic millimeters.
    """
    try:
        img = sitk.ReadImage(nifti_path)
        arr = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        voxel_volume_mm3 = np.prod(spacing)
        
        if not np.any(arr):
            logger.warning(f"Mask is empty (all zeros): {os.path.basename(nifti_path)}")
            return 0.0
            
        voxel_count = np.count_nonzero(arr)
        volume = voxel_count * voxel_volume_mm3
        return volume
        
    except Exception as e:
        logger.error(f"Error calculating volume for {nifti_path}: {str(e)}")
        return None

def calculate_volumes_from_nifti(nifti_path, is_labeled_clusters=False):
    """
    Calculates volumes from a NIfTI mask file using SimpleITK.
    If is_labeled_clusters is True, it calculates volume for each label.
    Otherwise, it calculates total volume for non-zero voxels (treating it as a binary mask).
    Volumes are returned in cubic millimeters (mm^3).

    Args:
        nifti_path (str): Path to the NIfTI mask file.
        is_labeled_clusters (bool): True if the mask contains multiple distinct labeled clusters.

    Returns:
        dict: 
            If is_labeled_clusters: {label_id (int): volume_mm3 (float), ...}
            Else: {'total_lesion_volume': volume_mm3 (float)}
        None if an error occurs.
    """
    try:
        img = sitk.ReadImage(nifti_path)
        arr = sitk.GetArrayFromImage(img)  # z, y, x
        spacing = img.GetSpacing()         # dx, dy, dz
        
        voxel_volume_mm3 = np.prod(spacing)
        
        volumes_mm3 = {}
        
        if not np.any(arr): # Check if the mask is all zeros
            logger.warning(f"Mask is empty (all zeros): {os.path.basename(nifti_path)}")
            if is_labeled_clusters:
                return {} # No labels to report
            else:
                return {'total_lesion_volume': 0.0}


        if is_labeled_clusters:
            unique_labels = np.unique(arr)
            logger.info(f"Processing LabeledClusters: {os.path.basename(nifti_path)} - Unique labels: {unique_labels}")
            for label_id in unique_labels:
                if label_id == 0:  # Skip background
                    continue
                voxel_count = np.sum(arr == label_id)
                volume = voxel_count * voxel_volume_mm3
                volumes_mm3[int(label_id)] = volume
                logger.debug(f"  Label {int(label_id)}: Count={voxel_count}, Volume={volume:.2f} mm^3")
        else: # Treat as a binary lesion mask
            logger.info(f"Processing LesionMask: {os.path.basename(nifti_path)}")
            lesion_labels = np.unique(arr[arr != 0])
            if not lesion_labels.any(): 
                 logger.warning(f"LesionMask is effectively empty (only zero-value voxels): {os.path.basename(nifti_path)}")
                 volumes_mm3['total_lesion_volume'] = 0.0
            elif len(lesion_labels) > 1 and np.array_equal(lesion_labels, np.array([1])) == False : # check if it's not just [1]
                logger.warning(f"LesionMask {os.path.basename(nifti_path)} has multiple non-zero labels: {lesion_labels} or a single label not equal to 1. Calculating total volume for all non-zero.")
                voxel_count = np.count_nonzero(arr)
                volume = voxel_count * voxel_volume_mm3
                volumes_mm3['total_lesion_volume'] = volume
                logger.debug(f"  Total Lesion: Count={voxel_count}, Volume={volume:.2f} mm^3 (sum of all non-zero labels)")
            else: # Single non-zero label (usually 1), or only one type of lesion
                voxel_count = np.count_nonzero(arr) 
                volume = voxel_count * voxel_volume_mm3
                volumes_mm3['total_lesion_volume'] = volume
                logger.debug(f"  Total Lesion: Count={voxel_count}, Volume={volume:.2f} mm^3")

        return volumes_mm3

    except FileNotFoundError:
        logger.error(f"Mask file not found: {nifti_path}")
        return None
    except RuntimeError as e: 
        logger.error(f"SimpleITK error processing file {nifti_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Generic error processing file {nifti_path}: {str(e)}")
        return None

# --- Main Script Logic ---
def main():
    logger.info(f"Starting lesion volume calculation from: {MASKS_INPUT_DIR}")
    logger.info(f"Output will be saved to: {OUTPUT_CSV_FILE}")

    if not os.path.exists(MASKS_INPUT_DIR):
        logger.error(f"Input directory not found: {MASKS_INPUT_DIR}")
        return

    all_files = os.listdir(MASKS_INPUT_DIR)
    nifti_files = sorted([f for f in all_files if f.endswith(('.nii', '.nii.gz'))])

    if not nifti_files:
        logger.warning(f"No NIfTI files found in {MASKS_INPUT_DIR}")
        return

    results_data = []

    for filename in nifti_files:
        file_path = os.path.join(MASKS_INPUT_DIR, filename)
        
        is_labeled = "_LabeledClusters.nii.gz" in filename
        is_lesion_mask = "_LesionMask.nii.gz" in filename
        
        if not (is_labeled or is_lesion_mask):
            continue
            
        common_prefix = ""
        if is_labeled:
            common_prefix = filename.replace("_LabeledClusters.nii.gz", "")
        elif is_lesion_mask:
            common_prefix = filename.replace("_LesionMask.nii.gz", "")
            
        logger.info(f"\nProcessing: {filename} (Prefix: {common_prefix})")
        
        # Get brain volume
        brain_mask_path = os.path.join(BRAIN_MASKS_DIR, f"{common_prefix}_BrainMask.nii.gz")
        brain_volume = calculate_volume_from_mask(brain_mask_path)
        
        if brain_volume is None:
            logger.error(f"Could not calculate brain volume for {common_prefix}")
            continue
            
        volumes = calculate_volumes_from_nifti(file_path, is_labeled_clusters=is_labeled)

        if volumes is None:
            results_data.append({
                'subject_scan_id': common_prefix,
                'mask_type': "LabeledClusters" if is_labeled else "LesionMask",
                'label_id': 'ERROR',
                'volume_mm3': 'ERROR',
                'brain_volume_mm3': brain_volume,
                'proportion_of_brain': 'ERROR'
            })
            continue

        if is_labeled:
            current_cluster_sum = 0.0
            if not volumes:
                results_data.append({
                    'subject_scan_id': common_prefix,
                    'mask_type': "LabeledClusters",
                    'label_id': 'NoLesions',
                    'volume_mm3': 0.0,
                    'brain_volume_mm3': brain_volume,
                    'proportion_of_brain': 0.0
                })
            else:
                for label_id, vol_mm3 in volumes.items():
                    proportion = (vol_mm3 / brain_volume * 100)
                    results_data.append({
                        'subject_scan_id': common_prefix,
                        'mask_type': "LabeledClusters",
                        'label_id': label_id,
                        'volume_mm3': vol_mm3,
                        'brain_volume_mm3': brain_volume,
                        'proportion_of_brain': proportion
                    })
                    current_cluster_sum += vol_mm3

            # Add the sum row for LabeledClusters
            total_proportion = (current_cluster_sum / brain_volume * 100)
            results_data.append({
                'subject_scan_id': common_prefix,
                'mask_type': "LabeledClusters",
                'label_id': 'SumOfLabels',
                'volume_mm3': current_cluster_sum,
                'brain_volume_mm3': brain_volume,
                'proportion_of_brain': total_proportion
            })

        elif is_lesion_mask:
            total_vol = volumes.get('total_lesion_volume', 0.0)
            proportion = (total_vol / brain_volume * 100)
            results_data.append({
                'subject_scan_id': common_prefix,
                'mask_type': "LesionMask",
                'label_id': 'Total',
                'volume_mm3': total_vol,
                'brain_volume_mm3': brain_volume,
                'proportion_of_brain': proportion
            })

    if not results_data:
        logger.info("No data processed to save to CSV.")
        return

    df = pd.DataFrame(results_data)
    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logger.info(f"Successfully saved lesion volumes to {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")

if __name__ == "__main__":
    main()