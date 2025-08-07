import SimpleITK as sitk
import numpy as np
import os
import logging
from pathlib import Path

# --- Configuration ---
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer/"

BRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "ms-1-skull")  # Directory containing your brain images
BRAIN_MASKS_DIR = os.path.join(BASE_DIR, "ms-1-mask-full")  # Directory where we'll save the brain masks

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_brain_mask(input_image, output_mask):
    """
    Create a brain mask using simple intensity thresholding.
    """
    try:
        # Read the image
        img = sitk.ReadImage(input_image)
        
        # Convert to float for processing
        img_float = sitk.Cast(img, sitk.sitkFloat32)
        
        # Get image statistics
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img_float)
        min_val = stats.GetMinimum()
        max_val = stats.GetMaximum()
        
        # Calculate threshold as 10% of the maximum intensity
        threshold = min_val + 0.1 * (max_val - min_val)
        
        # Create binary mask
        mask = img_float > threshold
        
        # Save the mask
        sitk.WriteImage(mask, output_mask)
        logger.info(f"Successfully created brain mask: {output_mask}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating brain mask: {str(e)}")
        return False

def main():
    # Create output directory if it doesn't exist
    os.makedirs(BRAIN_MASKS_DIR, exist_ok=True)
    
    # Get all NIfTI files in the brain images directory
    brain_images = [f for f in os.listdir(BRAIN_IMAGES_DIR) 
                   if f.endswith(('.nii', '.nii.gz')) and not f.endswith(('_mask.nii.gz', '_Mask.nii.gz'))]
    
    if not brain_images:
        logger.error(f"No brain images found in {BRAIN_IMAGES_DIR}")
        return
        
    logger.info(f"Found {len(brain_images)} brain images to process")
    
    # Process each brain image
    for image_file in brain_images:
        input_path = os.path.join(BRAIN_IMAGES_DIR, image_file)
        
        # Create output filename by adding _BrainMask before the extension
        base_name = Path(image_file).stem
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_mask = os.path.join(BRAIN_MASKS_DIR, f"{base_name}_BrainMask.nii.gz")
        
        logger.info(f"\nProcessing: {image_file}")
        success = create_brain_mask(input_path, output_mask)
        
        if success:
            logger.info(f"Created brain mask: {output_mask}")
        else:
            logger.error(f"Failed to create brain mask for {image_file}")

if __name__ == "__main__":
    main()