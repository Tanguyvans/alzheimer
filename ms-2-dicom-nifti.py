import os
import SimpleITK as sitk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = "ms-2/DICOM"
OUTPUT_DIR = "ms-2-nifti"
OUTPUT_FILENAME = "ms-2.nii.gz"

def convert_dicom_folder_to_nifti(input_dir, output_dir, output_filename_base):
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info("Creating SimpleITK reader...")
        reader = sitk.ImageSeriesReader()
        
        logger.info(f"Scanning DICOM series in {input_dir}...")
        series_IDs = reader.GetGDCMSeriesIDs(input_dir)
        if not series_IDs:
            logger.error(f"No DICOM series found in {input_dir}")
            return

        logger.info(f"Found {len(series_IDs)} series IDs: {series_IDs}")

        for i, series_id in enumerate(series_IDs):
            output_filename = f"{os.path.splitext(os.path.splitext(output_filename_base)[0])[0]}_series_{i+1}.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            logger.info(f"Processing series {i+1}/{len(series_IDs)}: {series_id}")
            
            try:
                series_file_names = reader.GetGDCMSeriesFileNames(input_dir, series_id)
                if not series_file_names:
                    logger.warning(f"No files found for series ID: {series_id}. Skipping.")
                    continue
                
                logger.info(f"Found {len(series_file_names)} files for series {series_id}")
                logger.info(f"First few files: {series_file_names[:3]}")
                reader.SetFileNames(series_file_names)
        
                logger.info(f"Executing reader for series {series_id}...")
        image = reader.Execute()
        
        logger.info(f"Image size: {image.GetSize()}")
        logger.info(f"Image spacing: {image.GetSpacing()}")
        
        logger.info("Permuting axes...")
                image = sitk.PermuteAxes(image, [2, 1, 0]) # Adjust permutation as needed, e.g., [0,1,2] or [2,0,1] etc.
        
                logger.info(f"Writing series {series_id} to {output_path}...")
        sitk.WriteImage(image, output_path)
        
        if os.path.exists(output_path):
                    logger.info(f"Successfully converted series {series_id} to {output_path}")
            logger.info(f"Output file size: {os.path.getsize(output_path)} bytes")
        else:
                    logger.error(f"File was not created for series {series_id}: {output_path}")
            except Exception as e:
                logger.error(f"Error converting series {series_id} to NIfTI: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"General error in DICOM to NIfTI conversion process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    convert_dicom_folder_to_nifti(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)