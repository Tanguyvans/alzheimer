import os
import numpy as np
import pandas as pd
import dicom2nifti
import logging
import matplotlib.pyplot as plt
import ants
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from monai.transforms import (
    LoadImage,
    EnsureChannelFirstd,
    Orientationd,
)
import subprocess

import nilearn.datasets
import nilearn.image
import nilearn.plotting
import json
import nibabel as nib
from HD_BET.run import run_hd_bet

logging.basicConfig(
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    handlers=[
        logging.FileHandler("/home_nfs/amelo/projects/feder/cohort1/npy_bet_log.log", mode='a')],
                    level=logging.INFO)
import multiprocessing_logging

multiprocessing_logging.install_mp_handler()

logger = logging.getLogger('npy logger')
# Add BASE_DIR constant
BASE_DIR = "/shared/databases/alzheimer"


def check_existing_files(config, adni_id):
    """Check if files already exist for this ADNI ID."""
    existing = {
        'processed': os.path.exists(os.path.join(config['processed_dir'], f"{adni_id}.nii.gz")),
        'npy_skull': os.path.exists(os.path.join(config['npy_skullskip_seg_dir'], f"{adni_id}.nii.gz")),
        'skullskip_seg_dir': os.path.exists(os.path.join(BASE_DIR, "output/skull_seg", f"{adni_id}.nii.gz")) 
    }
    return existing

def get_adni_id_from_path(path):
    """Extract ADNI ID (I##### format) from path."""
    parts = str(path).split('/')
    for part in parts:
        if part.startswith('I') and part[1:].isdigit():
            return part
    return None

def convert_dicom_to_nifti(dicom_folder, output_folder, adni_id):
    """Convert a folder of DICOM files to NIfTI format."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{adni_id}.nii.gz")
        
        # Check if DICOM folder exists and contains .dcm files
        if not os.path.exists(dicom_folder):
            logger.error(f"DICOM folder does not exist: {dicom_folder}")
            return None
            
        dcm_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        if not dcm_files:
            logger.error(f"No DICOM files found in {dicom_folder}")
            return None
            
        # Convert DICOM to NIfTI
        dicom2nifti.convert_directory(dicom_folder, output_folder, compression=True, reorient=True)
        
        # Rename the output file if it exists but has a different name
        nifti_files = [f for f in os.listdir(output_folder) if f.endswith('.nii.gz')]
        if nifti_files:
            old_path = os.path.join(output_folder, nifti_files[0])
            os.rename(old_path, output_file)
            
        logger.info(f"Successfully converted {dicom_folder} to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error converting DICOM to NIfTI for {adni_id}: {str(e)}")
        return None



def skullremoval(nifti_path, output_file):
    command = [
        "hd-bet",
        "-i", nifti_path,
        "-o", output_file,
        "-mode", "accurate",
        "-tta", "0",
        "-s", "0",
        "-device", "2"
    ]
    
    try:
        subprocess.run(command, check=True)
        logger.info(f"Skull stripping completed for {nifti_path}, output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"HD-BET failed for {nifti_path}: {str(e)}")
        return None
    
    return output_file
    


def main():
    """Main pipeline for processing ADNI MRI data."""
    config = { 
        'output_dir': os.path.join(BASE_DIR, "output"),
        'skullskip_seg_dir': os.path.join(BASE_DIR, "output/skull_seg"),
        'csv_path': "/shared/databases/alzheimer/dataset_MRI_cohort1_AD_CN_updated.csv",
        'processed_dir': os.path.join(BASE_DIR, "output/processed_"),
        'npy_skullskip_seg_dir': os.path.join(BASE_DIR, "output/npy_seg_skullskip_AD_CN")
    }

    # Create output directories
    for dir_path in config.values():
        if isinstance(dir_path, str) and dir_path.startswith(config['output_dir']):
            os.makedirs(dir_path, exist_ok=True)

    # Read CSV file
    logger.info(f"Reading CSV file: {config['csv_path']}")
    df = pd.read_csv(config['csv_path'])
    
  
    npy_skullskip_seg_files = {}
    processed_nifti = {}

    for _, row in df.iterrows():
            input_path = row['mri_path']
            adni_id = get_adni_id_from_path(input_path)
            existing = check_existing_files(config, adni_id)

            
            if not adni_id:
                logger.warning(f"Could not find ADNI ID in {input_path}")
                continue
     
            # Step 1: DICOM to NIfTI (if needed)
            nifti_path = os.path.join(config['processed_dir'], f"{adni_id}.nii.gz")
            if not existing['processed']:
                logger.info(f"Converting DICOM to NIfTI for {adni_id}")
                nifti_file = convert_dicom_to_nifti(input_path, config['processed_dir'], adni_id)
                if nifti_file:
                    processed_nifti[adni_id] = nifti_file
            else:
                logger.info(f"NIfTI file already exists for {adni_id}")
                processed_nifti[adni_id] = nifti_path

                
            #Step 2: NIFTI to skullremoval 
            if not existing["skullskip_seg_dir"]:
                logger.info(f"Skull removal for {adni_id}")
                skullremoval_path = os.path.join(config['skullskip_seg_dir'], f"{adni_id}.nii.gz")
                skullremoval(nifti_path,skullremoval_path)                
                

            # Step 5: Convert to NPY (full brain)
            if not existing['npy_skull']:
                logger.info(f"Converting full brain to NPY for {adni_id}")
                try:
                    npy_path = os.path.join(config['npy_skullskip_seg_dir'], f"{adni_id}.npy")
                    image = LoadImage(dtype=np.float32, image_only=True)(skullremoval_path)
                    image_dict = {"image": image}
                    image_dict = EnsureChannelFirstd(keys="image")(image_dict)
                    image_dict = Orientationd(keys="image", axcodes="RAS")(image_dict)
                    np.save(npy_path, image_dict["image"])
                    npy_skullskip_seg_files[adni_id] = npy_path
                except Exception as e:
                    logger.error(f"NPY conversion failed for {adni_id}: {str(e)}")
                    continue

         

   
    df["seg_skullskip_npy_path"] = df['adni_id'].apply(lambda x: npy_skullskip_seg_files.get(x, ''))
    df["processed_path"] = df['adni_id'].apply(lambda x: processed_nifti.get(x, '')) 
    
    # Save updated CSV
    updated_csv_path = os.path.join(config['root_dir'], "dataset_MRI_cohort1_ALL__updated.csv")
    df.to_csv(updated_csv_path, index=False)
    logger.info(f"Updated CSV saved to {updated_csv_path}")

if __name__ == "__main__":
    main()
    
    
    
