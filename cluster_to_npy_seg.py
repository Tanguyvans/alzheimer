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

import nilearn.datasets
import nilearn.image
import nilearn.plotting
import json
import nibabel as nib

# Add logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add BASE_DIR constant
BASE_DIR = "../../../shared/databases/alzheimer"

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

def convert_adni_to_nifti(adni_root, nifti_root):
    """Convert all DICOM folders in the ADNI directory to NIfTI format."""
    processed_files = {}
    
    for root, dirs, files in os.walk(adni_root):
        if files and any(file.endswith('.dcm') for file in files):
            # This is a DICOM folder
            adni_id = get_adni_id_from_path(root)
            if not adni_id:
                logger.warning(f"Could not find ADNI ID in {root}")
                continue
                
            output_folder = os.path.join(nifti_root, adni_id)
            nifti_file = convert_dicom_to_nifti(root, output_folder, adni_id)
            logger.info(f"Converted {root} to {nifti_file}")
            
            processed_files[adni_id] = {
                'original_path': root,
                'nifti_path': nifti_file
            }
    
    return processed_files

def compare_registration(original_file, registered_file, output_dir, adni_id, plot=True):
    """Compare original and registered images."""
    logging.info(f"Comparing registration for {adni_id}")
    
    # Load images
    original_img = ants.image_read(original_file)
    registered_img = ants.image_read(registered_file)
    
    # Convert to numpy arrays
    original_array = original_img.numpy()
    registered_array = registered_img.numpy()
    
    # Check dimensions
    if original_array.shape != registered_array.shape:
        logging.warning(f"Image dimensions don't match for {adni_id}. Resampling...")
        original_array = ants.resample_image_to_target(original_img, registered_img).numpy()
    
    # Calculate metrics
    ssim_value = ssim(original_array, registered_array, 
                      data_range=registered_array.max() - registered_array.min())
    mse_value = mean_squared_error(original_array, registered_array)
    
    if plot:
        # Create visualization directory
        vis_dir = os.path.join(output_dir, 'visualization', adni_id)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot middle sagittal slice
        mid_slice = original_array.shape[0] // 2
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(original_array[mid_slice, :, :], cmap='gray', aspect='auto')
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(registered_array[mid_slice, :, :], cmap='gray', aspect='auto')
        ax2.set_title('Registered')
        ax2.axis('off')
        
        diff_image = registered_array - original_array
        vmax = np.abs(diff_image).max()
        ax3.imshow(diff_image[mid_slice, :, :], cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        ax3.set_title('Difference')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'registration_comparison.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    # Save metrics
    metrics_file = os.path.join(vis_dir, 'registration_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"SSIM: {ssim_value}\n")
        f.write(f"MSE: {mse_value}\n")
    
    return {'ssim': ssim_value, 'mse': mse_value}

def register_to_mni_ants(input_file, output_dir, mni_template, adni_id):
    """Register image to MNI space using ANTs."""
    logging.info(f"Starting registration for {adni_id}")
    
    # Load images
    moving_image = ants.image_read(input_file)
    fixed_image = ants.image_read(mni_template)
    
    # Perform registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='SyN'
    )
    
    # Save registered image
    output_file = os.path.join(output_dir, f"{adni_id}.nii.gz")
    ants.image_write(registration['warpedmovout'], output_file)
    
    logging.info(f"Registration completed for {adni_id}")
    return output_file

def convert_nifti_to_npy(nifti_dir, npy_dir):
    """Convert registered NIfTI files to NPY format."""
    os.makedirs(npy_dir, exist_ok=True)
    logger.info(f"Converting files from {nifti_dir} to {npy_dir}")
    
    # Initialize MONAI transforms
    loader = LoadImage(dtype=np.float32, image_only=True)
    ensure_channel_first = EnsureChannelFirstd(keys="image")
    orientation_transform = Orientationd(keys="image", axcodes="RAS")
    
    processed_files = {}
    
    # Process each registered NIfTI file
    for nifti_file in os.listdir(nifti_dir):
        if nifti_file.endswith('.nii.gz'):
            try:
                adni_id = get_adni_id_from_path(nifti_file)
                if not adni_id:
                    logger.warning(f"Could not find ADNI ID in {nifti_file}")
                    continue
                
                nifti_path = os.path.join(nifti_dir, nifti_file)
                
                # Load and transform the image
                image = loader(nifti_path)
                image_dict = {"image": image}
                image_dict = ensure_channel_first(image_dict)
                image_dict = orientation_transform(image_dict)
                
                # Save as NPY
                npy_path = os.path.join(npy_dir, f"{adni_id}.npy")
                np.save(npy_path, image_dict["image"])
                
                processed_files[adni_id] = {
                    'nifti_path': nifti_path,
                    'npy_path': npy_path
                }
                
                logger.info(f"Converted {nifti_file} to {adni_id}.npy")
                
            except Exception as e:
                logger.error(f"Error converting {nifti_file}: {str(e)}")
                continue
    
    return processed_files

def check_existing_files(config, adni_id):
    """Check if files already exist for this ADNI ID."""
    existing = {
        'processed': os.path.exists(os.path.join(config['processed_dir'], f"{adni_id}.nii.gz")),
        'registered': os.path.exists(os.path.join(config['registered_dir'], f"{adni_id}.nii.gz")),
        'npy': os.path.exists(os.path.join(config['npy_dir'], f"{adni_id}.npy"))
    }
    return existing

def resize_to_mni_resolution(input_path, output_path):
    """Resize image to 128xYx128 resolution (preserving Y dimension)."""
    logger.info("Resizing to 128xYx128 resolution...")
    
    input_img = ants.image_read(input_path)
    input_shape = input_img.shape
    
    print(f"Original image shape: {input_shape}")
    target_dims = (128, input_shape[1], 128)
    
    resized_img = ants.resample_image(
        input_img,
        target_dims,
        use_voxels=True,
        interp_type=1
    )
    
    ants.image_write(resized_img, output_path)
    logger.info(f"Resized image saved to: {output_path}")
    return output_path

def segment_hippocampus(input_image_path, output_folder):
    """Segment the hippocampus using the AAL atlas."""
    print("\n=== Starting Hippocampus Segmentation ===")
    os.makedirs(output_folder, exist_ok=True)
    
    input_img = nib.load(input_image_path)
    
    # Load AAL atlas
    aal = nilearn.datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nilearn.image.load_img(aal.maps)
    
    # Resize AAL atlas to match input
    aal_resized = nilearn.image.resample_img(
        aal_img,
        target_affine=input_img.affine,
        target_shape=input_img.shape,
        interpolation='nearest'
    )
    
    # Define hippocampus regions
    HIPPOCAMPUS_L = 4101
    HIPPOCAMPUS_R = 4102
    
    # Create masks
    aal_data = aal_resized.get_fdata()
    mask_left = (aal_data == HIPPOCAMPUS_L).astype(np.float32)
    mask_right = (aal_data == HIPPOCAMPUS_R).astype(np.float32)
    
    # Save masks and segmented hippocampus
    left_hippo_path = os.path.join(output_folder, 'hippocampus_left_segmented.nii.gz')
    right_hippo_path = os.path.join(output_folder, 'hippocampus_right_segmented.nii.gz')
    
    nib.save(nib.Nifti1Image(mask_left * input_img.get_fdata(), input_img.affine), left_hippo_path)
    nib.save(nib.Nifti1Image(mask_right * input_img.get_fdata(), input_img.affine), right_hippo_path)
    
    return left_hippo_path, right_hippo_path

def extract_hippocampus_slices(left_hippo_path, right_hippo_path, input_path, output_dir):
    """Extract a 3D volume containing 50 coronal slices centered on hippocampus."""
    print("\n=== Starting Hippocampus Slice Extraction ===")
    os.makedirs(output_dir, exist_ok=True)
    
    input_img = nib.load(input_path)
    left_hippo = nib.load(left_hippo_path).get_fdata()
    right_hippo = nib.load(right_hippo_path).get_fdata()
    
    # Get dimensions
    _, n_coronal, _ = left_hippo.shape
    
    # Find slices containing hippocampus
    left_slices = set([i for i in range(n_coronal) if np.any(left_hippo[:, i, :])])
    right_slices = set([i for i in range(n_coronal) if np.any(right_hippo[:, i, :])])
    hippo_slices = sorted(list(left_slices.union(right_slices)))
    
    # Calculate center and extract 50 slices
    center_slice = hippo_slices[len(hippo_slices)//2]
    total_slices_needed = 50
    start_slice = max(0, center_slice - (total_slices_needed//2))
    
    # Create final slice list
    final_slices = list(range(start_slice, min(start_slice + total_slices_needed, n_coronal)))
    
    # Extract relevant slices
    image_extracted = input_img.get_fdata()[:, final_slices, :]
    
    # Save as new NIfTI
    new_affine = input_img.affine.copy()
    new_affine[1, 3] = new_affine[1, 3] + (final_slices[0] * new_affine[1, 1])
    
    extracted_nifti = nib.Nifti1Image(image_extracted, new_affine, input_img.header)
    output_path = os.path.join(output_dir, 'extracted_volume.nii.gz')
    nib.save(extracted_nifti, output_path)
    
    return output_path

def main():
    """Main pipeline for processing ADNI MRI data."""
    config = {
        'root_dir': BASE_DIR,
        'input_dir': os.path.join(BASE_DIR),
        'output_dir': os.path.join(BASE_DIR, "output"),
        'processed_dir': os.path.join(BASE_DIR, "output/processed"),
        'registered_dir': os.path.join(BASE_DIR, "output/registered"),
        'resized_dir': os.path.join(BASE_DIR, "output/resized"),
        'hippo_dir': os.path.join(BASE_DIR, "output/hippocampus"),
        'visualization_dir': os.path.join(BASE_DIR, "output/visualization"),
        'npy_dir': os.path.join(BASE_DIR, "output/npy"),
        'npy_seg_dir': os.path.join(BASE_DIR, "output/npy_seg"),
        'template_path': os.path.join(BASE_DIR, "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"),
        'csv_path': os.path.join(BASE_DIR, "dataset_MRI_cohort1.csv")
    }

    # Create output directories
    for dir_path in config.values():
        if isinstance(dir_path, str) and dir_path.startswith(config['output_dir']):
            os.makedirs(dir_path, exist_ok=True)

    # Read CSV file
    logger.info(f"Reading CSV file: {config['csv_path']}")
    df = pd.read_csv(config['csv_path'])
    
    processed_nifti = {}
    registered_files = {}
    npy_files = {}

    for _, row in df.iterrows():
        try:
            input_path = os.path.join(config['input_dir'], row['mri_path'].strip('/'))
            adni_id = get_adni_id_from_path(input_path)
            
            if not adni_id:
                logger.warning(f"Could not find ADNI ID in {input_path}")
                continue

            # Check existing files
            existing = check_existing_files(config, adni_id)
            
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

            # Step 2: Registration (if needed)
            registered_path = os.path.join(config['registered_dir'], f"{adni_id}.nii.gz")
            if not existing['registered'] and adni_id in processed_nifti:
                logger.info(f"Performing registration for {adni_id}")
                try:
                    registered_path = register_to_mni_ants(
                        processed_nifti[adni_id],
                        config['registered_dir'],
                        config['template_path'],
                        adni_id
                    )
                    
                    metrics = compare_registration(
                        processed_nifti[adni_id],
                        registered_path,
                        config['output_dir'],
                        adni_id,
                        plot=True
                    )
                    
                    registered_files[adni_id] = {
                        'path': registered_path,
                        'metrics': metrics
                    }
                except Exception as e:
                    logger.error(f"Registration failed for {adni_id}: {str(e)}")
                    continue
            elif existing['registered']:
                logger.info(f"Registered file already exists for {adni_id}")
                registered_files[adni_id] = {'path': registered_path}

            # Step 3: Resize
            resized_path = os.path.join(config['resized_dir'], f"{adni_id}.nii.gz")
            if not os.path.exists(resized_path):
                logger.info(f"Resizing image for {adni_id}")
                resized_path = resize_to_mni_resolution(registered_path, resized_path)

            # Step 4: Hippocampus segmentation
            hippo_dir = os.path.join(config['hippo_dir'], adni_id)
            extracted_path = os.path.join(hippo_dir, 'extracted_volume.nii.gz')
            if not os.path.exists(extracted_path):
                logger.info(f"Processing hippocampus for {adni_id}")
                left_hippo, right_hippo = segment_hippocampus(resized_path, hippo_dir)
                extracted_path = extract_hippocampus_slices(
                    left_hippo,
                    right_hippo,
                    resized_path,
                    hippo_dir
                )

            # Step 5: Convert to NPY (full brain)
            if not existing['npy']:
                logger.info(f"Converting full brain to NPY for {adni_id}")
                try:
                    npy_path = os.path.join(config['npy_dir'], f"{adni_id}.npy")
                    image = LoadImage(dtype=np.float32, image_only=True)(resized_path)
                    image_dict = {"image": image}
                    image_dict = EnsureChannelFirstd(keys="image")(image_dict)
                    image_dict = Orientationd(keys="image", axcodes="RAS")(image_dict)
                    np.save(npy_path, image_dict["image"])
                    npy_files[adni_id] = npy_path
                except Exception as e:
                    logger.error(f"NPY conversion failed for {adni_id}: {str(e)}")
                    continue

            # Step 6: Convert hippocampus to NPY
            npy_seg_path = os.path.join(config['npy_seg_dir'], f"{adni_id}.npy")
            if not os.path.exists(npy_seg_path) and os.path.exists(extracted_path):
                logger.info(f"Converting hippocampus to NPY for {adni_id}")
                try:
                    image = LoadImage(dtype=np.float32, image_only=True)(extracted_path)
                    image_dict = {"image": image}
                    image_dict = EnsureChannelFirstd(keys="image")(image_dict)
                    image_dict = Orientationd(keys="image", axcodes="RAS")(image_dict)
                    np.save(npy_seg_path, image_dict["image"])
                except Exception as e:
                    logger.error(f"Hippocampus NPY conversion failed for {adni_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing {adni_id}: {str(e)}")
            continue

    # Update DataFrame with new paths
    df['adni_id'] = df['mri_path'].apply(get_adni_id_from_path)
    df['processed_path'] = df['adni_id'].apply(lambda x: processed_nifti.get(x, ''))
    df['registered_path'] = df['adni_id'].apply(lambda x: registered_files.get(x, {}).get('path', ''))
    df['npy_path'] = df['adni_id'].apply(lambda x: npy_files.get(x, ''))
    
    # Add registration metrics if available
    df['registration_ssim'] = df['adni_id'].apply(
        lambda x: registered_files.get(x, {}).get('metrics', {}).get('ssim', None)
    )
    df['registration_mse'] = df['adni_id'].apply(
        lambda x: registered_files.get(x, {}).get('metrics', {}).get('mse', None)
    )
    
    # Save updated CSV
    updated_csv_path = os.path.join(config['root_dir'], "dataset_MRI_cohort1_updated.csv")
    df.to_csv(updated_csv_path, index=False)
    logger.info(f"Updated CSV saved to {updated_csv_path}")

if __name__ == "__main__":
    main()
