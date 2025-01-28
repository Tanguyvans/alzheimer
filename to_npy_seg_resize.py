import os
import numpy as np
import pandas as pd
import dicom2nifti
import logging
import matplotlib.pyplot as plt
import ants
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
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
import subprocess
from scipy.ndimage import median_filter, label, binary_fill_holes

# Add logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add BASE_DIR constant
BASE_DIR = ""

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

def resize_to_mni_resolution(input_path, output_path, hw=128):
    """Resize image to 128xYx128 resolution (preserving Y dimension)."""
    logger.info("Resizing to 128xYx128 resolution...")
    
    # Load the input image
    input_img = ants.image_read(input_path)
    input_shape = input_img.shape
    
    print(f"Original image shape: {input_shape}")
    
    # Target dimensions: 128 x original_Y x 128
    target_dims = (hw, input_shape[1], hw)
    
    print(f"Target shape: {target_dims}")
    
    # Resize image
    resized_img = ants.resample_image(
        input_img,
        target_dims,
        use_voxels=True,
        interp_type=1  # Linear interpolation
    )
    
    # Save the resized image
    ants.image_write(resized_img, output_path)
    logger.info(f"Resized image saved to: {output_path}")
    print(f"Final shape: {resized_img.shape}")
    
    return output_path

def crop_center(input_path, output_path, hw=128):
    """Crop left side of the image to hwxhw (preserving Y dimension)."""
    logger.info(f"Cropping to {hw}x{hw} from left side...")
    
    # Load the input image
    input_img = ants.image_read(input_path)
    input_array = input_img.numpy()
    input_shape = input_array.shape
    
    print(f"Original shape before crop: {input_shape}")
    
    # Calculate crop coordinates - taking left side of the image
    x_center = input_shape[0] // 2
    z_start = 0                # Start from the left side
    z_end = z_start + hw      # Take hw pixels from start
    
    x_start = x_center - hw//2
    x_end = x_start + hw
    
    # Ensure we don't go out of bounds
    x_start = max(0, x_start)
    x_end = min(input_shape[0], x_end)
    z_start = max(0, z_start)
    z_end = min(input_shape[2], z_end)
    
    # Crop the image
    cropped_array = input_array[x_start:x_end, :, z_start:z_end]
    
    # If the cropped size is smaller than desired, pad it
    if cropped_array.shape[0] < hw or cropped_array.shape[2] < hw:
        pad_x = max(0, hw - cropped_array.shape[0])
        pad_z = max(0, hw - cropped_array.shape[2])
        
        cropped_array = np.pad(
            cropped_array,
            ((pad_x//2, pad_x - pad_x//2),
             (0, 0),
             (pad_z//2, pad_z - pad_z//2)),
            mode='constant'
        )
    
    print(f"Final shape after crop: {cropped_array.shape}")

    # Create new ANTs image
    cropped_img = ants.from_numpy(
        cropped_array,
        origin=input_img.origin,
        spacing=input_img.spacing,
        direction=input_img.direction
    )
    
    # Save the cropped image
    ants.image_write(cropped_img, output_path)
    logger.info(f"Cropped image saved to: {output_path}")
    
    return output_path

def segment_hippocampus(input_image_path, output_folder):
    """Segment the hippocampus from an input image using the AAL atlas."""
    print("\n=== Starting Hippocampus Segmentation ===")
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the input image first
    input_img = nib.load(input_image_path)
    input_shape = input_img.shape
    print(f"Input image shape: {input_shape}")
    
    # Load the AAL atlas
    aal = nilearn.datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nilearn.image.load_img(aal.maps)
    print(f"Original AAL atlas shape: {aal_img.shape}")
    
    # Resize AAL atlas to match input image dimensions
    aal_resized = nilearn.image.resample_img(
        aal_img,
        target_affine=input_img.affine,
        target_shape=input_shape,
        interpolation='nearest'
    )
    print(f"Resized AAL atlas shape: {aal_resized.shape}")
    
    # Define the hippocampus regions in AAL atlas
    HIPPOCAMPUS_L = 4101
    HIPPOCAMPUS_R = 4102
    
    # Create masks for left and right hippocampus
    aal_data = aal_resized.get_fdata()
    mask_left = (aal_data == HIPPOCAMPUS_L).astype(np.float32)
    mask_right = (aal_data == HIPPOCAMPUS_R).astype(np.float32)
    
    # Create NIfTI images for the masks
    mask_left_img = nib.Nifti1Image(mask_left, input_img.affine)
    mask_right_img = nib.Nifti1Image(mask_right, input_img.affine)
    
    # Save the mask images
    left_mask_path = os.path.join(output_folder, 'hippocampus_left_mask.nii.gz')
    right_mask_path = os.path.join(output_folder, 'hippocampus_right_mask.nii.gz')
    nib.save(mask_left_img, left_mask_path)
    nib.save(mask_right_img, right_mask_path)
    
    # Apply masks to get segmented hippocampus
    input_data = input_img.get_fdata()
    left_hippo = input_data * mask_left
    right_hippo = input_data * mask_right
    
    # Save segmented hippocampus
    left_hippo_path = os.path.join(output_folder, 'hippocampus_left_segmented.nii.gz')
    right_hippo_path = os.path.join(output_folder, 'hippocampus_right_segmented.nii.gz')
    
    nib.save(nib.Nifti1Image(left_hippo, input_img.affine), left_hippo_path)
    nib.save(nib.Nifti1Image(right_hippo, input_img.affine), right_hippo_path)
    
    return left_mask_path, right_mask_path, left_hippo_path, right_hippo_path

def extract_hippocampus_slices(left_hippo_path, right_hippo_path, input_path, output_dir):
    """Extract a 3D volume containing 20 coronal slices centered on hippocampus with 192x192 resolution."""
    print("\n=== Starting Hippocampus Slice Extraction ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    input_img = nib.load(input_path)
    left_hippo = nib.load(left_hippo_path).get_fdata()
    right_hippo = nib.load(right_hippo_path).get_fdata()
    
    print(f"Input shape before processing: {input_img.get_fdata().shape}")
    
    # Get dimensions
    _, n_coronal, _ = left_hippo.shape
    
    # Find slices containing hippocampus
    left_slices = set([i for i in range(n_coronal) if np.any(left_hippo[:, i, :])])
    right_slices = set([i for i in range(n_coronal) if np.any(right_hippo[:, i, :])])
    
    # Combine slices from both hippocampi
    hippo_slices = sorted(list(left_slices.union(right_slices)))
    print(f"Found {len(hippo_slices)} slices containing hippocampus")
    
    # Calculate center of hippocampus region
    center_slice = hippo_slices[len(hippo_slices)//2]
    
    # Prendre 20 coupes centrées sur l'hippocampe
    total_slices_needed = 20
    half_window = total_slices_needed // 2
    
    start_slice = max(0, center_slice - half_window)
    end_slice = min(n_coronal, start_slice + total_slices_needed)
    
    # Ajuster si on atteint les bords
    if end_slice == n_coronal:
        start_slice = max(0, n_coronal - total_slices_needed)
    elif start_slice == 0:
        end_slice = min(n_coronal, total_slices_needed)
    
    # Create final slice list
    final_slices = list(range(start_slice, min(start_slice + total_slices_needed, n_coronal)))
    print(f"Selected {len(final_slices)} central slices")
    
    # Extract relevant slices
    image_extracted = input_img.get_fdata()[:, final_slices, :]
    hippo_mask_extracted = left_hippo[:, final_slices, :] + right_hippo[:, final_slices, :]
    print(f"Extracted volume shape: {image_extracted.shape}")
    
    # Resize to 192x20x192 (résolution plus élevée)
    target_shape = (192, 20, 192)
    
    # Convert to ANTs image for resizing
    extracted_ants = ants.from_numpy(image_extracted)
    resized_ants = ants.resample_image(
        extracted_ants,
        target_shape,
        use_voxels=True,
        interp_type=1
    )
    resized_array = resized_ants.numpy()
    
    # Resize hippocampus mask
    hippo_mask_ants = ants.from_numpy(hippo_mask_extracted)
    resized_mask = ants.resample_image(
        hippo_mask_ants,
        target_shape,
        use_voxels=True,
        interp_type=0  # Nearest neighbor for mask
    )
    resized_mask_array = resized_mask.numpy()
    
    print(f"Final resized shape: {resized_array.shape}")
    
    # Vérifier que l'hippocampe est bien visible dans les coupes
    hippo_intensity = np.mean(resized_array[resized_mask_array > 0])
    print(f"Average hippocampus intensity in selected slices: {hippo_intensity:.2f}")
    
    # Create new NIfTI image
    new_affine = input_img.affine.copy()
    new_affine[1, 3] = new_affine[1, 3] + (final_slices[0] * new_affine[1, 1])
    
    extracted_nifti = nib.Nifti1Image(resized_array, new_affine, input_img.header)
    output_path = os.path.join(output_dir, 'extracted_volume.nii.gz')
    nib.save(extracted_nifti, output_path)
    
    # Save slice information
    slice_info = {
        'total_original_slices': n_coronal,
        'original_hippo_slices': hippo_slices,
        'selected_slices': final_slices,
        'center_slice': center_slice,
        'total_extracted_slices': len(final_slices),
        'final_shape': [int(x) for x in resized_array.shape],
        'average_hippo_intensity': float(hippo_intensity)
    }
    
    with open(os.path.join(output_dir, 'slice_info.json'), 'w') as f:
        json.dump(slice_info, f, indent=4)
    
    return output_path, slice_info

def segment_brain_tissues(image):
    """Segmentation des tissus cérébraux."""
    # Segmentation matière grise/blanche
    segments = ants.atropos(image)
    
    # Extraction des caractéristiques volumétriques
    volumes = {
        'gray_matter': np.sum(segments['probability_images'][0]),
        'white_matter': np.sum(segments['probability_images'][1]),
        'csf': np.sum(segments['probability_images'][2])
    }
    
    return segments, volumes

def enhance_image_quality(image):
    """Amélioration spécifique pour la détection d'Alzheimer.
    
    This function enhances the quality of brain MRI images specifically for Alzheimer's detection by:
    1. Applying N4 bias field correction with aggressive parameters to remove intensity non-uniformity
    2. Denoising the image using a Gaussian noise model to reduce noise
    3. Performing adaptive intensity normalization to standardize intensities
    4. Enhancing contrast to better visualize brain atrophy
    
    Args:
        image: An ANTs image object containing the brain MRI scan
        
    Returns:
        enhanced: An ANTs image object with enhanced quality
    """
    logger.info("Enhancing image quality for Alzheimer's detection...")
    
    # 1. Correction de biais N4 (plus agressive)
    n4 = ants.n4_bias_field_correction(
        image,
        shrink_factor=2,  # Plus précis
        convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},  # Plus d'itérations
        spline_param=200  # Meilleure correction locale
    )
    
    # 2. Débruitage (optionnel)
    denoised = ants.denoise_image(n4, noise_model='Gaussian')
    
    # 3. Normalisation d'intensité adaptative
    normalized = ants.iMath(denoised, "Normalize")
    
    # 4. Amélioration du contraste pour mieux voir l'atrophie
    enhanced = ants.iMath(normalized, "Sharpen")
    
    return enhanced

def extract_brain_features(image):
    """Extraction de caractéristiques morphométriques."""
    logger.info("Extracting brain features...")
    features = {}
    
    # Volume total du cerveau
    brain_mask = ants.get_mask(image)
    features['brain_volume'] = float(np.sum(brain_mask.numpy()))
    
    # Intensité moyenne et écart-type
    brain_data = image.numpy() * brain_mask.numpy()
    features['mean_intensity'] = float(np.mean(brain_data[brain_data > 0]))
    features['std_intensity'] = float(np.std(brain_data[brain_data > 0]))
    
    # Asymétrie gauche-droite
    mid_slice = brain_data.shape[2] // 2
    left_half = brain_data[:, :, :mid_slice]
    right_half = brain_data[:, :, mid_slice:]
    features['asymmetry_index'] = float(np.abs(np.mean(left_half) - np.mean(right_half)))
    
    return features

def skull_stripping(input_path, output_path):
    """Remove skull using HD-BET (High-Definition Brain Extraction Tool).
    
    Args:
        input_path: Path to input NIfTI image
        output_path: Path where to save the skull-stripped image
        
    Returns:
        str: Path to the skull-stripped image
    """
    logger.info("Performing skull stripping using HD-BET...")
    
    try:
        # HD-BET command avec les bons arguments
        # -i: fichier d'entrée
        # -o: fichier de sortie
        # -device cpu: utilise le CPU (changer en 'cuda' si GPU disponible)
        # --disable_tta: désactive le test time augmentation pour plus de vitesse
        cmd = f"hd-bet -i {input_path} -o {output_path} -device cpu --disable_tta"
        
        # Exécuter HD-BET
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"HD-BET failed with error: {stderr.decode()}")
            
        # HD-BET ajoute '_bet' au nom du fichier
        actual_output = output_path.replace('.nii.gz', '_bet.nii.gz')
        if os.path.exists(actual_output):
            os.rename(actual_output, output_path)
            
        logger.info(f"Skull stripping completed. Output saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during skull stripping: {str(e)}")
        raise

def main():
    """Pipeline for processing a single NIfTI image."""
    config = {
        'root_dir': BASE_DIR,
        'output_dir': os.path.join(BASE_DIR, "output"),
        'npy_dir': os.path.join(BASE_DIR, "output/npy"),
        'npy_seg_dir': os.path.join(BASE_DIR, "output/npy_seg"),
        'register_dir': os.path.join(BASE_DIR, "output/register"),
        'template_path': os.path.join(BASE_DIR, "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"),
        'input_path': os.path.join(BASE_DIR, "normal_1.nii.gz"),
        
        # Flags pour les features optionnelles
        'enhance_quality': True,      # Amélioration de la qualité
        'extract_features': True,     # Extraction de caractéristiques
        'skull_strip': True,          # Retrait du crâne
    }

    try:
        # Créer les dossiers nécessaires
        subject_dir = os.path.join(config['register_dir'], 'normal')
        os.makedirs(subject_dir, exist_ok=True)
        os.makedirs(config['npy_dir'], exist_ok=True)

        # Step 1: Registration
        registered_path = os.path.join(subject_dir, "normal_1.nii.gz")
        if not os.path.exists(registered_path):
            registered_path = register_to_mni_ants(
                config['input_path'],
                subject_dir,
                config['template_path'],
                'normal'
            )
        
        if not os.path.exists(registered_path):
            raise FileNotFoundError(f"Registration failed: {registered_path} not found")
            
        current_path = registered_path

        # Step 2: Image Enhancement (avant skull stripping)
        enhanced_path = os.path.join(subject_dir, "normal_enhanced.nii.gz")
        if config['enhance_quality'] and not os.path.exists(enhanced_path):
            logger.info("Processing image enhancement")
            img = ants.image_read(current_path)
            enhanced_img = enhance_image_quality(img)
            ants.image_write(enhanced_img, enhanced_path)
            current_path = enhanced_path

        # Step 3: Skull Stripping (après enhancement)
        stripped_path = os.path.join(subject_dir, "normal_stripped.nii.gz")
        if config['skull_strip'] and not os.path.exists(stripped_path):
            logger.info("Performing skull stripping")
            stripped_path = skull_stripping(current_path, stripped_path)
            current_path = stripped_path

        # Step 4: Feature Extraction (optionnel)
        features_path = os.path.join(subject_dir, "brain_features.json")
        if config['extract_features'] and not os.path.exists(features_path):
            logger.info("Extracting brain features")
            img = ants.image_read(current_path)
            features = extract_brain_features(img)
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=4)

        # Step 5: Hippocampus Segmentation
        hippo_dir = os.path.join(subject_dir, 'hippocampus')
        os.makedirs(hippo_dir, exist_ok=True)
        
        # Segment hippocampus
        left_mask_path, right_mask_path, left_hippo_path, right_hippo_path = segment_hippocampus(
            current_path, 
            hippo_dir
        )
        
        # Extract hippocampus slices
        extracted_path, slice_info = extract_hippocampus_slices(
            left_hippo_path,
            right_hippo_path,
            current_path,
            hippo_dir
        )
        current_path = extracted_path

        # # Step 6: Resize and Crop
        # resized_path = os.path.join(subject_dir, "normal_resized.nii.gz")
        # if not os.path.exists(resized_path):
        #     hw = 192
        #     logger.info(f"Resizing to {hw}x{hw}")
        #     resized_path = resize_to_mni_resolution(current_path, resized_path, hw)
            
        #     cropped_path = os.path.join(subject_dir, "normal_cropped.nii.gz")
        #     logger.info("Cropping to 192x192")
        #     cropped_path = crop_center(resized_path, cropped_path, hw=192)
        #     current_path = cropped_path

        # Step 7: Convert to NPY
        npy_path = os.path.join(config['npy_dir'], "normal.npy")
        if not os.path.exists(npy_path):
            logger.info("Converting to NPY")
            image = LoadImage(dtype=np.float32, image_only=True)(current_path)
            image_dict = {"image": image}
            image_dict = EnsureChannelFirstd(keys="image")(image_dict)
            image_dict = Orientationd(keys="image", axcodes="RAS")(image_dict)
            np.save(npy_path, image_dict["image"])

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

    logger.info("Processing completed")

if __name__ == "__main__":
    main()
