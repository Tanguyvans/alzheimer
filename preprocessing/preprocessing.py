import os
import numpy as np
import logging

import SimpleITK as sitk

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from scipy.stats import entropy
import ants
import nibabel as nib

import nilearn.datasets
import nilearn.image
import nilearn.plotting
from dotenv import load_dotenv
import dicom2nifti

def convert_dicom_to_nifti(dicom_folder, output_folder):
    """Convert a folder of DICOM files to NIfTI format."""
    os.makedirs(output_folder, exist_ok=True)
    
    folder_name = os.path.basename(dicom_folder)
    output_file = os.path.join(output_folder, f"{folder_name}.nii.gz")
    dicom2nifti.convert_directory(dicom_folder, output_folder, compression=True, reorient=True)
    return output_file

def convert_adni_to_nifti(adni_root, nifti_root):
    """Convert all DICOM folders in the ADNI directory to NIfTI format."""
    for root, dirs, files in os.walk(adni_root):
        if files and any(file.endswith('.dcm') for file in files):
            # This is a DICOM folder
            relative_path = os.path.relpath(root, adni_root)
            output_folder = os.path.join(nifti_root, os.path.dirname(relative_path))
            nifti_file = convert_dicom_to_nifti(root, output_folder)
            print(f"Converted {root} to {nifti_file}")

def register_to_mni_ants(input_file, output_dir, mni_template):
    logging.info(f"Starting registration for {input_file}")
    
    # Charger les images
    moving_image = ants.image_read(input_file)
    fixed_image = ants.image_read(mni_template)
    
    # Effectuer le recalage
    registration = ants.registration(fixed=fixed_image, moving=moving_image,
                                     type_of_transform='SyN')
    
    # Sauvegarder l'image recalée
    output_file = os.path.join(output_dir, f'mni_{os.path.basename(input_file)}')
    ants.image_write(registration['warpedmovout'], output_file)
    
    logging.info(f"Registration completed. Output saved to {output_file}")
    
    return output_file

def compare_registration(original_file, registered_file, output_dir, plot=True):
    logging.info(f"Comparing registration for {original_file}")
    
    # Charger les images
    original_img = ants.image_read(original_file)
    registered_img = ants.image_read(registered_file)
    
    # Convertir en numpy arrays
    original_array = original_img.numpy()
    registered_array = registered_img.numpy()
    
    # Assurez-vous que les dimensions correspondent
    if original_array.shape != registered_array.shape:
        logging.warning("Les dimensions des images ne correspondent pas. Un redimensionnement peut être nécessaire.")
        # Redimensionner l'image originale pour correspondre à l'image recalée
        original_array = ants.resample_image_to_target(original_img, registered_img).numpy()
    
    # Calculer SSIM
    ssim_value = ssim(original_array, registered_array, data_range=registered_array.max() - registered_array.min())
    
    # Calculer MSE
    mse_value = mean_squared_error(original_array, registered_array)
    
    if plot:
        # Créer une visualisation
        mid_slice = original_array.shape[1] // 2  # Vue coronale
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(original_array[:, mid_slice, :], cmap='gray', aspect='auto')
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(registered_array[:, mid_slice, :], cmap='gray', aspect='auto')
        ax2.set_title('Recalée')
        ax2.axis('off')
        
        # Image de différence
        diff_image = registered_array - original_array
        vmax = np.abs(diff_image).max()
        ax3.imshow(diff_image[:, mid_slice, :], cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        ax3.set_title('Différence')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'registration_comparison.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # Sauvegarder les métriques
    with open(os.path.join(output_dir, 'registration_metrics.txt'), 'w') as f:
        f.write(f"SSIM entre original et recalé: {ssim_value}\n")
        f.write(f"MSE entre original et recalé: {mse_value}\n")
    
    logging.info(f"Comparaison de recalage effectuée. SSIM: {ssim_value}, MSE: {mse_value}")

def normalize_intensity_zscore(image):
    array = sitk.GetArrayFromImage(image)
    mean = np.mean(array)
    std = np.std(array)
    normalized = (array - mean) / std
    return sitk.GetImageFromArray(normalized)

def normalize_intensity_minmax(image):
    array = sitk.GetArrayFromImage(image)
    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val)
    return sitk.GetImageFromArray(normalized)

def calculate_metrics(image):
    """Calculate various metrics for the image."""
    mean = np.mean(image)
    std = np.std(image)
    contrast = np.max(image) - np.min(image)
    
    # Ajout d'une petite valeur pour éviter log(0)
    epsilon = 1e-10
    image_positive = image - np.min(image) + epsilon
    
    # Normalisation pour que la somme soit 1
    image_normalized = image_positive / np.sum(image_positive)
    
    img_entropy = entropy(image_normalized.flatten())
    
    return mean, std, contrast, img_entropy

def plot_comparison(original, zscore, minmax, slice_index, output_path):
    """Plot original, Z-score normalized, and min-max normalized images."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original[slice_index], cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(zscore[slice_index], cmap='gray')
    ax2.set_title('Z-score Normalized')
    ax2.axis('off')
    
    ax3.imshow(minmax[slice_index], cmap='gray')
    ax3.set_title('Min-Max Normalized')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_adni_folder(adni_root, output_root, mni_template, plot=True):
    """Process all NIfTI files in the ADNI_NII directory."""
    for root, dirs, files in os.walk(adni_root):
        for file in files:
            if file.endswith('.nii.gz'):
                nifti_file = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, adni_root)
                output_folder = os.path.join(output_root, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                try:
                    # Register to MNI space
                    mni_registered_file = register_to_mni_ants(nifti_file, output_folder, mni_template)
                    logging.info(f"Registered {nifti_file} to MNI space")
                    
                    # Compare registration
                    compare_registration(nifti_file, mni_registered_file, output_folder, plot=plot)
                    
                    # Use the MNI-registered image for further processing
                    image = sitk.ReadImage(mni_registered_file)
                    original_array = sitk.GetArrayFromImage(image)
                    
                    normalized_zscore = normalize_intensity_zscore(image)
                    zscore_array = sitk.GetArrayFromImage(normalized_zscore)
                    sitk.WriteImage(normalized_zscore, os.path.join(output_folder, f"{os.path.splitext(file)[0]}_normalized_zscore.nii.gz"))
                    
                    normalized_minmax = normalize_intensity_minmax(image)
                    minmax_array = sitk.GetArrayFromImage(normalized_minmax)
                    sitk.WriteImage(normalized_minmax, os.path.join(output_folder, f"{os.path.splitext(file)[0]}_normalized_minmax.nii.gz"))
                    
                    # Calculate metrics
                    orig_metrics = calculate_metrics(original_array)
                    zscore_metrics = calculate_metrics(zscore_array)
                    minmax_metrics = calculate_metrics(minmax_array)
                    
                    # Calculate SSIM
                    ssim_zscore = ssim(original_array, zscore_array, data_range=zscore_array.max() - zscore_array.min())
                    ssim_minmax = ssim(original_array, minmax_array, data_range=minmax_array.max() - minmax_array.min())
                    
                    # Write metrics to a file
                    with open(os.path.join(output_folder, f"{os.path.splitext(file)[0]}_metrics.txt"), 'w') as f:
                        f.write("Metrics (mean, std, contrast, entropy):\n")
                        f.write(f"Original (MNI space): {orig_metrics}\n")
                        f.write(f"Z-score: {zscore_metrics}\n")
                        f.write(f"Min-Max: {minmax_metrics}\n")
                        f.write(f"SSIM (Z-score): {ssim_zscore}\n")
                        f.write(f"SSIM (Min-Max): {ssim_minmax}\n")
                    
                    if plot:
                        # Plot comparison
                        middle_slice = original_array.shape[1] // 2  # Vue coronale
                        plot_comparison(original_array[:, middle_slice, :], 
                                        zscore_array[:, middle_slice, :], 
                                        minmax_array[:, middle_slice, :], 
                                        slice_index=None,  # Pas besoin de slice_index car on a déjà sélectionné la coupe
                                        output_path=os.path.join(output_folder, f"{os.path.splitext(file)[0]}_comparison.png"))
                    
                    logging.info(f"Normalized, plotted, and calculated metrics for {mni_registered_file}")
                except Exception as e:
                    logging.error(f"Error processing {nifti_file}: {str(e)}")

def analyze_mri_dimensions(root_dir, is_output=False):
    dimensions = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                if is_output and not file.startswith('mni_'):
                    continue  # Pour output_root, on ne s'intéresse qu'aux fichiers recalés
                nifti_file = os.path.join(root, file)
                try:
                    img = nib.load(nifti_file)
                    dim = img.shape
                    if dim in dimensions:
                        dimensions[dim].append(nifti_file)
                    else:
                        dimensions[dim] = [nifti_file]
                except Exception as e:
                    logging.error(f"Erreur lors de la lecture de {nifti_file}: {str(e)}")

    logging.info(f"Analyse des dimensions des MRI terminée pour {root_dir}")
    for dim, files in dimensions.items():
        logging.info(f"Dimension {dim}: {len(files)} fichiers")
        if len(files) < 5:
            for f in files:
                logging.info(f"  - {f}")
        else:
            logging.info(f"  - {files[0]}")
            logging.info(f"  - {files[-1]}")
            logging.info(f"  - ... et {len(files)-2} autres")

    return dimensions

def segment_hippocampus(input_image_path, output_folder):
    """
    Segment the hippocampus from an input image using the AAL atlas.
    
    Parameters:
    input_image_path (str): Path to the input NIfTI image file.
    output_folder (str): Path to the folder where output files will be saved.
    
    Returns:
    tuple: Paths to the left and right hippocampus mask files.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the AAL atlas
    aal = nilearn.datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nilearn.image.load_img(aal.maps)
    
    # Define the regions of interest
    Hippocampus_L = 4101
    Hippocampus_R = 4102
    
    # Load the input image
    input_img = nilearn.image.load_img(input_image_path)
    
    # Resample input image to match the AAL atlas
    resampled_img = nilearn.image.resample_to_img(input_img, aal_img, interpolation='nearest')
    
    # Create masks for left and right hippocampus
    mask_left = (aal_img.get_fdata() == Hippocampus_L).astype(int)
    mask_right = (aal_img.get_fdata() == Hippocampus_R).astype(int)
    
    # Create NIfTI images for the masks
    mask_left_img = nib.Nifti1Image(mask_left, aal_img.affine, aal_img.header)
    mask_right_img = nib.Nifti1Image(mask_right, aal_img.affine, aal_img.header)
    
    # Save the mask images
    left_mask_path = os.path.join(output_folder, 'hippocampus_left_mask.nii.gz')
    right_mask_path = os.path.join(output_folder, 'hippocampus_right_mask.nii.gz')
    nib.save(mask_left_img, left_mask_path)
    nib.save(mask_right_img, right_mask_path)
    
    # Apply masks to the input image data
    input_data = resampled_img.get_fdata()
    hippocampus_l_data = input_data * mask_left
    hippocampus_r_data = input_data * mask_right
    
    # Create new image objects for the masked data
    hippocampus_l_img = nilearn.image.new_img_like(aal_img, hippocampus_l_data)
    hippocampus_r_img = nilearn.image.new_img_like(aal_img, hippocampus_r_data)
    
    # Save the segmented hippocampus images
    left_hippo_path = os.path.join(output_folder, 'hippocampus_left_segmented.nii.gz')
    right_hippo_path = os.path.join(output_folder, 'hippocampus_right_segmented.nii.gz')
    nib.save(hippocampus_l_img, left_hippo_path)
    nib.save(hippocampus_r_img, right_hippo_path)
    
    print(f"Hippocampus segmentation completed. Results saved in {output_folder}")
    
    return left_mask_path, right_mask_path, left_hippo_path, right_hippo_path

def find_middle_slice(data, axis):
    non_zero = np.any(data != 0, axis=(0, 1) if axis == 2 else ((1, 2) if axis == 0 else (0, 2)))
    return np.median(np.where(non_zero)[0]).astype(int)

def plot_hippocampus(left_hippo_path, right_hippo_path, output_path):
    """
    Plot the segmented left and right hippocampus.
    
    Parameters:
    left_hippo_path (str): Path to the left hippocampus segmented image.
    right_hippo_path (str): Path to the right hippocampus segmented image.
    output_path (str): Path to save the output plot.
    """
    # Load the hippocampus images
    left_hippo = nib.load(left_hippo_path)
    right_hippo = nib.load(right_hippo_path)
    
    left_data = left_hippo.get_fdata()
    right_data = right_hippo.get_fdata()
    
    # Find the middle slices
    axial_slice = find_middle_slice(left_data, 2)
    coronal_slice = find_middle_slice(left_data, 1)
    sagittal_slice = find_middle_slice(left_data, 0)
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Segmented Hippocampus', fontsize=16)
    
    # Plot left hippocampus mask
    axes[0, 0].imshow(left_data[:, :, axial_slice].T, cmap='binary')
    axes[0, 0].set_title('Left Hippocampus (Axial)')
    axes[0, 1].imshow(left_data[:, coronal_slice, :].T, cmap='binary')
    axes[0, 1].set_title('Left Hippocampus (Coronal)')
    axes[0, 2].imshow(left_data[sagittal_slice, :, :].T, cmap='binary')
    axes[0, 2].set_title('Left Hippocampus (Sagittal)')
    
    # Plot right hippocampus mask
    axes[1, 0].imshow(right_data[:, :, axial_slice].T, cmap='binary')
    axes[1, 0].set_title('Right Hippocampus (Axial)')
    axes[1, 1].imshow(right_data[:, coronal_slice, :].T, cmap='binary')
    axes[1, 1].set_title('Right Hippocampus (Coronal)')
    axes[1, 2].imshow(right_data[sagittal_slice, :, :].T, cmap='binary')
    axes[1, 2].set_title('Right Hippocampus (Sagittal)')
    
    # Remove axis ticks
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Hippocampus plot saved to {output_path}")
    
    # Plot combined mask
    combined_data = left_data + 2 * right_data
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Combined Hippocampus Mask', fontsize=16)
    
    axes[0].imshow(combined_data[:, :, axial_slice].T, cmap='Set1', vmin=0, vmax=2)
    axes[0].set_title('Axial')
    axes[1].imshow(combined_data[:, coronal_slice, :].T, cmap='Set1', vmin=0, vmax=2)
    axes[1].set_title('Coronal')
    axes[2].imshow(combined_data[sagittal_slice, :, :].T, cmap='Set1', vmin=0, vmax=2)
    axes[2].set_title('Sagittal')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    combined_output_path = output_path.replace('.png', '_combined.png')
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Combined hippocampus plot saved to {combined_output_path}")
    
# Dans votre fonction principale
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Utiliser les variables d'environnement ADNI_PATH et MNI_TEMPLATE
    root_path = os.path.abspath(os.getenv('ROOT_PATH', '../../MRI'))
    
    # Définir les chemins relatifs au ROOT_PATH
    adni_root = os.path.join(root_path, 'ADNI_small')
    nifti_root = os.path.join(root_path, 'ADNI_NII')
    output_root = os.path.join(root_path, 'ADNI_processed')
    mni_template = os.path.join(root_path, "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii")
    
    # Ajout d'un paramètre pour contrôler les plots
    generate_plots = True  # Changez à False pour désactiver les plots
    
    # Nouveau paramètre pour contrôler la segmentation de l'hippocampe
    segment_hippocampus_flag = True  # Changez à False pour désactiver la segmentation de l'hippocampe
    
    if not os.path.exists(mni_template):
        logging.error(f"Erreur : Le template MNI n'existe pas à l'emplacement {mni_template}")
    elif not os.path.exists(adni_root):
        logging.error(f"Erreur : Le dossier ADNI n'existe pas à l'emplacement {adni_root}")
    else:
        # Convertir les fichiers DICOM en NIfTI
        logging.info("Conversion des fichiers DICOM en NIfTI...")
        convert_adni_to_nifti(adni_root, nifti_root)
        
        # Analyser les dimensions avant le prétraitement
        logging.info("Analyse des dimensions des MRI originaux :")
        original_dimensions = analyze_mri_dimensions(nifti_root)
        
        # Effectuer le prétraitement
        process_adni_folder(nifti_root, output_root, mni_template, plot=generate_plots)
        
        # Analyser les dimensions après le prétraitement
        logging.info("Analyse des dimensions des MRI après recalage :")
        processed_dimensions = analyze_mri_dimensions(output_root, is_output=True)
        
        # Comparer les résultats
        logging.info("Comparaison des dimensions avant et après prétraitement :")
        logging.info(f"Dimensions originales : {list(original_dimensions.keys())}")
        logging.info(f"Dimensions après recalage : {list(processed_dimensions.keys())}")
        
    # Segmentation de l'hippocampe (optionnel)
    if segment_hippocampus_flag:
        for root, dirs, files in os.walk(output_root):
            for file in files:
                if file.startswith('mni_') and file.endswith('.nii.gz'):
                    nifti_file = os.path.join(root, file)
                    try:
                        logging.info(f"Processing file: {nifti_file}")
                        left_mask, right_mask, left_hippo, right_hippo = segment_hippocampus(nifti_file, root)
                        logging.info(f"Segmentation de l'hippocampe réussie pour {nifti_file}")
                        
                        # Plot the segmented hippocampus
                        hippo_plot_path = os.path.join(root, f"{os.path.splitext(file)[0]}_hippocampus_plot.png")
                        plot_hippocampus(left_hippo, right_hippo, hippo_plot_path)
                        logging.info(f"Hippocampus plot created for {nifti_file}")
                    except Exception as e:
                        logging.error(f"Erreur lors de la segmentation de l'hippocampe pour {nifti_file}: {str(e)}")
                        logging.exception("Traceback complet:")