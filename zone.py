import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, image
from scipy import ndimage

# 1. Load your brain MRI
brain_img_path = "I54829.nii.gz"
brain_img = nib.load(brain_img_path)
brain_data = brain_img.get_fdata()

# 2. Get MNI152 template and CSF probability map
mni = datasets.fetch_icbm152_2009()
template = nib.load(mni['t1'])
csf = nib.load(mni['csf'])  # CSF probability map includes ventricles

# 3. Resample CSF map to match your image dimensions
csf_resampled = image.resample_to_img(csf, brain_img, interpolation='nearest')
csf_mask = csf_resampled.get_fdata() > 0.5  # Threshold CSF probability map

# 4. Create a brain mask to exclude non-brain regions
brain_data_norm = (brain_data - np.min(brain_data)) / (np.max(brain_data) - np.min(brain_data))
brain_mask = brain_data_norm > 0.1
brain_mask = ndimage.binary_closing(brain_mask, iterations=3)
brain_mask = ndimage.binary_fill_holes(brain_mask)

# 5. Apply spatial constraints to isolate ventricles
# Get brain dimensions
x_dim, y_dim, z_dim = brain_data.shape

# Create a mask for the central region of the brain where ventricles are located
center_x, center_y, center_z = x_dim // 2, y_dim // 2, z_dim // 2
x_range = x_dim // 3
y_range = y_dim // 3
z_range = z_dim // 3

central_mask = np.zeros_like(brain_data, dtype=bool)
central_mask[
    center_x - x_range//2:center_x + x_range//2,
    center_y - y_range//2:center_y + y_range//2,
    center_z - z_range//2:center_z + z_range//2
] = True

# Combine masks to isolate ventricles
ventricle_mask = csf_mask & brain_mask & central_mask

# 6. Clean up the mask
ventricle_mask = ndimage.binary_opening(ventricle_mask, iterations=1)
ventricle_mask = ndimage.binary_closing(ventricle_mask, iterations=2)

# 7. Keep only the largest connected components (likely to be ventricles)
labeled_mask, num_features = ndimage.label(ventricle_mask)
if num_features > 0:
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # Ignore background
    
    # Keep only the 4 largest components (likely to be ventricles)
    num_ventricles = min(4, num_features)
    largest_components = np.argsort(sizes)[-num_ventricles:]
    ventricle_mask = np.isin(labeled_mask, largest_components)

# 8. Calculate volume
voxel_dimensions = brain_img.header.get_zooms()
voxel_volume = np.prod(voxel_dimensions)  # mm³
ventricle_volume_mm3 = np.sum(ventricle_mask) * voxel_volume
ventricle_volume_ml = ventricle_volume_mm3 / 1000

print(f"Ventricle volume: {ventricle_volume_mm3:.2f} mm³ ({ventricle_volume_ml:.2f} ml)")

# 9. Visualize results
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# Get middle slices
slice_x = brain_data.shape[0] // 2
slice_y = brain_data.shape[1] // 2
slice_z = brain_data.shape[2] // 2

# Axial view
axes[0, 0].imshow(brain_data_norm[:, :, slice_z].T, cmap='gray')
axes[0, 0].set_title('Original MRI (Axial)')
axes[0, 0].axis('off')

axes[0, 1].imshow(brain_data_norm[:, :, slice_z].T, cmap='gray')
axes[0, 1].imshow(ventricle_mask[:, :, slice_z].T, alpha=0.5, cmap='hot')
axes[0, 1].set_title('Ventricle Segmentation (Axial)')
axes[0, 1].axis('off')

# Sagittal view
axes[1, 0].imshow(brain_data_norm[slice_x, :, :].T, cmap='gray')
axes[1, 0].set_title('Original MRI (Sagittal)')
axes[1, 0].axis('off')

axes[1, 1].imshow(brain_data_norm[slice_x, :, :].T, cmap='gray')
axes[1, 1].imshow(ventricle_mask[slice_x, :, :].T, alpha=0.5, cmap='hot')
axes[1, 1].set_title('Ventricle Segmentation (Sagittal)')
axes[1, 1].axis('off')

# Coronal view
axes[2, 0].imshow(brain_data_norm[:, slice_y, :].T, cmap='gray')
axes[2, 0].set_title('Original MRI (Coronal)')
axes[2, 0].axis('off')

axes[2, 1].imshow(brain_data_norm[:, slice_y, :].T, cmap='gray')
axes[2, 1].imshow(ventricle_mask[:, slice_y, :].T, alpha=0.5, cmap='hot')
axes[2, 1].set_title('Ventricle Segmentation (Coronal)')
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig('ventricle_segmentation_refined.png')
plt.show()

# 10. Save the ventricle mask
ventricle_nifti = nib.Nifti1Image(ventricle_mask.astype(np.uint8), affine=brain_img.affine)
nib.save(ventricle_nifti, "ventricles_mask_refined.nii.gz")

print("Ventricle segmentation complete. Results saved to 'ventricles_mask_refined.nii.gz'")
