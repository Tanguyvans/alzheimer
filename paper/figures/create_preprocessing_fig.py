#!/usr/bin/env python3
"""
Create preprocessing pipeline figure with REAL MRI images
Shows actual brain slices at each preprocessing step
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
import dicom2nifti
import shutil

# Patient info
PATIENT_ID = "029_S_0836"
DICOM_PATH = "/Volumes/KINGSTON/ADNI/029_S_0836/MP-RAGE/2006-08-23_12_49_43.0/I23231"
MNI_TEMPLATE = str(Path(__file__).parent.parent.parent / "mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii")

OUTPUT_DIR = Path(__file__).parent / "preprocessing_stages"


def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM to NIfTI"""
    print(f"Converting DICOM to NIfTI...")
    dicom2nifti.convert_directory(dicom_dir, str(output_path.parent), compression=True)
    for f in output_path.parent.glob("*.nii.gz"):
        shutil.move(str(f), str(output_path))
        break
    return output_path


def apply_n4_bias_correction(input_path, output_path):
    """Apply N4 bias field correction"""
    print(f"Applying N4 bias correction...")
    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(image, mask)
    sitk.WriteImage(corrected, str(output_path))
    return output_path


def register_to_mni(input_path, template_path, output_path):
    """Register to MNI template using ANTs"""
    print(f"Registering to MNI template...")
    import ants
    moving = ants.image_read(str(input_path))
    fixed = ants.image_read(str(template_path))
    registration = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='Affine', verbose=False
    )
    ants.image_write(registration['warpedmovout'], str(output_path))
    return output_path


def skull_strip(input_path, output_path):
    """Skull stripping using SynthStrip Docker"""
    print(f"Skull stripping with SynthStrip Docker...")

    # Import from existing preprocessing module
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from preprocessing.imaging.skull_stripping import synthstrip_skull_strip, setup_synthstrip_docker

    # Setup Docker
    if not setup_synthstrip_docker():
        raise RuntimeError("Failed to setup SynthStrip Docker. Make sure Docker is running.")

    # Run skull stripping
    success = synthstrip_skull_strip(str(input_path), str(output_path))
    if not success:
        raise RuntimeError("SynthStrip failed")

    return output_path


def get_slice_at_position(nifti_path, slice_idx=None, view='axial'):
    """Get slice from NIfTI at specific index or middle"""
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if view == 'axial':
        idx = slice_idx if slice_idx is not None else data.shape[2] // 2
        return np.rot90(data[:, :, idx])
    elif view == 'coronal':
        idx = slice_idx if slice_idx is not None else data.shape[1] // 2
        return np.rot90(data[:, idx, :])
    else:
        idx = slice_idx if slice_idx is not None else data.shape[0] // 2
        return np.rot90(data[idx, :, :])


def find_brain_center_slice(nifti_path, view='axial'):
    """Find slice with maximum brain content (best for visualization)"""
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if view == 'axial':
        # Find slice with maximum non-zero content in upper half of brain
        axis = 2
        start = data.shape[axis] // 3  # Skip lower slices
        end = int(data.shape[axis] * 0.7)  # Skip very top
    elif view == 'coronal':
        axis = 1
        start = data.shape[axis] // 4
        end = int(data.shape[axis] * 0.75)
    else:
        axis = 0
        start = data.shape[axis] // 4
        end = int(data.shape[axis] * 0.75)

    # Find slice with most brain content
    best_idx = start
    best_score = 0
    for i in range(start, end):
        if axis == 2:
            slc = data[:, :, i]
        elif axis == 1:
            slc = data[:, i, :]
        else:
            slc = data[i, :, :]

        # Score based on non-zero content and variance (good contrast)
        score = np.sum(slc > 0) * np.std(slc)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def normalize_slice(s):
    """Normalize for display"""
    s = s.astype(float)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return s


def find_matching_slice(native_path, registered_path, registered_slice_idx, view='axial'):
    """Find the native slice that best matches the registered slice using cross-correlation"""
    from scipy.ndimage import zoom as scipy_zoom

    # Get registered slice as reference
    ref_slice = normalize_slice(get_slice_at_position(registered_path, registered_slice_idx, view))

    native_img = nib.load(str(native_path))
    native_data = native_img.get_fdata()

    if view == 'axial':
        n_slices = native_data.shape[2]
    elif view == 'coronal':
        n_slices = native_data.shape[1]
    else:
        n_slices = native_data.shape[0]

    best_idx = n_slices // 2
    best_score = -1

    # Search through slices
    for i in range(n_slices // 4, int(n_slices * 0.85)):
        native_slice = normalize_slice(get_slice_at_position(native_path, i, view))

        # Resize to match if needed
        if native_slice.shape != ref_slice.shape:
            zoom_factors = [r / n for r, n in zip(ref_slice.shape, native_slice.shape)]
            native_slice = scipy_zoom(native_slice, zoom_factors, order=1)

        # Compute normalized cross-correlation
        score = np.corrcoef(native_slice.flatten(), ref_slice.flatten())[0, 1]

        if score > best_score:
            best_score = score
            best_idx = i

    print(f"    Best matching slice: {best_idx} (correlation: {best_score:.3f})")
    return best_idx


def create_figure(stages, output_path):
    """Create the preprocessing figure"""
    print("Creating figure...")

    titles = ['Original', 'N4 Corrected', 'MNI Registered', 'Skull Stripped']
    labels = ['DICOM→NIfTI', 'ANTs N4', 'Affine to MNI', 'SynthStrip']

    # Manual slice settings (found with slice_viewer.py)
    REGISTERED_SLICE = 63
    NATIVE_SLICE = 148

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for idx, (key, path) in enumerate(stages.items()):
        if key in ['nifti', 'n4']:
            slice_idx = NATIVE_SLICE
        else:
            slice_idx = REGISTERED_SLICE

        slice_data = normalize_slice(get_slice_at_position(path, slice_idx, view='axial'))
        print(f"  {key}: using slice {slice_idx}")

        axes[idx].imshow(slice_data, cmap='gray', aspect='equal')
        axes[idx].set_title(titles[idx], fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        axes[idx].text(0.5, -0.08, labels[idx], transform=axes[idx].transAxes,
                       ha='center', fontsize=9, style='italic', color='gray')

    plt.suptitle('MRI Preprocessing Pipeline (ADNI)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved {output_path.with_suffix('.pdf')} and {output_path.with_suffix('.png')}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        'nifti': OUTPUT_DIR / f"{PATIENT_ID}_01_nifti.nii.gz",
        'n4': OUTPUT_DIR / f"{PATIENT_ID}_02_n4.nii.gz",
        'registered': OUTPUT_DIR / f"{PATIENT_ID}_03_registered.nii.gz",
        'skull_stripped': OUTPUT_DIR / f"{PATIENT_ID}_04_skull_stripped.nii.gz",
    }

    if not os.path.exists(DICOM_PATH):
        print(f"ERROR: DICOM not found: {DICOM_PATH}")
        sys.exit(1)

    if not paths['nifti'].exists():
        convert_dicom_to_nifti(DICOM_PATH, paths['nifti'])

    if not paths['n4'].exists():
        apply_n4_bias_correction(paths['nifti'], paths['n4'])

    if not paths['registered'].exists():
        register_to_mni(paths['n4'], MNI_TEMPLATE, paths['registered'])

    if not paths['skull_stripped'].exists():
        skull_strip(paths['registered'], paths['skull_stripped'])

    create_figure(paths, OUTPUT_DIR.parent / "preprocessing_pipeline")


if __name__ == "__main__":
    main()
