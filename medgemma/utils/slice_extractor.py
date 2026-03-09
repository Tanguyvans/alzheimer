"""
Slice Extractor for MedGemma

Extracts 2D slices from 3D MRI volumes for use with MedGemma vision-language model.
Focused on coronal hippocampus-centered slices for Alzheimer's classification.
"""

import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union
from scipy.ndimage import zoom


class SliceExtractor:
    """
    Extracts 2D slices from 3D MRI volumes.

    Default configuration extracts coronal slices from the hippocampus region,
    which is optimal for Alzheimer's disease assessment.
    """

    def __init__(
        self,
        view: str = "coronal",
        n_slices: int = 5,
        region_start: float = 0.40,
        region_end: float = 0.60,
        output_size: int = 896,
        normalize_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            view: Slice orientation - 'axial', 'coronal', or 'sagittal'
            n_slices: Number of slices to extract
            region_start: Start of region as fraction of axis (0.0-1.0)
            region_end: End of region as fraction of axis (0.0-1.0)
            output_size: Output image size (square)
            normalize_range: Output intensity range (default [-1, 1] for MedGemma)
        """
        self.view = view.lower()
        self.n_slices = n_slices
        self.region_start = region_start
        self.region_end = region_end
        self.output_size = output_size
        self.normalize_range = normalize_range

        # Axis mapping for different views
        self.axis_map = {
            "sagittal": 0,  # x-axis, left-right
            "coronal": 1,   # y-axis, front-back (hippocampus view)
            "axial": 2      # z-axis, top-bottom
        }

        if self.view not in self.axis_map:
            raise ValueError(f"Invalid view: {view}. Must be 'axial', 'coronal', or 'sagittal'")

    def extract_from_nifti(self, nifti_path: Union[str, Path]) -> List[Image.Image]:
        """
        Extract slices from a NIfTI file.

        Args:
            nifti_path: Path to .nii or .nii.gz file

        Returns:
            List of PIL Images (RGB, output_size x output_size)
        """
        nifti = nib.load(str(nifti_path))
        volume = nifti.get_fdata().astype(np.float32)
        return self.extract_from_volume(volume)

    def extract_from_volume(self, volume: np.ndarray) -> List[Image.Image]:
        """
        Extract slices from a 3D numpy array.

        Args:
            volume: 3D numpy array (D, H, W) or (H, W, D)

        Returns:
            List of PIL Images (RGB, output_size x output_size)
        """
        # Get slice indices
        axis = self.axis_map[self.view]
        depth = volume.shape[axis]

        start_idx = int(self.region_start * depth)
        end_idx = int(self.region_end * depth)

        # Generate evenly spaced indices within the region
        indices = np.linspace(start_idx, end_idx, self.n_slices, dtype=int)

        # Extract slices
        slices = []
        for idx in indices:
            # Extract 2D slice based on view
            if axis == 0:
                slice_2d = volume[idx, :, :]
            elif axis == 1:
                slice_2d = volume[:, idx, :]
            else:
                slice_2d = volume[:, :, idx]

            # Process slice
            processed = self._process_slice(slice_2d)
            slices.append(processed)

        return slices

    def _process_slice(self, slice_2d: np.ndarray) -> Image.Image:
        """
        Process a 2D slice: normalize, resize, convert to PIL Image.

        Args:
            slice_2d: 2D numpy array

        Returns:
            PIL Image (RGB, output_size x output_size)
        """
        # Normalize intensity using percentile clipping
        slice_2d = self._normalize_intensity(slice_2d)

        # Resize to output size using bicubic interpolation
        if slice_2d.shape[0] != self.output_size or slice_2d.shape[1] != self.output_size:
            zoom_factors = (
                self.output_size / slice_2d.shape[0],
                self.output_size / slice_2d.shape[1]
            )
            slice_2d = zoom(slice_2d, zoom_factors, order=3)  # bicubic

        # Scale to [0, 255] for PIL Image
        min_val, max_val = self.normalize_range
        slice_2d = (slice_2d - min_val) / (max_val - min_val)  # [0, 1]
        slice_2d = np.clip(slice_2d * 255, 0, 255).astype(np.uint8)

        # Convert to RGB (MedGemma expects RGB)
        slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)

        return Image.fromarray(slice_rgb, mode='RGB')

    def _normalize_intensity(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Normalize slice intensity using percentile clipping.

        Args:
            slice_2d: 2D numpy array

        Returns:
            Normalized array in normalize_range
        """
        # Get non-zero voxels (brain tissue)
        brain_mask = slice_2d > 0

        if brain_mask.sum() > 0:
            brain_voxels = slice_2d[brain_mask]
            p1, p99 = np.percentile(brain_voxels, (1, 99))

            # Clip and normalize
            slice_2d = np.clip(slice_2d, p1, p99)
            slice_2d = (slice_2d - p1) / (p99 - p1 + 1e-8)
        else:
            # Handle empty slices
            slice_2d = np.zeros_like(slice_2d)

        # Scale to target range
        min_val, max_val = self.normalize_range
        slice_2d = slice_2d * (max_val - min_val) + min_val

        return slice_2d


def extract_coronal_slices(
    volume: np.ndarray,
    n_slices: int = 5,
    region_start: float = 0.40,
    region_end: float = 0.60,
    output_size: int = 896
) -> List[Image.Image]:
    """
    Convenience function to extract coronal hippocampus-focused slices.

    Args:
        volume: 3D numpy array
        n_slices: Number of slices to extract
        region_start: Start of hippocampus region (fraction of y-axis)
        region_end: End of hippocampus region (fraction of y-axis)
        output_size: Output image size

    Returns:
        List of PIL Images
    """
    extractor = SliceExtractor(
        view="coronal",
        n_slices=n_slices,
        region_start=region_start,
        region_end=region_end,
        output_size=output_size
    )
    return extractor.extract_from_volume(volume)


class MultiViewSliceExtractor:
    """
    Extracts slices from multiple views (coronal + axial) for comprehensive analysis.
    """

    def __init__(
        self,
        n_coronal: int = 2,
        n_axial: int = 2,
        coronal_region: Tuple[float, float] = (0.40, 0.60),
        axial_region: Tuple[float, float] = (0.40, 0.60),
        output_size: int = 448,
        normalize_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            n_coronal: Number of coronal slices (hippocampus view)
            n_axial: Number of axial slices (horizontal view)
            coronal_region: Start/end fraction for coronal slices
            axial_region: Start/end fraction for axial slices
            output_size: Output image size
            normalize_range: Output intensity range
        """
        self.n_coronal = n_coronal
        self.n_axial = n_axial
        self.output_size = output_size

        # Create extractors for each view
        self.coronal_extractor = SliceExtractor(
            view="coronal",
            n_slices=n_coronal,
            region_start=coronal_region[0],
            region_end=coronal_region[1],
            output_size=output_size,
            normalize_range=normalize_range
        )

        self.axial_extractor = SliceExtractor(
            view="axial",
            n_slices=n_axial,
            region_start=axial_region[0],
            region_end=axial_region[1],
            output_size=output_size,
            normalize_range=normalize_range
        )

    def extract_from_nifti(self, nifti_path) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Extract slices from a NIfTI file.

        Returns:
            Tuple of (coronal_slices, axial_slices)
        """
        import nibabel as nib
        nifti = nib.load(str(nifti_path))
        volume = nifti.get_fdata().astype(np.float32)
        return self.extract_from_volume(volume)

    def extract_from_volume(self, volume: np.ndarray) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Extract slices from a 3D numpy array.

        Returns:
            Tuple of (coronal_slices, axial_slices)
        """
        coronal = self.coronal_extractor.extract_from_volume(volume)
        axial = self.axial_extractor.extract_from_volume(volume)
        return coronal, axial

    def extract_all(self, nifti_path) -> List[Image.Image]:
        """
        Extract all slices as a single list (coronal first, then axial).

        Returns:
            List of PIL Images [coronal_1, coronal_2, ..., axial_1, axial_2, ...]
        """
        coronal, axial = self.extract_from_nifti(nifti_path)
        return coronal + axial

    @property
    def total_slices(self) -> int:
        return self.n_coronal + self.n_axial


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python slice_extractor.py <nifti_path> [output_dir]")
        sys.exit(1)

    nifti_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./slices")
    output_dir.mkdir(exist_ok=True)

    extractor = SliceExtractor()
    slices = extractor.extract_from_nifti(nifti_path)

    print(f"Extracted {len(slices)} slices from {nifti_path}")

    for i, img in enumerate(slices):
        out_path = output_dir / f"slice_{i:02d}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}")
