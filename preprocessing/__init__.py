"""
Medical Image Preprocessing Pipeline

A complete preprocessing pipeline for medical imaging with the following steps:
1. DICOM to NIfTI conversion
2. Image enhancement (MNI registration + N4 bias correction)
3. Skull stripping using SynthStrip

Usage:
    from preprocessing import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline(output_dir, mni_template_path)
    results = pipeline.run_full_pipeline(dicom_root_dir)
"""

from .dicom_to_nifti import convert_dicom_to_nifti, convert_patient_directory
from .image_enhancement import register_to_mni, apply_n4_bias_correction, process_nifti_file, process_directory
from .skull_stripping import synthstrip_skull_strip, skull_strip_directory, setup_synthstrip_docker
from .pipeline import PreprocessingPipeline

__version__ = "1.0.0"
__author__ = "Medical Imaging Pipeline"

__all__ = [
    'PreprocessingPipeline',
    'convert_dicom_to_nifti',
    'convert_patient_directory', 
    'register_to_mni',
    'apply_n4_bias_correction',
    'process_nifti_file',
    'process_directory',
    'synthstrip_skull_strip',
    'skull_strip_directory',
    'setup_synthstrip_docker'
]