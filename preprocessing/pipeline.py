#!/usr/bin/env python3
"""
Complete preprocessing pipeline for medical imaging:
1. DICOM to NIfTI conversion
2. Image enhancement (MNI registration + N4 bias correction)
3. Skull stripping using SynthStrip

Usage:
    python pipeline.py --input /path/to/dicom/root --output /path/to/output --template /path/to/mni/template.nii
    python pipeline.py --nifti-input /path/to/nifti/files --output /path/to/output --template /path/to/mni/template.nii
"""

import os
import argparse
import logging
import time
from typing import Dict, List
from tqdm import tqdm
import warnings

# Import our preprocessing modules
from dicom_to_nifti import convert_patient_directory
from image_enhancement import process_directory, process_nifti_file
from skull_stripping import skull_strip_directory, setup_synthstrip_docker

# Disable warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Complete preprocessing pipeline for medical imaging."""
    
    def __init__(self, output_root: str, mni_template: str):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            output_root: Root directory for all outputs
            mni_template: Path to MNI template file
        """
        self.output_root = output_root
        self.mni_template = mni_template
        
        # Create output subdirectories
        self.nifti_dir = os.path.join(output_root, "01_nifti")
        self.processed_dir = os.path.join(output_root, "02_processed") 
        self.skull_stripped_dir = os.path.join(output_root, "03_skull_stripped")
        
        # Create directories
        for dir_path in [self.nifti_dir, self.processed_dir, self.skull_stripped_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_dicom_conversion(self, dicom_root: str, patient_id: str = None) -> List[str]:
        """
        Step 1: Convert DICOM files to NIfTI format.
        
        Args:
            dicom_root: Root directory containing patient DICOM folders
            patient_id: Specific patient to process (optional)
            
        Returns:
            List of created NIfTI files
        """
        logger.info("="*60)
        logger.info("STEP 1: DICOM TO NIFTI CONVERSION")
        logger.info("="*60)
        
        start_time = time.time()
        
        nifti_files = convert_patient_directory(dicom_root, self.nifti_dir, patient_id)
        
        elapsed_time = time.time() - start_time
        logger.info(f"DICOM conversion completed: {len(nifti_files)} files created in {elapsed_time/60:.2f} minutes")
        
        return nifti_files
    
    def run_processing(self, nifti_files: List[str] = None) -> Dict:
        """
        Step 2: N4 bias correction + MNI registration.
        
        Args:
            nifti_files: List of NIfTI files to process (if None, process all in nifti_dir)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("="*60)
        logger.info("STEP 2: N4 BIAS CORRECTION + MNI REGISTRATION")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Check MNI template exists
        if not os.path.exists(self.mni_template):
            logger.error(f"MNI template not found: {self.mni_template}")
            return {'successful': [], 'failed': [], 'total': 0, 'error': 'template_not_found'}
        
        if nifti_files:
            # Process specific files
            results = {'successful': [], 'failed': [], 'total': len(nifti_files)}
            
            with tqdm(total=len(nifti_files), desc="Processing images") as pbar:
                for nifti_file in nifti_files:
                    filename = os.path.basename(nifti_file)
                    subject_id = filename.replace('.nii.gz', '')
                    output_file = os.path.join(self.processed_dir, f"{subject_id}_processed.nii.gz")
                    
                    pbar.set_description(f"Processing {subject_id}")
                    
                    # Skip if already exists
                    if os.path.exists(output_file):
                        logger.info(f"Processed file already exists: {output_file}")
                        results['successful'].append(subject_id)
                    else:
                        success = process_nifti_file(nifti_file, output_file, self.mni_template, subject_id)
                        if success:
                            results['successful'].append(subject_id)
                        else:
                            results['failed'].append(subject_id)
                    
                    pbar.update(1)
        else:
            # Process all files in nifti directory
            results = process_directory(self.nifti_dir, self.processed_dir, self.mni_template)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed: {len(results['successful'])} successful, {len(results['failed'])} failed in {elapsed_time/60:.2f} minutes")
        
        return results
    
    def run_skull_stripping(self) -> Dict:
        """
        Step 3: Skull stripping using SynthStrip.
        
        Returns:
            Dictionary with processing results
        """
        logger.info("="*60)
        logger.info("STEP 3: SKULL STRIPPING")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Setup Docker environment
        if not setup_synthstrip_docker():
            logger.error("Failed to setup SynthStrip Docker environment")
            return {'successful': [], 'failed': [], 'total': 0, 'error': 'docker_setup_failed'}
        
        # Use processed files if available, otherwise use original NIfTI files
        if os.path.exists(self.processed_dir) and os.listdir(self.processed_dir):
            input_dir = self.processed_dir
            file_pattern = "*_processed.nii.gz"
        else:
            logger.info("No processed files found, using original NIfTI files for skull stripping")
            input_dir = self.nifti_dir
            file_pattern = "*.nii.gz"
        
        results = skull_strip_directory(input_dir, self.skull_stripped_dir, file_pattern)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Skull stripping completed: {len(results['successful'])} successful, {len(results['failed'])} failed in {elapsed_time/60:.2f} minutes")
        
        return results
    
    def run_full_pipeline(self, dicom_root: str = None, nifti_input: str = None, patient_id: str = None) -> Dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            dicom_root: Root directory containing DICOM files (for full pipeline)
            nifti_input: Directory containing NIfTI files (to skip DICOM conversion)
            patient_id: Specific patient to process
            
        Returns:
            Dictionary with complete processing results
        """
        logger.info("="*80)
        logger.info("STARTING COMPLETE PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        pipeline_start_time = time.time()
        results = {}
        
        # Step 1: DICOM to NIfTI conversion (if needed)
        nifti_files = []
        if dicom_root:
            nifti_files = self.run_dicom_conversion(dicom_root, patient_id)
            results['dicom_conversion'] = {'files_created': len(nifti_files)}
        elif nifti_input:
            # Copy NIfTI files to our pipeline structure
            import shutil
            import glob
            
            logger.info("Copying NIfTI files to pipeline structure...")
            source_files = glob.glob(os.path.join(nifti_input, "*.nii.gz"))
            
            for source_file in source_files:
                dest_file = os.path.join(self.nifti_dir, os.path.basename(source_file))
                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    nifti_files.append(dest_file)
                else:
                    nifti_files.append(dest_file)
            
            logger.info(f"Prepared {len(nifti_files)} NIfTI files for processing")
            results['dicom_conversion'] = {'files_prepared': len(nifti_files)}
        
        # Step 2: N4 bias correction + MNI registration
        processing_results = self.run_processing(nifti_files if nifti_files else None)
        results['processing'] = processing_results
        
        # Step 3: Skull stripping
        skull_stripping_results = self.run_skull_stripping()
        results['skull_stripping'] = skull_stripping_results
        
        # Final summary
        total_time = time.time() - pipeline_start_time
        logger.info("="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total processing time: {total_time/3600:.2f} hours")
        
        if 'processing' in results:
            logger.info(f"Processed images (N4 + MNI): {len(results['processing']['successful'])}")
        if 'skull_stripping' in results:
            logger.info(f"Skull-stripped images: {len(results['skull_stripping']['successful'])}")
        
        logger.info(f"Final outputs in: {self.skull_stripped_dir}")
        logger.info("="*80)
        
        results['total_time_hours'] = total_time / 3600
        
        return results

def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Medical imaging preprocessing pipeline')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '--dicom-input', 
                           help='Root directory containing patient DICOM folders')
    input_group.add_argument('--nifti-input', 
                           help='Directory containing NIfTI files (skip DICOM conversion)')
    
    # Required arguments
    parser.add_argument('--output', required=True,
                       help='Output root directory')
    parser.add_argument('--template', '--mni-template', required=True,
                       help='Path to MNI template file')
    
    # Optional arguments
    parser.add_argument('--patient-id', 
                       help='Specific patient ID to process')
    parser.add_argument('--step', choices=['dicom', 'process', 'skull', 'all'], 
                       default='all',
                       help='Specific step to run (default: all)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(args.output, args.template)
    
    # Run specified steps
    if args.step == 'all':
        pipeline.run_full_pipeline(
            dicom_root=args.input,
            nifti_input=args.nifti_input,
            patient_id=args.patient_id
        )
    elif args.step == 'dicom':
        if not args.input:
            logger.error("--input is required for DICOM conversion step")
            return
        pipeline.run_dicom_conversion(args.input, args.patient_id)
    elif args.step == 'process':
        pipeline.run_processing()
    elif args.step == 'skull':
        pipeline.run_skull_stripping()

if __name__ == '__main__':
    main()