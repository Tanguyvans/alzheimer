#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module to avoid __init__.py issues
import preprocessing.dicom_to_nifti as dcm2nii
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_mprage_sequences(input_dir, output_dir):
    """Convert only MPRAGE sequences (any folder containing MPRAGE or MP-RAGE)."""
    
    print("Converting sequences containing 'MPRAGE' or 'MP-RAGE'")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Output structure: {output_dir}/patient_id/sequence_date_series.nii.gz\n")
    
    os.makedirs(output_dir, exist_ok=True)
    all_output_files = []
    processed_sequences = defaultdict(int)
    skipped_sequences = defaultdict(int)
    patients_processed = set()
    
    # Get all patient directories
    patient_dirs = [d for d in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]
    
    print(f"Found {len(patient_dirs)} patient directories\n")
    
    for i, patient in enumerate(patient_dirs):
        patient_path = os.path.join(input_dir, patient)
        patient_has_mprage = False
        
        try:
            sequence_dirs = [d for d in os.listdir(patient_path) 
                           if os.path.isdir(os.path.join(patient_path, d))]
        except FileNotFoundError:
            logging.warning(f"Cannot access patient directory: {patient_path}")
            continue
        
        for sequence in sequence_dirs:
            # Check if sequence name contains MPRAGE (case-insensitive)
            if 'MPRAGE' not in sequence.upper() and 'MP-RAGE' not in sequence.upper() and 'MP_RAGE' not in sequence.upper():
                skipped_sequences[sequence] += 1
                continue
            
            processed_sequences[sequence] += 1
            sequence_path = os.path.join(patient_path, sequence)
            
            try:
                date_dirs = [d for d in os.listdir(sequence_path) 
                           if os.path.isdir(os.path.join(sequence_path, d))]
            except FileNotFoundError:
                logging.warning(f"Cannot access sequence directory: {sequence_path}")
                continue
            
            for date_dir in date_dirs:
                date_path = os.path.join(sequence_path, date_dir)
                
                # Check for DICOM files directly in date directory
                dicom_files = [f for f in os.listdir(date_path) 
                             if f.lower().endswith('.dcm') and not f.startswith('.')]
                
                if dicom_files:
                    timepoint = f"{sequence}_{date_dir}"
                    logging.info(f"Converting {patient}/{sequence}/{date_dir} ({len(dicom_files)} files)")
                    output_files = dcm2nii.convert_dicom_to_nifti(date_path, output_dir, patient, timepoint)
                    all_output_files.extend(output_files)
                    if output_files:
                        patient_has_mprage = True
                else:
                    # Check for subdirectories (ADNI's I##### folders)
                    try:
                        series_dirs = [d for d in os.listdir(date_path) 
                                     if os.path.isdir(os.path.join(date_path, d))]
                        
                        for series_dir in series_dirs:
                            series_path = os.path.join(date_path, series_dir)
                            series_dicom_files = [f for f in os.listdir(series_path) 
                                                if f.lower().endswith('.dcm') and not f.startswith('.')]
                            
                            if series_dicom_files:
                                timepoint = f"{sequence}_{date_dir}_{series_dir}"
                                logging.info(f"Converting {patient}/{sequence}/{date_dir}/{series_dir} ({len(series_dicom_files)} files)")
                                output_files = dcm2nii.convert_dicom_to_nifti(series_path, output_dir, patient, timepoint)
                                all_output_files.extend(output_files)
                                if output_files:
                                    patient_has_mprage = True
                    except FileNotFoundError:
                        logging.warning(f"Cannot access date directory: {date_path}")
                        continue
        
        if patient_has_mprage:
            patients_processed.add(patient)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(patient_dirs)} patients checked")
            print(f"  - Patients with MPRAGE: {len(patients_processed)}")
            print(f"  - NIfTI files created: {len(all_output_files)}\n")
    
    # Final summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Total patients processed: {len(patients_processed)}/{len(patient_dirs)}")
    print(f"Total NIfTI files created: {len(all_output_files)}")
    
    print(f"\nMPRAGE sequences found:")
    for seq, count in sorted(processed_sequences.items(), key=lambda x: x[1], reverse=True):
        print(f"  {seq}: {count} occurrences")
    
    if len(skipped_sequences) > 0:
        print(f"\nTop skipped (non-MPRAGE) sequences:")
        for seq, count in sorted(skipped_sequences.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {seq}: {count} occurrences")
    
    return all_output_files

if __name__ == "__main__":
    input_dir = "/Volumes/KINGSTON/ADNI"
    output_dir = "/Volumes/KINGSTON/ADNI_nifti"
    
    print("="*50)
    print("ADNI DICOM to NIfTI Converter (MPRAGE only)")
    print("="*50)
    
    result = convert_mprage_sequences(input_dir, output_dir)
    
    if result and len(result) > 0:
        print(f"\nSample output files created:")
        # Show a few examples to verify the structure
        for file in result[:5]:
            # Show relative path from output_dir
            rel_path = os.path.relpath(file, output_dir)
            print(f"  {rel_path}")