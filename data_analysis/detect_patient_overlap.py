#!/usr/bin/env python3
"""
Detect Patient Overlap Across Diagnosis Directories
Check for patients appearing in multiple diagnosis folders (data leakage detection)
"""

import os
import pandas as pd
import re
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_patient_id_from_filename(filename):
    """Extract patient ID from ADNI filename"""
    # ADNI format: ADNI_site_S_subject_...
    # Full patient ID should be: site_S_subject (e.g., 002_S_0619)
    match = re.match(r'ADNI_(\d+)_S_(\d+)_', filename)
    if match:
        site_id = match.group(1)
        subject_id = match.group(2)
        return f"{site_id}_S_{subject_id}"
    return None

def scan_all_patients():
    """Scan all diagnosis directories and find patient overlaps"""
    
    base_path = "/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise"
    
    # Dictionary to track which directories each patient appears in
    patient_diagnoses = defaultdict(set)
    patient_files = defaultdict(list)
    
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_path = os.path.join(base_path, diagnosis)
        
        if not os.path.exists(diagnosis_path):
            logger.warning(f"Directory {diagnosis_path} not found!")
            continue
            
        logger.info(f"Scanning {diagnosis} directory...")
        
        nii_files = [f for f in os.listdir(diagnosis_path) if f.endswith('.nii.gz')]
        logger.info(f"Found {len(nii_files)} files in {diagnosis}")
        
        for filename in nii_files:
            patient_id = extract_patient_id_from_filename(filename)
            
            if patient_id:
                patient_diagnoses[patient_id].add(diagnosis)
                patient_files[patient_id].append({
                    'filename': filename,
                    'diagnosis': diagnosis,
                    'path': os.path.join(diagnosis_path, filename)
                })
    
    return patient_diagnoses, patient_files

def analyze_overlaps(patient_diagnoses, patient_files):
    """Analyze patient overlaps and create detailed report"""
    
    logger.info("Analyzing patient overlaps...")
    
    # Find patients in multiple diagnosis folders
    overlap_patients = {pid: diagnoses for pid, diagnoses in patient_diagnoses.items() 
                       if len(diagnoses) > 1}
    
    logger.info(f"Found {len(overlap_patients)} patients with data leakage!")
    
    # Create detailed analysis
    overlap_analysis = []
    
    for patient_id, diagnoses in overlap_patients.items():
        patient_data = {
            'patient_id': patient_id,
            'diagnoses': list(diagnoses),
            'num_diagnoses': len(diagnoses),
            'files_by_diagnosis': {}
        }
        
        # Get files for this patient in each diagnosis
        for file_info in patient_files[patient_id]:
            diagnosis = file_info['diagnosis']
            if diagnosis not in patient_data['files_by_diagnosis']:
                patient_data['files_by_diagnosis'][diagnosis] = []
            patient_data['files_by_diagnosis'][diagnosis].append(file_info['filename'])
        
        overlap_analysis.append(patient_data)
    
    # Sort by number of diagnoses (most problematic first)
    overlap_analysis.sort(key=lambda x: x['num_diagnoses'], reverse=True)
    
    return overlap_analysis

def create_clean_patient_mapping(patient_diagnoses, patient_files):
    """Create a clean patient mapping strategy"""
    
    logger.info("Creating clean patient mapping strategy...")
    
    clean_patients = {}
    problematic_patients = {}
    
    for patient_id, diagnoses in patient_diagnoses.items():
        if len(diagnoses) == 1:
            # Clean patient - only in one diagnosis folder
            diagnosis = list(diagnoses)[0]
            clean_patients[patient_id] = {
                'diagnosis': diagnosis,
                'files': [f['filename'] for f in patient_files[patient_id]]
            }
        else:
            # Problematic patient - in multiple diagnosis folders
            problematic_patients[patient_id] = {
                'diagnoses': list(diagnoses),
                'files_by_diagnosis': {}
            }
            
            for file_info in patient_files[patient_id]:
                diagnosis = file_info['diagnosis']
                if diagnosis not in problematic_patients[patient_id]['files_by_diagnosis']:
                    problematic_patients[patient_id]['files_by_diagnosis'][diagnosis] = []
                problematic_patients[patient_id]['files_by_diagnosis'][diagnosis].append(file_info['filename'])
    
    logger.info(f"Clean patients: {len(clean_patients)}")
    logger.info(f"Problematic patients: {len(problematic_patients)}")
    
    return clean_patients, problematic_patients

def create_reports(overlap_analysis, clean_patients, problematic_patients):
    """Create detailed reports"""
    
    # Create overlap report
    overlap_records = []
    for patient_data in overlap_analysis:
        for diagnosis in patient_data['diagnoses']:
            files = patient_data['files_by_diagnosis'].get(diagnosis, [])
            for filename in files:
                overlap_records.append({
                    'patient_id': patient_data['patient_id'],
                    'diagnosis': diagnosis,
                    'filename': filename,
                    'total_diagnoses': patient_data['num_diagnoses'],
                    'all_diagnoses': ','.join(patient_data['diagnoses'])
                })
    
    overlap_df = pd.DataFrame(overlap_records)
    overlap_path = "/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/patient_overlap_analysis.csv"
    overlap_df.to_csv(overlap_path, index=False)
    logger.info(f"Overlap analysis saved to: {overlap_path}")
    
    # Create clean patients report
    clean_records = []
    for patient_id, data in clean_patients.items():
        for filename in data['files']:
            clean_records.append({
                'patient_id': patient_id,
                'diagnosis': data['diagnosis'],
                'filename': filename,
                'is_clean': True
            })
    
    clean_df = pd.DataFrame(clean_records)
    clean_path = "/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/clean_patients.csv"
    clean_df.to_csv(clean_path, index=False)
    logger.info(f"Clean patients list saved to: {clean_path}")
    
    # Create summary report
    summary = {
        'total_unique_patients': len(clean_patients) + len(problematic_patients),
        'clean_patients': len(clean_patients),
        'problematic_patients': len(problematic_patients),
        'data_leakage_percentage': len(problematic_patients) / (len(clean_patients) + len(problematic_patients)) * 100
    }
    
    # Count clean patients by diagnosis
    clean_by_diagnosis = defaultdict(int)
    for patient_data in clean_patients.values():
        clean_by_diagnosis[patient_data['diagnosis']] += 1
    
    summary['clean_patients_by_diagnosis'] = dict(clean_by_diagnosis)
    
    return summary, overlap_df, clean_df

def print_detailed_analysis(overlap_analysis, summary):
    """Print detailed analysis to console"""
    
    print("\n" + "="*80)
    print("PATIENT OVERLAP ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nSUMMARY:")
    print(f"  Total unique patients: {summary['total_unique_patients']}")
    print(f"  Clean patients (no overlap): {summary['clean_patients']}")
    print(f"  Problematic patients (overlap): {summary['problematic_patients']}")
    print(f"  Data leakage percentage: {summary['data_leakage_percentage']:.1f}%")
    
    print(f"\nClean patients by diagnosis:")
    for diagnosis, count in summary['clean_patients_by_diagnosis'].items():
        print(f"  {diagnosis}: {count} patients")
    
    print(f"\nMOST PROBLEMATIC PATIENTS (top 10):")
    for i, patient_data in enumerate(overlap_analysis[:10]):
        print(f"\n{i+1}. Patient {patient_data['patient_id']}:")
        print(f"   Appears in: {', '.join(patient_data['diagnoses'])}")
        for diagnosis, files in patient_data['files_by_diagnosis'].items():
            print(f"   {diagnosis}: {len(files)} files")

def suggest_solutions(problematic_patients):
    """Suggest solutions for handling problematic patients"""
    
    print(f"\n" + "="*80)
    print("SUGGESTED SOLUTIONS")
    print("="*80)
    
    print(f"\n1. EXCLUDE PROBLEMATIC PATIENTS:")
    print(f"   - Remove all {len(problematic_patients)} patients with overlaps")
    print(f"   - Use only clean patients for training")
    print(f"   - Ensures no data leakage")
    
    print(f"\n2. USE LONGITUDINAL STRATEGY:")
    print(f"   - Keep patients but assign based on latest scan")
    print(f"   - Or use progression modeling (CN→MCI→AD)")
    print(f"   - Requires careful temporal analysis")
    
    print(f"\n3. PATIENT-LEVEL SPLIT:")
    print(f"   - Ensure no patient appears in both train and test")
    print(f"   - Use patient ID for splitting, not individual scans")
    print(f"   - Recommended approach for medical data")

def main():
    """Main function"""
    
    logger.info("Starting patient overlap detection...")
    
    # Scan all patients
    patient_diagnoses, patient_files = scan_all_patients()
    
    # Analyze overlaps
    overlap_analysis = analyze_overlaps(patient_diagnoses, patient_files)
    
    # Create clean mapping
    clean_patients, problematic_patients = create_clean_patient_mapping(patient_diagnoses, patient_files)
    
    # Create reports
    summary, overlap_df, clean_df = create_reports(overlap_analysis, clean_patients, problematic_patients)
    
    # Print analysis
    print_detailed_analysis(overlap_analysis, summary)
    
    # Suggest solutions
    suggest_solutions(problematic_patients)
    
    print(f"\n" + "="*80)
    print("FILES CREATED:")
    print("="*80)
    print("  - data_analysis/patient_overlap_analysis.csv")
    print("  - data_analysis/clean_patients.csv")
    print("\nNext steps:")
    print("  1. Review the overlap analysis file")
    print("  2. Decide on strategy (exclude overlaps vs longitudinal)")
    print("  3. Regenerate dataset with proper patient-level splits")

if __name__ == "__main__":
    main()