#!/usr/bin/env python3
"""
ADNI Dataset Analysis
Analyzes the ADNI medical imaging dataset for patient counts and MRI distributions
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re

def analyze_adni_directory_structure():
    """Analyze ADNI directory structure"""
    print("=" * 60)
    print("ADNI DATASET ANALYSIS")
    print("=" * 60)
    
    base_path = "/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise"
    
    if not os.path.exists(base_path):
        print(f"ADNI directory not found at {base_path}")
        return None, None
    
    diagnosis_counts = {}
    patient_mri_counts = defaultdict(list)
    all_patients = set()
    
    # Check each diagnosis directory
    for diagnosis in ['AD', 'MCI', 'CN']:
        diagnosis_path = os.path.join(base_path, diagnosis)
        
        if os.path.exists(diagnosis_path):
            print(f"\n{diagnosis} Directory Analysis:")
            print("-" * 30)
            
            # Get all .nii.gz files
            nii_files = [f for f in os.listdir(diagnosis_path) 
                        if f.endswith('.nii.gz')]
            
            print(f"Total MRI files: {len(nii_files)}")
            
            # Extract patient IDs from filenames
            # ADNI filename format: ADNI_site_S_subject_MR_...
            patient_ids = set()
            for filename in nii_files:
                match = re.match(r'ADNI_(\d+)_S_(\d+)_', filename)
                if match:
                    site_id = match.group(1)
                    subject_id = match.group(2)
                    patient_id = f"{site_id}_S_{subject_id}"
                    patient_ids.add(patient_id)
                    all_patients.add(f"{diagnosis}_{patient_id}")
            
            diagnosis_counts[diagnosis] = len(patient_ids)
            print(f"Unique patients: {len(patient_ids)}")
            
            # Count MRIs per patient
            patient_mri_count = defaultdict(int)
            for filename in nii_files:
                match = re.match(r'ADNI_(\d+)_S_(\d+)_', filename)
                if match:
                    site_id = match.group(1)
                    subject_id = match.group(2)
                    patient_id = f"{site_id}_S_{subject_id}"
                    patient_mri_count[patient_id] += 1
            
            mri_counts = list(patient_mri_count.values())
            patient_mri_counts[diagnosis] = mri_counts
            
            if mri_counts:
                print(f"MRIs per patient - Mean: {sum(mri_counts)/len(mri_counts):.1f}")
                print(f"MRIs per patient - Median: {sorted(mri_counts)[len(mri_counts)//2]}")
                print(f"MRIs per patient - Range: {min(mri_counts)} - {max(mri_counts)}")
                
                # Show some example patients
                print(f"Sample patients:")
                for i, (pid, count) in enumerate(list(patient_mri_count.items())[:5]):
                    print(f"  {pid}: {count} MRIs")
                if len(patient_mri_count) > 5:
                    print(f"  ... and {len(patient_mri_count) - 5} more patients")
        else:
            print(f"{diagnosis} directory not found")
            diagnosis_counts[diagnosis] = 0
    
    return diagnosis_counts, patient_mri_counts

def analyze_csv_files():
    """Analyze CSV files in ADNI directory"""
    print("\n" + "=" * 60)
    print("CSV FILES ANALYSIS")
    print("=" * 60)
    
    base_path = "/Users/tanguyvans/Desktop/umons/alzheimer/ADNIDenoise"
    csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        print(f"\n{csv_file}:")
        print("-" * 40)
        
        try:
            df = pd.read_csv(os.path.join(base_path, csv_file))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for diagnosis column
            diagnosis_cols = [col for col in df.columns 
                            if 'diagnosis' in col.lower() or 'group' in col.lower() 
                            or 'label' in col.lower() or 'class' in col.lower()]
            
            if diagnosis_cols:
                for col in diagnosis_cols:
                    print(f"Diagnosis distribution ({col}):")
                    print(df[col].value_counts().to_string())
            
            print(f"First few rows:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

def create_visualizations(diagnosis_counts, patient_mri_counts):
    """Create visualizations"""
    
    if not diagnosis_counts or not patient_mri_counts:
        print("No data to visualize")
        return
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ADNI Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Patient counts by diagnosis
    diagnoses = list(diagnosis_counts.keys())
    counts = list(diagnosis_counts.values())
    
    axes[0, 0].bar(diagnoses, counts, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0, 0].set_title('Number of Patients by Diagnosis')
    axes[0, 0].set_ylabel('Number of Patients')
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + 1, str(v), ha='center', va='bottom')
    
    # 2. Patient distribution pie chart
    axes[0, 1].pie(counts, labels=diagnoses, autopct='%1.1f%%', startangle=90,
                   colors=['#ff7f0e', '#2ca02c', '#1f77b4'])
    axes[0, 1].set_title('Patient Distribution')
    
    # 3. MRIs per patient distribution
    all_mri_counts = []
    mri_labels = []
    for diagnosis, mri_counts in patient_mri_counts.items():
        all_mri_counts.extend(mri_counts)
        mri_labels.extend([diagnosis] * len(mri_counts))
    
    if all_mri_counts:
        mri_df = pd.DataFrame({'mri_count': all_mri_counts, 'diagnosis': mri_labels})
        sns.boxplot(data=mri_df, x='diagnosis', y='mri_count', ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of MRIs per Patient')
        axes[1, 0].set_ylabel('Number of MRIs per Patient')
    
    # 4. Total MRIs by diagnosis
    total_mris = {}
    for diagnosis, mri_counts in patient_mri_counts.items():
        total_mris[diagnosis] = sum(mri_counts)
    
    if total_mris:
        axes[1, 1].bar(total_mris.keys(), total_mris.values(), 
                       color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        axes[1, 1].set_title('Total MRIs by Diagnosis')
        axes[1, 1].set_ylabel('Total Number of MRIs')
        for i, (k, v) in enumerate(total_mris.items()):
            axes[1, 1].text(i, v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/adni_analysis.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'adni_analysis.png'")

def create_summary_report(diagnosis_counts, patient_mri_counts):
    """Create summary report"""
    
    report_path = '/Users/tanguyvans/Desktop/umons/alzheimer/data_analysis/adni_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("ADNI DATASET ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PATIENT SUMMARY:\n")
        total_patients = sum(diagnosis_counts.values())
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / total_patients * 100) if total_patients > 0 else 0
            f.write(f"  {diagnosis}: {count} patients ({percentage:.1f}%)\n")
        f.write(f"  Total: {total_patients} patients\n\n")
        
        f.write("MRI SUMMARY:\n")
        total_mris = 0
        for diagnosis, mri_counts in patient_mri_counts.items():
            diagnosis_total = sum(mri_counts)
            total_mris += diagnosis_total
            avg_per_patient = diagnosis_total / len(mri_counts) if mri_counts else 0
            f.write(f"  {diagnosis}: {diagnosis_total} MRIs (avg {avg_per_patient:.1f} per patient)\n")
        f.write(f"  Total: {total_mris} MRIs\n\n")
        
        f.write("DETAILED MRI STATISTICS:\n")
        for diagnosis, mri_counts in patient_mri_counts.items():
            if mri_counts:
                f.write(f"\n{diagnosis}:\n")
                f.write(f"  Patients: {len(mri_counts)}\n")
                f.write(f"  Total MRIs: {sum(mri_counts)}\n")
                f.write(f"  Mean MRIs per patient: {sum(mri_counts)/len(mri_counts):.1f}\n")
                f.write(f"  Median MRIs per patient: {sorted(mri_counts)[len(mri_counts)//2]}\n")
                f.write(f"  Min MRIs per patient: {min(mri_counts)}\n")
                f.write(f"  Max MRIs per patient: {max(mri_counts)}\n")
    
    print(f"Detailed report saved as 'adni_report.txt'")

def main():
    # Analyze directory structure
    diagnosis_counts, patient_mri_counts = analyze_adni_directory_structure()
    
    # Analyze CSV files
    analyze_csv_files()
    
    # Create visualizations and report
    if diagnosis_counts and patient_mri_counts:
        create_visualizations(diagnosis_counts, patient_mri_counts)
        create_summary_report(diagnosis_counts, patient_mri_counts)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total_patients = sum(diagnosis_counts.values())
        total_mris = sum(sum(mri_counts) for mri_counts in patient_mri_counts.values())
        
        print(f"Total patients: {total_patients}")
        print(f"Total MRI scans: {total_mris}")
        print(f"Average MRIs per patient: {total_mris/total_patients:.1f}")
        
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / total_patients * 100) if total_patients > 0 else 0
            mri_count = sum(patient_mri_counts[diagnosis])
            avg_mri = mri_count / count if count > 0 else 0
            print(f"{diagnosis}: {count} patients ({percentage:.1f}%), {mri_count} MRIs (avg {avg_mri:.1f} per patient)")

if __name__ == "__main__":
    main()