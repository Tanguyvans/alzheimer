#!/usr/bin/env python3
"""
ADNI dxsum.csv Analysis Script

A well-structured script to analyze the ADNI diagnosis summary data (dxsum.csv).
Generates comprehensive text reports and visualization charts.

Author: ADNI Analysis Pipeline
Date: 2025-10-09
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DxsumAnalyzer:
    """
    Comprehensive analyzer for ADNI diagnosis summary data.
    
    This class provides modular analysis of patient diagnoses, transitions,
    and progression patterns from the dxsum.csv file.
    """
    
    # Diagnosis code mapping
    DIAGNOSIS_MAP = {
        1: 'CN',   # Cognitively Normal
        2: 'MCI',  # Mild Cognitive Impairment
        3: 'AD'    # Alzheimer's Disease
    }
    
    # Color scheme for visualizations
    COLORS = {
        'CN': '#2ca02c',   # Green
        'MCI': '#1f77b4',  # Blue
        'AD': '#ff7f0e'    # Orange
    }
    
    def __init__(self, csv_path: str, output_dir: str, mri_csv_path: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            csv_path: Path to dxsum.csv file
            output_dir: Directory for output files
            mri_csv_path: Optional path to MRI metadata CSV file
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.mri_csv_path = mri_csv_path
        self.df = None
        self.mri_df = None
        
        # Analysis results storage
        self.stats = {}
        self.patient_trajectories = {}
        self.transition_details = []
        self.baseline_outcomes = {}
        self.mri_analysis = {}
        self.fluctuation_patterns = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self) -> bool:
        """
        Load and preprocess the dxsum.csv data and optionally MRI data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 80)
        print("DXSUM.CSV ANALYSIS - DATA LOADING")
        print("=" * 80)
        
        if not os.path.exists(self.csv_path):
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        
        try:
            # Load diagnosis data
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded {len(self.df)} records from dxsum.csv")
            
            # Convert dates
            self.df['EXAMDATE'] = pd.to_datetime(self.df['EXAMDATE'], errors='coerce')
            
            # Sort by patient and date
            self.df = self.df.sort_values(['RID', 'EXAMDATE'])
            
            # Basic info
            print(f"\nDataset Overview:")
            print(f"  Total records: {len(self.df)}")
            print(f"  Unique patients (RID): {self.df['RID'].nunique()}")
            print(f"  Date range: {self.df['EXAMDATE'].min()} to {self.df['EXAMDATE'].max()}")
            print(f"  Columns: {len(self.df.columns)}")
            
            # Load MRI data if path provided
            if self.mri_csv_path and os.path.exists(self.mri_csv_path):
                print(f"\n📊 Loading MRI metadata...")
                self.mri_df = pd.read_csv(self.mri_csv_path)
                print(f"✅ Loaded {len(self.mri_df)} MRI scans")
                print(f"  Unique subjects in MRI data: {self.mri_df['subject_id'].nunique()}")
            elif self.mri_csv_path:
                print(f"⚠️  MRI file not found at {self.mri_csv_path}")
                print(f"   Continuing without MRI analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_basic_stats(self) -> Dict:
        """
        Analyze basic statistics of the dataset.
        
        Returns:
            Dict containing basic statistics
        """
        print("\n" + "=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)
        
        stats = {
            'total_records': len(self.df),
            'total_patients': self.df['RID'].nunique(),
            'date_range': (self.df['EXAMDATE'].min(), self.df['EXAMDATE'].max()),
            'visits_per_patient': self.df.groupby('RID').size()
        }
        
        # Visit statistics
        visits = stats['visits_per_patient']
        print(f"\nVisit Statistics:")
        print(f"  Total unique patients: {stats['total_patients']}")
        print(f"  Mean visits per patient: {visits.mean():.2f}")
        print(f"  Median visits per patient: {visits.median():.1f}")
        print(f"  Min visits: {visits.min()}")
        print(f"  Max visits: {visits.max()}")
        print(f"  Std deviation: {visits.std():.2f}")
        
        # Visit distribution
        single_visit = len(visits[visits == 1])
        multi_visit = len(visits[visits > 1])
        print(f"\n  Patients with 1 visit: {single_visit} ({single_visit/len(visits)*100:.1f}%)")
        print(f"  Patients with 2+ visits: {multi_visit} ({multi_visit/len(visits)*100:.1f}%)")
        
        self.stats.update(stats)
        return stats
    
    def analyze_diagnosis_distribution(self) -> Dict:
        """
        Analyze diagnosis distribution across all visits and at baseline/final.
        
        Returns:
            Dict containing diagnosis distribution data
        """
        print("\n" + "=" * 80)
        print("DIAGNOSIS DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Overall distribution (all visits)
        print("\n📊 Overall Diagnosis Distribution (All Records):")
        print("-" * 50)
        overall_dist = self.df['DIAGNOSIS'].value_counts().sort_index()
        
        for dx_code, count in overall_dist.items():
            label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
            percentage = (count / len(self.df)) * 100
            print(f"  {label:10s}: {count:5d} records ({percentage:5.1f}%)")
        
        # Baseline and final distributions
        baseline_dx = []
        final_dx = []
        
        for rid in self.df['RID'].unique():
            patient_data = self.df[self.df['RID'] == rid]
            diagnoses = patient_data['DIAGNOSIS'].dropna().tolist()
            
            if len(diagnoses) > 0:
                baseline_dx.append(diagnoses[0])
                final_dx.append(diagnoses[-1])
        
        # Baseline distribution
        print("\n📊 Baseline Diagnosis Distribution (First Visit):")
        print("-" * 50)
        baseline_counts = pd.Series(baseline_dx).value_counts().sort_index()
        
        for dx_code, count in baseline_counts.items():
            label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
            percentage = (count / len(baseline_dx)) * 100
            print(f"  {label:10s}: {count:5d} patients ({percentage:5.1f}%)")
        
        # Final distribution
        print("\n📊 Final Diagnosis Distribution (Last Visit):")
        print("-" * 50)
        final_counts = pd.Series(final_dx).value_counts().sort_index()
        
        for dx_code, count in final_counts.items():
            label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
            percentage = (count / len(final_dx)) * 100
            print(f"  {label:10s}: {count:5d} patients ({percentage:5.1f}%)")
        
        distribution_data = {
            'overall': overall_dist.to_dict(),
            'baseline': baseline_counts.to_dict(),
            'final': final_counts.to_dict()
        }
        
        self.stats['distribution'] = distribution_data
        return distribution_data
    
    def analyze_transitions(self) -> Tuple[Dict, List]:
        """
        Analyze diagnosis transitions for each patient.
        
        Returns:
            Tuple of (patient_trajectories dict, transition_details list)
        """
        print("\n" + "=" * 80)
        print("DIAGNOSIS TRANSITION ANALYSIS")
        print("=" * 80)
        
        patient_trajectories = {}
        transition_details = []
        
        # Analyze each patient
        for rid in self.df['RID'].unique():
            patient_data = self.df[self.df['RID'] == rid].copy()
            diagnoses = patient_data['DIAGNOSIS'].dropna().tolist()
            dates = patient_data['EXAMDATE'].tolist()
            
            if len(diagnoses) == 0:
                continue
            
            # Store patient trajectory
            trajectory = {
                'rid': rid,
                'ptid': patient_data.iloc[0]['PTID'],
                'num_visits': len(patient_data),
                'diagnoses': diagnoses,
                'dates': dates,
                'first_dx': diagnoses[0],
                'last_dx': diagnoses[-1],
                'changed': len(set(diagnoses)) > 1,
                'num_changes': len([i for i in range(len(diagnoses)-1) 
                                   if diagnoses[i] != diagnoses[i+1]]),
                'progression': len(set(diagnoses)) > 1 and diagnoses[-1] > diagnoses[0],
                'improvement': len(set(diagnoses)) > 1 and diagnoses[-1] < diagnoses[0],
                'fluctuated': len(set(diagnoses)) > 1 and diagnoses[-1] == diagnoses[0]
            }
            
            patient_trajectories[rid] = trajectory
            
            # Track individual transitions
            for i in range(len(diagnoses) - 1):
                if diagnoses[i] != diagnoses[i+1]:
                    days_between = None
                    if pd.notna(dates[i]) and pd.notna(dates[i+1]):
                        days_between = (dates[i+1] - dates[i]).days
                    
                    transition_details.append({
                        'rid': rid,
                        'from_dx': diagnoses[i],
                        'to_dx': diagnoses[i+1],
                        'from_label': self.DIAGNOSIS_MAP.get(diagnoses[i], 'Unknown'),
                        'to_label': self.DIAGNOSIS_MAP.get(diagnoses[i+1], 'Unknown'),
                        'date_from': dates[i],
                        'date_to': dates[i+1],
                        'days_between': days_between
                    })
        
        # Summary statistics
        total_patients = len(patient_trajectories)
        changed = sum(1 for t in patient_trajectories.values() if t['changed'])
        progressed = sum(1 for t in patient_trajectories.values() if t['progression'])
        improved = sum(1 for t in patient_trajectories.values() if t['improvement'])
        fluctuated = sum(1 for t in patient_trajectories.values() if t['fluctuated'])
        stable = total_patients - changed
        
        print(f"\n📊 Transition Summary:")
        print(f"  Total patients: {total_patients}")
        print(f"  Stable diagnosis: {stable} ({stable/total_patients*100:.1f}%)")
        print(f"  Changed diagnosis: {changed} ({changed/total_patients*100:.1f}%)")
        print(f"    → Disease progression: {progressed} ({progressed/total_patients*100:.1f}%)")
        print(f"    → Improvement: {improved} ({improved/total_patients*100:.1f}%)")
        print(f"    → Fluctuated (returned to baseline): {fluctuated} ({fluctuated/total_patients*100:.1f}%)")
        
        # Count specific transitions
        print(f"\n🔄 Transition Frequencies:")
        print("-" * 50)
        transition_counts = Counter()
        for t in transition_details:
            key = f"{t['from_label']} → {t['to_label']}"
            transition_counts[key] += 1
        
        for transition, count in transition_counts.most_common():
            percentage = (count / len(transition_details)) * 100 if transition_details else 0
            print(f"  {transition:12s}: {count:4d} transitions ({percentage:5.1f}%)")
        
        # Time analysis
        print(f"\n⏱️  Average Time to Transition:")
        print("-" * 50)
        transition_times = defaultdict(list)
        for t in transition_details:
            if t['days_between'] is not None and t['days_between'] > 0:
                key = f"{t['from_label']} → {t['to_label']}"
                transition_times[key].append(t['days_between'])
        
        for transition in ['CN → MCI', 'CN → AD', 'MCI → AD', 'MCI → CN']:
            if transition in transition_times and transition_times[transition]:
                times = transition_times[transition]
                avg_years = np.mean(times) / 365.25
                median_years = np.median(times) / 365.25
                print(f"  {transition:12s}: {avg_years:.1f} years (median: {median_years:.1f} years)")
        
        self.patient_trajectories = patient_trajectories
        self.transition_details = transition_details
        
        return patient_trajectories, transition_details
    
    def analyze_fluctuation_patterns(self) -> Dict:
        """
        Analyze detailed patterns for patients who fluctuated (returned to baseline).
        
        Returns:
            Dict containing fluctuation pattern data
        """
        print("\n" + "=" * 80)
        print("FLUCTUATION PATTERN ANALYSIS")
        print("=" * 80)
        
        fluctuation_patterns = defaultdict(list)
        
        # Analyze each fluctuated patient
        for rid, trajectory in self.patient_trajectories.items():
            if trajectory['fluctuated']:
                diagnoses = trajectory['diagnoses']
                
                # Create diagnosis path string
                dx_labels = [self.DIAGNOSIS_MAP.get(dx, 'Unknown') for dx in diagnoses]
                
                # For pattern analysis, simplify to unique transitions
                # e.g., [CN, CN, MCI, MCI, CN] -> "CN → MCI → CN"
                simplified_path = [dx_labels[0]]
                for i in range(1, len(dx_labels)):
                    if dx_labels[i] != dx_labels[i-1]:
                        simplified_path.append(dx_labels[i])
                
                pattern = " → ".join(simplified_path)
                
                fluctuation_patterns[pattern].append({
                    'rid': rid,
                    'ptid': trajectory['ptid'],
                    'num_visits': trajectory['num_visits'],
                    'full_path': dx_labels,
                    'num_transitions': trajectory['num_changes']
                })
        
        # Display results
        print(f"\n📊 Fluctuation Patterns (Patients who returned to baseline):")
        print(f"Total fluctuated patients: {sum(len(v) for v in fluctuation_patterns.values())}")
        print("=" * 70)
        
        # Sort patterns by frequency
        sorted_patterns = sorted(fluctuation_patterns.items(), 
                                key=lambda x: len(x[1]), reverse=True)
        
        for pattern, patients in sorted_patterns:
            count = len(patients)
            avg_visits = np.mean([p['num_visits'] for p in patients])
            print(f"\n{pattern}")
            print(f"  Count: {count} patients")
            print(f"  Average visits: {avg_visits:.1f}")
            print(f"  Average transitions: {np.mean([p['num_transitions'] for p in patients]):.1f}")
            
            # Show a few example patient IDs
            if count <= 3:
                for p in patients:
                    print(f"    • {p['ptid']} ({p['num_visits']} visits)")
        
        # Detailed breakdown by transition type
        print(f"\n" + "=" * 70)
        print("📊 Fluctuation by Baseline Diagnosis:")
        print("=" * 70)
        
        cn_fluct = sum(len(patients) for pattern, patients in fluctuation_patterns.items() 
                      if pattern.startswith('CN'))
        mci_fluct = sum(len(patients) for pattern, patients in fluctuation_patterns.items() 
                       if pattern.startswith('MCI'))
        ad_fluct = sum(len(patients) for pattern, patients in fluctuation_patterns.items() 
                      if pattern.startswith('AD'))
        
        print(f"  CN baseline fluctuations: {cn_fluct}")
        print(f"  MCI baseline fluctuations: {mci_fluct}")
        print(f"  AD baseline fluctuations: {ad_fluct}")
        
        self.fluctuation_patterns = dict(fluctuation_patterns)
        return self.fluctuation_patterns
    
    def analyze_progression_by_baseline(self) -> Dict:
        """
        Analyze progression outcomes by baseline diagnosis.
        
        Returns:
            Dict containing baseline outcome data
        """
        print("\n" + "=" * 80)
        print("PROGRESSION ANALYSIS BY BASELINE DIAGNOSIS")
        print("=" * 80)
        
        baseline_outcomes = defaultdict(lambda: {
            'total': 0, 'to_CN': 0, 'to_MCI': 0, 'to_AD': 0
        })
        
        # Count outcomes for each baseline
        for rid, trajectory in self.patient_trajectories.items():
            if len(trajectory['diagnoses']) < 2:
                continue
            
            first_label = self.DIAGNOSIS_MAP.get(trajectory['first_dx'], 'Unknown')
            last_label = self.DIAGNOSIS_MAP.get(trajectory['last_dx'], 'Unknown')
            
            baseline_outcomes[first_label]['total'] += 1
            baseline_outcomes[first_label][f'to_{last_label}'] += 1
        
        # Display results
        print("\n📊 Outcomes by Baseline Diagnosis:")
        print("=" * 70)
        
        for baseline in ['CN', 'MCI', 'AD']:
            if baseline in baseline_outcomes:
                data = baseline_outcomes[baseline]
                total = data['total']
                
                print(f"\n{baseline} Patients (n={total}):")
                print("-" * 60)
                
                for outcome in ['CN', 'MCI', 'AD']:
                    count = data[f'to_{outcome}']
                    percentage = (count / total * 100) if total > 0 else 0
                    bar = '█' * int(percentage / 3)
                    status = '[STABLE]' if baseline == outcome else '[CHANGED]'
                    print(f"  → {outcome:5s}: {count:4d} ({percentage:5.1f}%) {bar} {status}")
        
        self.baseline_outcomes = dict(baseline_outcomes)
        return self.baseline_outcomes
    
    def analyze_mri_by_stability(self) -> Dict:
        """
        Analyze MRI counts by diagnosis stability status.
        
        Returns:
            Dict containing MRI analysis data
        """
        print("\n" + "=" * 80)
        print("MRI ANALYSIS BY DIAGNOSIS STABILITY")
        print("=" * 80)
        
        if self.mri_df is None:
            print("⚠️  No MRI data available - skipping MRI analysis")
            return {}
        
        # Count MRIs per patient
        mri_counts = self.mri_df.groupby('subject_id').size().to_dict()
        
        # Map PTID to subject_id format (e.g., "002_S_0295")
        # PTID format: "002_S_0295", subject_id format: "002_S_0295"
        
        # Categorize patients by stability
        stable_mri_counts = []
        changed_mri_counts = []
        progressed_mri_counts = []
        improved_mri_counts = []
        
        matched_count = 0
        unmatched_count = 0
        
        for rid, trajectory in self.patient_trajectories.items():
            ptid = trajectory['ptid']
            
            # Try to find MRI count for this patient
            mri_count = mri_counts.get(ptid, None)
            
            if mri_count is not None:
                matched_count += 1
                if trajectory['changed']:
                    changed_mri_counts.append(mri_count)
                    if trajectory['progression']:
                        progressed_mri_counts.append(mri_count)
                    elif trajectory['improvement']:
                        improved_mri_counts.append(mri_count)
                else:
                    stable_mri_counts.append(mri_count)
            else:
                unmatched_count += 1
        
        print(f"\n📊 MRI Matching Statistics:")
        print(f"  Patients matched with MRI data: {matched_count}")
        print(f"  Patients without MRI data: {unmatched_count}")
        print(f"  Total MRI scans in dataset: {len(self.mri_df)}")
        
        # Statistical comparison
        print(f"\n📊 MRI Count Statistics by Stability:")
        print("=" * 70)
        
        if stable_mri_counts:
            print(f"\n✓ Stable Diagnosis Patients (n={len(stable_mri_counts)}):")
            print(f"  Mean MRIs: {np.mean(stable_mri_counts):.2f}")
            print(f"  Median MRIs: {np.median(stable_mri_counts):.1f}")
            print(f"  Range: {min(stable_mri_counts)} - {max(stable_mri_counts)}")
            print(f"  Std dev: {np.std(stable_mri_counts):.2f}")
        
        if changed_mri_counts:
            print(f"\n⚠️  Changed Diagnosis Patients (n={len(changed_mri_counts)}):")
            print(f"  Mean MRIs: {np.mean(changed_mri_counts):.2f}")
            print(f"  Median MRIs: {np.median(changed_mri_counts):.1f}")
            print(f"  Range: {min(changed_mri_counts)} - {max(changed_mri_counts)}")
            print(f"  Std dev: {np.std(changed_mri_counts):.2f}")
        
        if progressed_mri_counts:
            print(f"\n📈 Disease Progression Patients (n={len(progressed_mri_counts)}):")
            print(f"  Mean MRIs: {np.mean(progressed_mri_counts):.2f}")
            print(f"  Median MRIs: {np.median(progressed_mri_counts):.1f}")
            print(f"  Range: {min(progressed_mri_counts)} - {max(progressed_mri_counts)}")
        
        if improved_mri_counts:
            print(f"\n📉 Improved Patients (n={len(improved_mri_counts)}):")
            print(f"  Mean MRIs: {np.mean(improved_mri_counts):.2f}")
            print(f"  Median MRIs: {np.median(improved_mri_counts):.1f}")
        
        # Statistical significance test (if enough data)
        if len(stable_mri_counts) > 0 and len(changed_mri_counts) > 0:
            diff_means = np.mean(changed_mri_counts) - np.mean(stable_mri_counts)
            print(f"\n📊 Comparison:")
            print(f"  Difference in means: {diff_means:+.2f} MRIs")
            print(f"  Changed patients have {'MORE' if diff_means > 0 else 'FEWER'} MRIs on average")
        
        # Scanner information
        if 'scanner_manufacturer' in self.mri_df.columns:
            print(f"\n🔬 Scanner Information:")
            scanner_counts = self.mri_df['scanner_manufacturer'].value_counts()
            for scanner, count in scanner_counts.items():
                percentage = (count / len(self.mri_df)) * 100
                print(f"  {scanner}: {count} scans ({percentage:.1f}%)")
        
        # Field strength information
        if 'magnetic_field_strength' in self.mri_df.columns:
            print(f"\n⚡ Magnetic Field Strength:")
            field_counts = self.mri_df['magnetic_field_strength'].value_counts()
            for field, count in field_counts.items():
                percentage = (count / len(self.mri_df)) * 100
                print(f"  {field}T: {count} scans ({percentage:.1f}%)")
        
        # Store results
        self.mri_analysis = {
            'stable_counts': stable_mri_counts,
            'changed_counts': changed_mri_counts,
            'progressed_counts': progressed_mri_counts,
            'improved_counts': improved_mri_counts,
            'matched_patients': matched_count,
            'unmatched_patients': unmatched_count
        }
        
        return self.mri_analysis
    
    def create_visualizations(self) -> None:
        """
        Create comprehensive visualization charts.
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Set up the figure - add extra row if we have MRI data
        has_mri = len(self.mri_analysis) > 0 and len(self.mri_analysis.get('stable_counts', [])) > 0
        
        if has_mri:
            fig = plt.figure(figsize=(20, 20))
            gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # 1. Overall diagnosis distribution (all records)
        ax1 = fig.add_subplot(gs[0, 0])
        overall_dist = self.df['DIAGNOSIS'].value_counts().sort_index()
        labels = [self.DIAGNOSIS_MAP.get(int(dx), str(dx)) for dx in overall_dist.index]
        colors = [self.COLORS.get(label, 'gray') for label in labels]
        ax1.bar(labels, overall_dist.values, color=colors, alpha=0.7)
        ax1.set_ylabel('Number of Records')
        ax1.set_title('Overall Diagnosis Distribution\n(All Records)', fontweight='bold')
        for i, v in enumerate(overall_dist.values):
            ax1.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Baseline diagnosis distribution
        ax2 = fig.add_subplot(gs[0, 1])
        baseline_dx = [self.DIAGNOSIS_MAP.get(t['first_dx'], 'Unknown') 
                      for t in self.patient_trajectories.values()]
        baseline_counts = Counter(baseline_dx)
        dx_labels = ['CN', 'MCI', 'AD']
        dx_counts = [baseline_counts.get(dx, 0) for dx in dx_labels]
        colors = [self.COLORS[dx] for dx in dx_labels]
        ax2.bar(dx_labels, dx_counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Baseline Diagnosis Distribution\n(First Visit)', fontweight='bold')
        for i, v in enumerate(dx_counts):
            ax2.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Final diagnosis distribution
        ax3 = fig.add_subplot(gs[0, 2])
        final_dx = [self.DIAGNOSIS_MAP.get(t['last_dx'], 'Unknown') 
                   for t in self.patient_trajectories.values()]
        final_counts = Counter(final_dx)
        dx_counts = [final_counts.get(dx, 0) for dx in dx_labels]
        ax3.bar(dx_labels, dx_counts, color=colors, alpha=0.7)
        ax3.set_ylabel('Number of Patients')
        ax3.set_title('Final Diagnosis Distribution\n(Last Visit)', fontweight='bold')
        for i, v in enumerate(dx_counts):
            ax3.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 4. Transition matrix heatmap
        ax4 = fig.add_subplot(gs[1, 0:2])
        transition_matrix = np.zeros((3, 3))
        for baseline in ['CN', 'MCI', 'AD']:
            if baseline in self.baseline_outcomes:
                data = self.baseline_outcomes[baseline]
                i = dx_labels.index(baseline)
                for j, outcome in enumerate(dx_labels):
                    transition_matrix[i, j] = data[f'to_{outcome}']
        
        sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=dx_labels, yticklabels=dx_labels, ax=ax4,
                   cbar_kws={'label': 'Number of Patients'})
        ax4.set_xlabel('Final Diagnosis', fontsize=11)
        ax4.set_ylabel('Baseline Diagnosis', fontsize=11)
        ax4.set_title('Diagnosis Transition Matrix\n(Baseline → Final)', 
                     fontsize=12, fontweight='bold')
        
        # 5. Disease trajectory pie chart
        ax5 = fig.add_subplot(gs[1, 2])
        progressed = sum(1 for t in self.patient_trajectories.values() if t['progression'])
        improved = sum(1 for t in self.patient_trajectories.values() if t['improvement'])
        fluctuated = sum(1 for t in self.patient_trajectories.values() if t['fluctuated'])
        stable = sum(1 for t in self.patient_trajectories.values() if not t['changed'])
        
        trajectory_data = [progressed, stable, improved, fluctuated]
        trajectory_labels = [f'Progression\n({progressed})', f'Stable\n({stable})', 
                           f'Improvement\n({improved})', f'Fluctuated\n({fluctuated})']
        trajectory_colors = ['#ff6b6b', '#95e1d3', '#6bcf7f', '#ffd93d']
        ax5.pie(trajectory_data, labels=trajectory_labels, autopct='%1.1f%%',
               colors=trajectory_colors, startangle=90)
        ax5.set_title('Disease Trajectory\n(Baseline to Final)', fontweight='bold')
        
        # 6. Visit count distribution
        ax6 = fig.add_subplot(gs[2, 0])
        num_visits = [t['num_visits'] for t in self.patient_trajectories.values()]
        ax6.hist(num_visits, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax6.set_xlabel('Number of Visits')
        ax6.set_ylabel('Number of Patients')
        ax6.set_title('Visit Count Distribution', fontweight='bold')
        ax6.axvline(np.mean(num_visits), color='red', linestyle='--',
                   label=f'Mean: {np.mean(num_visits):.1f}')
        ax6.legend()
        
        # 7. Diagnosis change frequency
        ax7 = fig.add_subplot(gs[2, 1])
        num_changes = [t['num_changes'] for t in self.patient_trajectories.values()]
        change_counts = Counter(num_changes)
        ax7.bar(change_counts.keys(), change_counts.values(), color='coral', alpha=0.7)
        ax7.set_xlabel('Number of Diagnosis Changes')
        ax7.set_ylabel('Number of Patients')
        ax7.set_title('Frequency of Diagnosis Changes', fontweight='bold')
        
        # 8. Stable vs Changed
        ax8 = fig.add_subplot(gs[2, 2])
        changed = sum(1 for t in self.patient_trajectories.values() if t['changed'])
        unchanged = len(self.patient_trajectories) - changed
        ax8.pie([unchanged, changed], labels=[f'Stable\n({unchanged})', f'Changed\n({changed})'],
               autopct='%1.1f%%', colors=['#90ee90', '#ffb6c1'], startangle=90)
        ax8.set_title('Diagnosis Stability', fontweight='bold')
        
        # 9-11. Baseline outcome breakdowns (CN, MCI, AD)
        for idx, baseline in enumerate(['CN', 'MCI', 'AD']):
            ax = fig.add_subplot(gs[3, idx])
            if baseline in self.baseline_outcomes:
                data = self.baseline_outcomes[baseline]
                outcomes = ['CN', 'MCI', 'AD']
                counts = [data[f'to_{dx}'] for dx in outcomes]
                bars = ax.bar(outcomes, counts, 
                            color=[self.COLORS[dx] for dx in outcomes], alpha=0.7)
                
                # Highlight stable diagnosis
                stable_idx = outcomes.index(baseline)
                bars[stable_idx].set_edgecolor('black')
                bars[stable_idx].set_linewidth(3)
                
                ax.set_ylabel('Number of Patients')
                ax.set_title(f'{baseline} Baseline → Outcomes\n(n={data["total"]})', 
                           fontweight='bold')
                
                for i, v in enumerate(counts):
                    percentage = (v / data['total'] * 100) if data['total'] > 0 else 0
                    ax.text(i, v + 3, f'{v}\n({percentage:.0f}%)', 
                          ha='center', va='bottom', fontsize=9)
        
        # MRI visualizations (if available)
        if has_mri:
            stable_counts = self.mri_analysis['stable_counts']
            changed_counts = self.mri_analysis['changed_counts']
            
            # 12. MRI distribution comparison (box plot)
            ax12 = fig.add_subplot(gs[4, 0])
            data_to_plot = []
            labels_to_plot = []
            if stable_counts:
                data_to_plot.append(stable_counts)
                labels_to_plot.append(f'Stable\n(n={len(stable_counts)})')
            if changed_counts:
                data_to_plot.append(changed_counts)
                labels_to_plot.append(f'Changed\n(n={len(changed_counts)})')
            
            if data_to_plot:
                bp = ax12.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                bp['boxes'][0].set_facecolor('#90ee90')
                if len(bp['boxes']) > 1:
                    bp['boxes'][1].set_facecolor('#ffb6c1')
                ax12.set_ylabel('Number of MRI Scans')
                ax12.set_title('MRI Count by Diagnosis Stability', fontweight='bold')
                ax12.grid(axis='y', alpha=0.3)
            
            # 13. MRI histogram comparison
            ax13 = fig.add_subplot(gs[4, 1])
            if stable_counts and changed_counts:
                ax13.hist([stable_counts, changed_counts], 
                         bins=15, label=['Stable', 'Changed'],
                         alpha=0.6, color=['#90ee90', '#ffb6c1'])
                ax13.set_xlabel('Number of MRI Scans')
                ax13.set_ylabel('Number of Patients')
                ax13.set_title('MRI Count Distribution', fontweight='bold')
                ax13.legend()
                ax13.axvline(np.mean(stable_counts), color='green', linestyle='--', 
                           linewidth=2, label=f'Stable mean: {np.mean(stable_counts):.1f}')
                ax13.axvline(np.mean(changed_counts), color='red', linestyle='--',
                           linewidth=2, label=f'Changed mean: {np.mean(changed_counts):.1f}')
            
            # 14. MRI statistics summary
            ax14 = fig.add_subplot(gs[4, 2])
            ax14.axis('off')
            
            summary_text = "MRI Analysis Summary\n" + "="*30 + "\n\n"
            
            if stable_counts:
                summary_text += f"✓ Stable Patients:\n"
                summary_text += f"  n = {len(stable_counts)}\n"
                summary_text += f"  Mean = {np.mean(stable_counts):.2f}\n"
                summary_text += f"  Median = {np.median(stable_counts):.1f}\n\n"
            
            if changed_counts:
                summary_text += f"⚠️ Changed Patients:\n"
                summary_text += f"  n = {len(changed_counts)}\n"
                summary_text += f"  Mean = {np.mean(changed_counts):.2f}\n"
                summary_text += f"  Median = {np.median(changed_counts):.1f}\n\n"
            
            if stable_counts and changed_counts:
                diff = np.mean(changed_counts) - np.mean(stable_counts)
                summary_text += f"Difference:\n"
                summary_text += f"  {diff:+.2f} MRIs\n"
                summary_text += f"  Changed patients have\n"
                summary_text += f"  {'MORE' if diff > 0 else 'FEWER'} MRIs on avg"
            
            ax14.text(0.1, 0.5, summary_text, transform=ax14.transAxes,
                     fontsize=11, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                     family='monospace')
        
        # Main title
        fig.suptitle('ADNI Diagnosis Analysis - dxsum.csv', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save
        output_path = os.path.join(self.output_dir, 'dxsum_visualizations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved: {output_path}")
    
    def generate_report(self) -> None:
        """
        Generate comprehensive Markdown report.
        """
        print("\n" + "=" * 80)
        print("GENERATING MARKDOWN REPORT")
        print("=" * 80)
        
        report_path = os.path.join(self.output_dir, 'dxsum_analysis_report.md')
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# ADNI Diagnosis Summary Analysis Report\n\n")
            f.write("**Analysis of dxsum.csv**\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Data source:** `{self.csv_path}`\n\n")
            f.write("---\n\n")
            
            # 1. Dataset Overview
            f.write("## 1. Dataset Overview\n\n")
            
            f.write(f"- **Total records:** {self.stats['total_records']}\n")
            f.write(f"- **Total unique patients:** {self.stats['total_patients']}\n")
            f.write(f"- **Date range:** {self.stats['date_range'][0]} to {self.stats['date_range'][1]}\n\n")
            
            visits = self.stats['visits_per_patient']
            f.write("### Visit Statistics\n\n")
            f.write(f"- Mean visits per patient: **{visits.mean():.2f}**\n")
            f.write(f"- Median visits per patient: **{visits.median():.1f}**\n")
            f.write(f"- Range: **{visits.min()} - {visits.max()}** visits\n")
            f.write(f"- Standard deviation: {visits.std():.2f}\n\n")
            
            # 2. Diagnosis Code Mapping
            f.write("## 2. Diagnosis Code Mapping\n\n")
            
            f.write("| Code | Label | Description |\n")
            f.write("|------|-------|-------------|\n")
            f.write("| 1 | CN | Cognitively Normal |\n")
            f.write("| 2 | MCI | Mild Cognitive Impairment |\n")
            f.write("| 3 | AD | Alzheimer's Disease |\n\n")
            
            # 3. Diagnosis Distribution
            f.write("## 3. Diagnosis Distribution\n\n")
            
            f.write("### Overall Distribution (All Records)\n\n")
            f.write("| Diagnosis | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for dx_code, count in sorted(self.stats['distribution']['overall'].items()):
                label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
                percentage = (count / self.stats['total_records']) * 100
                f.write(f"| {label} | {count} | {percentage:.1f}% |\n")
            
            f.write("\n### Baseline Distribution (First Visit)\n\n")
            f.write("| Diagnosis | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for dx_code, count in sorted(self.stats['distribution']['baseline'].items()):
                label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
                percentage = (count / len(self.patient_trajectories)) * 100
                f.write(f"| {label} | {count} | {percentage:.1f}% |\n")
            
            f.write("\n### Final Distribution (Last Visit)\n\n")
            f.write("| Diagnosis | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for dx_code, count in sorted(self.stats['distribution']['final'].items()):
                label = self.DIAGNOSIS_MAP.get(dx_code, f'Unknown({dx_code})')
                percentage = (count / len(self.patient_trajectories)) * 100
                f.write(f"| {label} | {count} | {percentage:.1f}% |\n")
            
            # 4. Transition Analysis
            f.write("\n## 4. Diagnosis Transition Analysis\n\n")
            
            total_patients = len(self.patient_trajectories)
            changed = sum(1 for t in self.patient_trajectories.values() if t['changed'])
            progressed = sum(1 for t in self.patient_trajectories.values() if t['progression'])
            improved = sum(1 for t in self.patient_trajectories.values() if t['improvement'])
            fluctuated = sum(1 for t in self.patient_trajectories.values() if t['fluctuated'])
            stable = total_patients - changed
            
            f.write("### Overall Trajectory Summary\n\n")
            f.write(f"- **Stable diagnosis:** {stable} ({stable/total_patients*100:.1f}%)\n")
            f.write(f"- **Changed diagnosis:** {changed} ({changed/total_patients*100:.1f}%)\n")
            f.write(f"  - Disease progression: {progressed} ({progressed/total_patients*100:.1f}%)\n")
            f.write(f"  - Improvement: {improved} ({improved/total_patients*100:.1f}%)\n")
            f.write(f"  - Fluctuated (returned to baseline): {fluctuated} ({fluctuated/total_patients*100:.1f}%)\n\n")
            
            f.write("### Transition Frequencies\n\n")
            f.write("| Transition | Count | Percentage |\n")
            f.write("|------------|-------|------------|\n")
            transition_counts = Counter()
            for t in self.transition_details:
                key = f"{t['from_label']} → {t['to_label']}"
                transition_counts[key] += 1
            
            for transition, count in transition_counts.most_common():
                percentage = (count / len(self.transition_details)) * 100 if self.transition_details else 0
                f.write(f"| {transition} | {count} | {percentage:.1f}% |\n")
            
            # 4b. Fluctuation Pattern Analysis
            if self.fluctuation_patterns:
                f.write("\n### Detailed Fluctuation Patterns\n\n")
                f.write("**Patients who changed diagnosis but returned to baseline:**\n\n")
                
                total_fluctuated = sum(len(patients) for patients in self.fluctuation_patterns.values())
                f.write(f"Total fluctuated patients: **{total_fluctuated}**\n\n")
                
                # Sort patterns by frequency
                sorted_patterns = sorted(self.fluctuation_patterns.items(), 
                                        key=lambda x: len(x[1]), reverse=True)
                
                f.write("| Pattern | Count | Avg Visits | Avg Transitions |\n")
                f.write("|---------|-------|------------|----------------|\n")
                
                for pattern, patients in sorted_patterns:
                    count = len(patients)
                    avg_visits = np.mean([p['num_visits'] for p in patients])
                    avg_transitions = np.mean([p['num_transitions'] for p in patients])
                    f.write(f"| {pattern} | {count} | {avg_visits:.1f} | {avg_transitions:.1f} |\n")
                
                # Breakdown by baseline
                cn_fluct = sum(len(patients) for pattern, patients in self.fluctuation_patterns.items() 
                              if pattern.startswith('CN'))
                mci_fluct = sum(len(patients) for pattern, patients in self.fluctuation_patterns.items() 
                               if pattern.startswith('MCI'))
                ad_fluct = sum(len(patients) for pattern, patients in self.fluctuation_patterns.items() 
                              if pattern.startswith('AD'))
                
                f.write("\n**Fluctuation by Baseline Diagnosis:**\n\n")
                f.write(f"- **CN baseline:** {cn_fluct} patients fluctuated\n")
                f.write(f"- **MCI baseline:** {mci_fluct} patients fluctuated\n")
                f.write(f"- **AD baseline:** {ad_fluct} patients fluctuated\n\n")
            
            # 5. Progression by Baseline
            f.write("\n## 5. Progression Analysis by Baseline Diagnosis\n\n")
            
            for baseline in ['CN', 'MCI', 'AD']:
                if baseline in self.baseline_outcomes:
                    data = self.baseline_outcomes[baseline]
                    total = data['total']
                    
                    f.write(f"### {baseline} Patients (n={total})\n\n")
                    f.write("| Final Diagnosis | Count | Percentage | Status |\n")
                    f.write("|-----------------|-------|------------|--------|\n")
                    
                    for outcome in ['CN', 'MCI', 'AD']:
                        count = data[f'to_{outcome}']
                        percentage = (count / total * 100) if total > 0 else 0
                        status = '✓ Stable' if baseline == outcome else '⚠️ Changed'
                        f.write(f"| → {outcome} | {count} | {percentage:.1f}% | {status} |\n")
                    f.write("\n")
            
            # 6. Key Insights
            f.write("## 6. Key Insights\n\n")
            
            # CN insights
            if 'CN' in self.baseline_outcomes:
                cn_data = self.baseline_outcomes['CN']
                cn_to_mci = cn_data['to_MCI']
                cn_to_ad = cn_data['to_AD']
                cn_total = cn_data['total']
                f.write(f"### 🧠 CN Progression\n\n")
                f.write(f"- **{cn_to_mci}/{cn_total}** ({cn_to_mci/cn_total*100:.1f}%) progressed to MCI\n")
                f.write(f"- **{cn_to_ad}/{cn_total}** ({cn_to_ad/cn_total*100:.1f}%) progressed to AD\n")
                f.write(f"- **{cn_data['to_CN']}/{cn_total}** ({cn_data['to_CN']/cn_total*100:.1f}%) remained stable\n\n")
            
            # MCI insights
            if 'MCI' in self.baseline_outcomes:
                mci_data = self.baseline_outcomes['MCI']
                mci_to_ad = mci_data['to_AD']
                mci_to_cn = mci_data['to_CN']
                mci_total = mci_data['total']
                f.write(f"### ⚠️ MCI Critical Window\n\n")
                f.write(f"- **{mci_to_ad}/{mci_total}** ({mci_to_ad/mci_total*100:.1f}%) progressed to AD\n")
                f.write(f"- **{mci_to_cn}/{mci_total}** ({mci_to_cn/mci_total*100:.1f}%) improved to CN\n")
                f.write(f"- **{mci_data['to_MCI']}/{mci_total}** ({mci_data['to_MCI']/mci_total*100:.1f}%) remained stable\n\n")
            
            # Overall insights
            f.write(f"### 📊 Overall Statistics\n\n")
            f.write(f"- **Overall Progression Rate:** {progressed/total_patients*100:.1f}%\n")
            f.write(f"- **Overall Stability Rate:** {stable/total_patients*100:.1f}%\n")
            f.write(f"- **Most Common Transition:** {transition_counts.most_common(1)[0][0]} ")
            f.write(f"({transition_counts.most_common(1)[0][1]} cases)\n\n")
            
            # 7. MRI Analysis (if available)
            if self.mri_analysis and len(self.mri_analysis.get('stable_counts', [])) > 0:
                f.write("## 7. MRI Analysis by Diagnosis Stability\n\n")
                
                stable_counts = self.mri_analysis['stable_counts']
                changed_counts = self.mri_analysis['changed_counts']
                
                f.write(f"**Patients matched with MRI data:** {self.mri_analysis['matched_patients']}  \n")
                f.write(f"**Patients without MRI data:** {self.mri_analysis['unmatched_patients']}\n\n")
                
                f.write("### MRI Count Statistics\n\n")
                f.write("| Group | N | Mean | Median | Range |\n")
                f.write("|-------|---|------|--------|-------|\n")
                
                if stable_counts:
                    f.write(f"| ✓ Stable | {len(stable_counts)} | {np.mean(stable_counts):.2f} | ")
                    f.write(f"{np.median(stable_counts):.1f} | {min(stable_counts)}-{max(stable_counts)} |\n")
                
                if changed_counts:
                    f.write(f"| ⚠️ Changed | {len(changed_counts)} | {np.mean(changed_counts):.2f} | ")
                    f.write(f"{np.median(changed_counts):.1f} | {min(changed_counts)}-{max(changed_counts)} |\n")
                
                if stable_counts and changed_counts:
                    diff = np.mean(changed_counts) - np.mean(stable_counts)
                    f.write(f"\n### 📊 Key Finding\n\n")
                    f.write(f"Patients whose diagnosis **changed** have an average of **{abs(diff):.2f} ")
                    f.write(f"{'MORE' if diff > 0 else 'FEWER'}** MRI scans compared to stable patients.\n\n")
                    
                    if diff > 0:
                        f.write("This suggests that patients with more longitudinal imaging data ")
                        f.write("are more likely to show diagnosis changes, possibly due to better ")
                        f.write("monitoring and detection of cognitive changes over time.\n\n")
                
                # Scanner info if available
                if self.mri_df is not None and 'scanner_manufacturer' in self.mri_df.columns:
                    f.write("### 🔬 Scanner Information\n\n")
                    f.write("| Manufacturer | Scans | Percentage |\n")
                    f.write("|--------------|-------|------------|\n")
                    scanner_counts = self.mri_df['scanner_manufacturer'].value_counts()
                    for scanner, count in scanner_counts.items():
                        percentage = (count / len(self.mri_df)) * 100
                        f.write(f"| {scanner} | {count} | {percentage:.1f}% |\n")
                    f.write("\n")
                
                # Field strength info
                if self.mri_df is not None and 'magnetic_field_strength' in self.mri_df.columns:
                    f.write("### ⚡ Magnetic Field Strength\n\n")
                    f.write("| Field Strength | Scans | Percentage |\n")
                    f.write("|----------------|-------|------------|\n")
                    field_counts = self.mri_df['magnetic_field_strength'].value_counts()
                    for field, count in field_counts.items():
                        percentage = (count / len(self.mri_df)) * 100
                        f.write(f"| {field}T | {count} | {percentage:.1f}% |\n")
                    f.write("\n")
            
            # Footer
            f.write("---\n\n")
            f.write("*End of Report*\n")
        
        print(f"✅ Report saved: {report_path}")
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete analysis pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "🧠 " * 40)
        print("DXSUM.CSV COMPREHENSIVE ANALYSIS")
        print("🧠 " * 40 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Run analyses
        self.analyze_basic_stats()
        self.analyze_diagnosis_distribution()
        self.analyze_transitions()
        self.analyze_fluctuation_patterns()
        self.analyze_progression_by_baseline()
        
        # Analyze MRI data if available
        if self.mri_df is not None:
            self.analyze_mri_by_stability()
        
        # Generate outputs
        self.create_visualizations()
        self.generate_report()
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nOutput files generated in: {self.output_dir}/")
        print("  • dxsum_visualizations.png - Comprehensive charts")
        print("  • dxsum_analysis_report.md - Detailed Markdown report")
        print("\nKey Statistics:")
        print(f"  • Total patients analyzed: {len(self.patient_trajectories)}")
        print(f"  • Total records: {self.stats['total_records']}")
        print(f"  • Progression rate: {sum(1 for t in self.patient_trajectories.values() if t['progression'])/len(self.patient_trajectories)*100:.1f}%")
        print("=" * 80 + "\n")
        
        return True


def main():
    """Main execution function."""
    # Configuration
    BASE_PATH = '/Users/tanguyvans/Desktop/umons/alzheimer'
    CSV_PATH = os.path.join(BASE_PATH, 'dxsum.csv')
    MRI_CSV_PATH = os.path.join(BASE_PATH, '3D_MPRAGE_Cohort_Study_Key_MRI_18Sep2025.csv')
    OUTPUT_DIR = os.path.join(BASE_PATH, 'data_analysis')
    
    # Create analyzer and run
    analyzer = DxsumAnalyzer(
        csv_path=CSV_PATH, 
        output_dir=OUTPUT_DIR,
        mri_csv_path=MRI_CSV_PATH
    )
    success = analyzer.run_complete_analysis()
    
    if not success:
        print("❌ Analysis failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

