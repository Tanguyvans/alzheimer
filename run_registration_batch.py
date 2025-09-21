#!/usr/bin/env python3
"""
Batch N4 bias correction and MNI registration for ADNI dataset.
Processes NIfTI files and stores results on external drive.
Automatically skips files that have already been processed.

Usage:
    python run_registration_batch.py           # Process all files
    python run_registration_batch.py --resume  # Resume and update progress tracking
"""

import os
import sys
import argparse
import signal
import json
import time
from pathlib import Path
from datetime import datetime

# Add preprocessing module to path
sys.path.append('./preprocessing')
from pipeline import PreprocessingPipeline

# Global flag for graceful shutdown
should_stop = False

def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global should_stop
    print("\n\n🛑 Stopping processing after current file completes...")
    print("Press Ctrl+C again to force quit")
    should_stop = True

def load_progress(progress_file):
    """Load processing progress from JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'processed': [],
        'failed': [],
        'last_updated': None,
        'total_files': 0
    }

def save_progress(progress_file, progress_data):
    """Save processing progress to JSON file."""
    progress_data['last_updated'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def find_nifti_files(input_dir):
    """Find all NIfTI files in the ADNI_nifti directory."""
    nifti_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Skip hidden files (macOS metadata files starting with ._)
            if file.endswith('.nii.gz') and not file.startswith('.'):
                nifti_files.append(os.path.join(root, file))
    
    # Sort for consistent ordering
    nifti_files.sort()
    
    return nifti_files

def main():
    parser = argparse.ArgumentParser(description='N4 bias correction and MNI registration for ADNI dataset')
    parser.add_argument('--input', 
                       default='/Volumes/KINGSTON/ADNI_nifti',
                       help='Input directory (default: /Volumes/KINGSTON/ADNI_nifti)')
    parser.add_argument('--output',
                       default='/Volumes/KINGSTON/ADNI-registered',
                       help='Output directory (default: /Volumes/KINGSTON/ADNI-registered)')
    parser.add_argument('--template', 
                       default='mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii',
                       help='MNI template path')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progress')
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if external drive is mounted
    if not os.path.exists('/Volumes/KINGSTON'):
        print("❌ External drive not mounted at /Volumes/KINGSTON")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Progress tracking file
    progress_file = os.path.join(args.output, 'processing_progress.json')
    progress = load_progress(progress_file)
    
    # Find all NIfTI files
    print(f"🔍 Searching for NIfTI files in {args.input}...")
    all_files = find_nifti_files(args.input)
    progress['total_files'] = len(all_files)
    print(f"📁 Found {len(all_files)} NIfTI files")
    
    # Filter out already processed files if resuming
    if args.resume:
        files_to_process = [f for f in all_files 
                          if os.path.basename(f) not in progress['processed']]
        print(f"📊 Resume mode: {len(progress['processed'])} already processed, "
              f"{len(progress['failed'])} failed, {len(files_to_process)} remaining")
    else:
        files_to_process = all_files
        progress['processed'] = []
        progress['failed'] = []
        print(f"🆕 Starting fresh processing of {len(files_to_process)} files")
    
    if not files_to_process:
        print("✅ All files already processed!")
        return
    
    # We won't use the full pipeline structure, just process and save directly
    print(f"\n⚙️  Initializing registration...")
    print(f"   Output: {args.output}")
    print(f"   Template: {args.template}")
    
    # Process files
    print(f"\n🚀 Starting N4 bias correction + MNI registration")
    print("   Press Ctrl+C to stop gracefully after current file\n")
    print("-" * 60)
    
    start_time = time.time()
    processed_count = len(progress['processed'])
    
    for i, nifti_file in enumerate(files_to_process, 1):
        if should_stop:
            print(f"\n⏸️  Stopped by user. Progress saved.")
            break
        
        filename = os.path.basename(nifti_file)
        patient_id = os.path.basename(os.path.dirname(nifti_file))
        
        print(f"\n[{processed_count + 1}/{progress['total_files']}] Patient: {patient_id}")
        print(f"   File: {filename}")
        
        try:
            # Create output path maintaining same structure as ADNI_nifti
            output_dir = os.path.join(args.output, patient_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Output file in same directory structure, with _registered suffix
            output_filename = filename.replace('.nii.gz', '_registered.nii.gz')
            output_file = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_file):
                print(f"   ⏭️  Already processed (output exists), skipping")
                if filename not in progress['processed']:
                    progress['processed'].append(filename)
                processed_count += 1
            else:
                # Run N4 bias correction + MNI registration
                print(f"   🔄 Processing with N4 bias correction + MNI registration...")
                file_start = time.time()
                
                try:
                    # Process single file with explicit error handling
                    import subprocess
                    import tempfile
                    
                    # Write a temporary processing script to isolate potential crashes
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(f"""
import sys
sys.path.append('./preprocessing')
from image_enhancement import process_nifti_file

success = process_nifti_file(
    '{nifti_file}',
    '{output_file}',
    '{args.template}',
    '{patient_id}_{filename.replace(".nii.gz", "")}'
)
print('SUCCESS' if success else 'FAILED')
""")
                        temp_script = f.name
                    
                    # Run in subprocess to isolate segmentation faults
                    result = subprocess.run(
                        ['python3', temp_script],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout per file
                    )
                    
                    os.unlink(temp_script)
                    
                    file_time = time.time() - file_start
                    
                    if 'SUCCESS' in result.stdout:
                        progress['processed'].append(filename)
                        processed_count += 1
                        print(f"   ✅ Success ({file_time:.1f}s)")
                    else:
                        progress['failed'].append(filename)
                        print(f"   ❌ Failed")
                        if result.stderr:
                            print(f"      Error: {result.stderr[:100]}...")
                            
                except subprocess.TimeoutExpired:
                    progress['failed'].append(filename)
                    print(f"   ⏱️ Timeout (>5 minutes)")
                except Exception as e:
                    progress['failed'].append(filename)
                    print(f"   ❌ Error: {str(e)[:100]}")
            
            # Save progress after each file
            save_progress(progress_file, progress)
            
            # Show progress statistics
            elapsed = time.time() - start_time
            avg_time = elapsed / i if i > 0 else 0
            remaining = len(files_to_process) - i
            eta = remaining * avg_time if avg_time > 0 else 0
            
            completion = (processed_count / progress['total_files']) * 100
            print(f"   📊 Progress: {completion:.1f}% | Avg: {avg_time:.1f}s | ETA: {eta/60:.1f} min")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            progress['failed'].append(filename)
            save_progress(progress_file, progress)
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 N4 BIAS CORRECTION + MNI REGISTRATION COMPLETE")
    print("="*60)
    print(f"✅ Processed: {len(progress['processed'])} files")
    print(f"❌ Failed: {len(progress['failed'])} files") 
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"📁 Output directory: {args.output}")
    print(f"   Structure: {args.output}/patient_id/filename_registered.nii.gz")
    print(f"📄 Progress saved to: {progress_file}")
    
    if progress['failed']:
        print(f"\n⚠️  Failed files:")
        for failed in progress['failed'][:10]:  # Show first 10
            print(f"   - {failed}")
        if len(progress['failed']) > 10:
            print(f"   ... and {len(progress['failed'])-10} more")

if __name__ == '__main__':
    main()