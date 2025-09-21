#!/usr/bin/env python3
"""
Reprocess failed files identified by verification or in progress tracking.
This script specifically targets files that failed during batch processing.

Usage:
    python reprocess_failed_files.py                # Reprocess all failed files
    python reprocess_failed_files.py --limit 10     # Process only first 10 failed files
    python reprocess_failed_files.py --patient 128_S_4742  # Process specific patient
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

sys.path.append('./preprocessing')

def load_progress(progress_file):
    """Load processing progress from JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def find_source_file(filename, patient_id, input_dir):
    """Find the source NIFTI file in the input directory."""
    # Try patient-specific directory first
    patient_dir = os.path.join(input_dir, patient_id)
    if os.path.exists(patient_dir):
        source_file = os.path.join(patient_dir, filename)
        if os.path.exists(source_file):
            return source_file
    
    # Search recursively
    for root, dirs, files in os.walk(input_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            # Verify it's the right patient
            if patient_id in full_path:
                return full_path
    
    return None

def process_single_file(nifti_file, output_file, template_path, identifier):
    """Process a single file with enhanced error handling."""
    try:
        # Create temporary Python script for isolated processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(f"""
import sys
import os
import traceback

sys.path.append('./preprocessing')

try:
    from image_enhancement import process_nifti_file
    
    success = process_nifti_file(
        '{nifti_file}',
        '{output_file}',
        '{template_path}',
        '{identifier}'
    )
    
    if success and os.path.exists('{output_file}'):
        # Verify the output file is valid
        import nibabel as nib
        try:
            img = nib.load('{output_file}')
            data = img.get_fdata()
            if data is not None and data.size > 0:
                print('SUCCESS')
            else:
                print('FAILED: Output file is empty')
        except Exception as e:
            print(f'FAILED: Invalid output file - {{e}}')
    else:
        print('FAILED: Processing returned False or output not created')
        
except Exception as e:
    print(f'FAILED: {{e}}')
    traceback.print_exc()
""")
            temp_script = f.name
        
        # Run with increased timeout and memory limits
        env = os.environ.copy()
        env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'  # Limit threads to avoid conflicts
        
        result = subprocess.run(
            ['python3', temp_script],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=env
        )
        
        os.unlink(temp_script)
        
        if 'SUCCESS' in result.stdout:
            return True, None
        else:
            error_msg = result.stdout.replace('FAILED:', '').strip()
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (>10 minutes)"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Reprocess failed registration files')
    parser.add_argument('--input', 
                       default='/Volumes/KINGSTON/ADNI_nifti',
                       help='Input directory with original files')
    parser.add_argument('--output',
                       default='/Volumes/KINGSTON/ADNI-registered',
                       help='Output directory')
    parser.add_argument('--template', 
                       default='mni_template/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii',
                       help='MNI template path')
    parser.add_argument('--limit', type=int,
                       help='Limit number of files to reprocess')
    parser.add_argument('--patient',
                       help='Process specific patient ID only')
    parser.add_argument('--verify-first', action='store_true',
                       help='Run verification before reprocessing')
    
    args = parser.parse_args()
    
    # Check external drive
    if not os.path.exists('/Volumes/KINGSTON'):
        print("❌ External drive not mounted at /Volumes/KINGSTON")
        return
    
    # Run verification first if requested
    if args.verify_first:
        print("🔍 Running verification first...")
        subprocess.run(['python3', 'verify_registration_results.py'])
        print()
    
    # Load progress
    progress_file = os.path.join(args.output, 'processing_progress.json')
    progress = load_progress(progress_file)
    
    if not progress:
        print("❌ No progress file found. Run batch processing first.")
        return
    
    # Get failed files
    failed_files = progress.get('failed', [])
    
    if not failed_files:
        print("✅ No failed files to reprocess!")
        return
    
    print(f"📋 Found {len(failed_files)} failed files")
    
    # Filter by patient if specified
    if args.patient:
        # Try to find files for this patient
        patient_failed = []
        for filename in failed_files:
            source = find_source_file(filename, args.patient, args.input)
            if source:
                patient_failed.append(filename)
        
        if not patient_failed:
            print(f"❌ No failed files found for patient {args.patient}")
            return
        
        failed_files = patient_failed
        print(f"   Filtering to {len(failed_files)} files for patient {args.patient}")
    
    # Apply limit if specified
    if args.limit:
        failed_files = failed_files[:args.limit]
        print(f"   Limited to first {args.limit} files")
    
    # Create reprocessing log
    reprocess_log_file = os.path.join(args.output, f'reprocess_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    reprocess_log = {
        'start_time': datetime.now().isoformat(),
        'total_files': len(failed_files),
        'results': []
    }
    
    print(f"\n🔄 Starting reprocessing of {len(failed_files)} files")
    print("-" * 60)
    
    success_count = 0
    still_failed = []
    start_time = time.time()
    
    for i, filename in enumerate(failed_files, 1):
        print(f"\n[{i}/{len(failed_files)}] Processing: {filename}")
        
        # Find patient ID and source file
        patient_id = None
        source_file = None
        
        # Search for the source file
        for root, dirs, files in os.walk(args.input):
            if filename in files:
                source_file = os.path.join(root, filename)
                patient_id = os.path.basename(os.path.dirname(source_file))
                break
        
        if not source_file or not os.path.exists(source_file):
            print(f"   ❌ Source file not found")
            still_failed.append({
                'filename': filename,
                'error': 'Source file not found',
                'patient_id': patient_id
            })
            continue
        
        print(f"   Patient: {patient_id}")
        print(f"   Source: {source_file}")
        
        # Create output path
        output_dir = os.path.join(args.output, patient_id)
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = filename.replace('.nii.gz', '_registered.nii.gz')
        output_file = os.path.join(output_dir, output_filename)
        
        # Check if already exists (maybe processed in another run)
        if os.path.exists(output_file):
            print(f"   ⏭️  Output already exists, verifying...")
            # Verify it's valid
            try:
                import nibabel as nib
                img = nib.load(output_file)
                data = img.get_fdata()
                if data is not None and data.size > 0:
                    print(f"   ✅ Valid existing file found")
                    success_count += 1
                    
                    # Update progress
                    if filename in progress['failed']:
                        progress['failed'].remove(filename)
                    if filename not in progress['processed']:
                        progress['processed'].append(filename)
                    
                    continue
            except:
                print(f"   ⚠️  Existing file is corrupted, reprocessing...")
                os.remove(output_file)
        
        # Process the file
        print(f"   🔄 Processing with N4 + MNI registration...")
        file_start = time.time()
        
        identifier = f"{patient_id}_{filename.replace('.nii.gz', '')}"
        success, error_msg = process_single_file(
            source_file,
            output_file,
            args.template,
            identifier
        )
        
        file_time = time.time() - file_start
        
        if success:
            print(f"   ✅ Success ({file_time:.1f}s)")
            success_count += 1
            
            # Update progress
            if filename in progress['failed']:
                progress['failed'].remove(filename)
            if filename not in progress['processed']:
                progress['processed'].append(filename)
            
            reprocess_log['results'].append({
                'filename': filename,
                'patient_id': patient_id,
                'status': 'success',
                'time': file_time
            })
        else:
            print(f"   ❌ Failed: {error_msg}")
            still_failed.append({
                'filename': filename,
                'patient_id': patient_id,
                'error': error_msg
            })
            
            reprocess_log['results'].append({
                'filename': filename,
                'patient_id': patient_id,
                'status': 'failed',
                'error': error_msg,
                'time': file_time
            })
        
        # Save updated progress after each file
        progress['last_updated'] = datetime.now().isoformat()
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Show ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = len(failed_files) - i
        eta = remaining * avg_time
        print(f"   📊 Progress: {(i/len(failed_files))*100:.1f}% | ETA: {eta/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 REPROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully reprocessed: {success_count}/{len(failed_files)}")
    print(f"❌ Still failed: {len(still_failed)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if still_failed:
        print(f"\n❌ Files that still failed ({len(still_failed)}):")
        for failed in still_failed[:10]:
            print(f"   - {failed['patient_id']}/{failed['filename']}: {failed['error']}")
        if len(still_failed) > 10:
            print(f"   ... and {len(still_failed)-10} more")
    
    # Save reprocessing log
    reprocess_log['end_time'] = datetime.now().isoformat()
    reprocess_log['total_time'] = total_time
    reprocess_log['success_count'] = success_count
    reprocess_log['failed_count'] = len(still_failed)
    reprocess_log['still_failed'] = still_failed
    
    with open(reprocess_log_file, 'w') as f:
        json.dump(reprocess_log, f, indent=2)
    
    print(f"\n📄 Reprocessing log saved to: {reprocess_log_file}")
    print(f"📄 Updated progress saved to: {progress_file}")
    
    # Suggest next steps
    if still_failed:
        print("\n💡 Suggestions for remaining failures:")
        print("   1. Check if source files are corrupted")
        print("   2. Try processing with different parameters")
        print("   3. Manually inspect problematic files")
        print("   4. Consider skipping files with persistent errors")

if __name__ == '__main__':
    main()