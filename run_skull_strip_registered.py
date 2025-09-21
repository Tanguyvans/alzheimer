#!/usr/bin/env python3
"""
Batch skull stripping using SynthStrip for registered NIfTI files.
Processes registered images and stores skull-stripped results.

Usage:
    python run_skull_strip_registered.py           # Process all registered files
    python run_skull_strip_registered.py --resume  # Resume and update progress tracking
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
from skull_stripping import synthstrip_skull_strip, setup_synthstrip_docker

# Global flag for graceful shutdown
should_stop = False

def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global should_stop
    print("\n\n🛑 Stopping skull stripping after current file completes...")
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

def find_registered_files(input_dir):
    """Find all registered NIfTI files in the input directory."""
    registered_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Skip hidden files (macOS metadata files starting with ._)
            if file.endswith('_registered.nii.gz') and not file.startswith('.'):
                registered_files.append(os.path.join(root, file))

    # Sort for consistent ordering
    registered_files.sort()

    return registered_files

def main():
    parser = argparse.ArgumentParser(description='Skull stripping for registered ADNI dataset')
    parser.add_argument('--input',
                       default='/Volumes/KINGSTON/ADNI-registered',
                       help='Input directory containing registered NIfTI files (default: /Volumes/KINGSTON/ADNI-registered)')
    parser.add_argument('--output',
                       default='/Volumes/KINGSTON/ADNI-skull',
                       help='Output directory for skull-stripped files (default: /Volumes/KINGSTON/ADNI-skull)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progress')

    args = parser.parse_args()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Check if external drive is mounted
    if not os.path.exists('/Volumes/KINGSTON'):
        print("❌ External drive not mounted at /Volumes/KINGSTON")
        return

    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Progress tracking file
    progress_file = os.path.join(args.output, 'skull_stripping_progress.json')
    progress = load_progress(progress_file)

    # Setup SynthStrip Docker environment
    print("🐳 Setting up SynthStrip Docker environment...")
    if not setup_synthstrip_docker():
        print("❌ Failed to setup SynthStrip Docker environment")
        return
    print("✅ SynthStrip Docker environment ready\n")

    # Find all registered NIfTI files
    print(f"🔍 Searching for registered NIfTI files in {args.input}...")
    all_files = find_registered_files(args.input)
    progress['total_files'] = len(all_files)
    print(f"📁 Found {len(all_files)} registered files")

    # Filter out already processed files if resuming
    if args.resume:
        files_to_process = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            patient_id = os.path.basename(os.path.dirname(file_path))
            expected_output = os.path.join(args.output, patient_id, filename.replace('_registered.nii.gz', '_registered_skull_stripped.nii.gz'))

            if os.path.exists(expected_output):
                # Mark as processed if output exists
                if filename not in progress['processed']:
                    progress['processed'].append(filename)
            else:
                files_to_process.append(file_path)

        print(f"📊 Resume mode: {len(progress['processed'])} already processed, "
              f"{len(progress['failed'])} failed, {len(files_to_process)} remaining")
    else:
        files_to_process = all_files
        progress['processed'] = []
        progress['failed'] = []
        print(f"🆕 Starting fresh skull stripping of {len(files_to_process)} files")

    if not files_to_process:
        print("✅ All files already skull-stripped!")
        return

    # Process files
    print(f"\n⚙️  Starting skull stripping...")
    print(f"   Output: {args.output}")
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
            # Create output path maintaining same structure as input
            output_dir = os.path.join(args.output, patient_id)
            os.makedirs(output_dir, exist_ok=True)

            # Output file: replace _registered.nii.gz with _registered_skull_stripped.nii.gz
            output_filename = filename.replace('_registered.nii.gz', '_registered_skull_stripped.nii.gz')
            output_file = os.path.join(output_dir, output_filename)

            if os.path.exists(output_file):
                print(f"   ⏭️  Already skull-stripped (output exists), skipping")
                if filename not in progress['processed']:
                    progress['processed'].append(filename)
                processed_count += 1
            else:
                # Run skull stripping
                print(f"   🧠 Skull stripping...")
                file_start = time.time()

                subject_id = f"{patient_id}_{filename.replace('_registered.nii.gz', '')}"
                success = synthstrip_skull_strip(nifti_file, output_file, subject_id)

                file_time = time.time() - file_start

                if success:
                    progress['processed'].append(filename)
                    processed_count += 1
                    print(f"   ✅ Success ({file_time:.1f}s)")
                else:
                    progress['failed'].append(filename)
                    print(f"   ❌ Failed")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            progress['failed'].append(filename)
            save_progress(progress_file, progress)
            continue

        # Save progress after each file
        save_progress(progress_file, progress)

        # Show progress statistics
        elapsed = time.time() - start_time
        avg_time = elapsed / i if i > 0 else 0
        remaining = len(files_to_process) - i
        eta = remaining * avg_time if avg_time > 0 else 0

        completion = (processed_count / progress['total_files']) * 100
        print(f"   📊 Progress: {completion:.1f}% | Avg: {avg_time:.1f}s | ETA: {eta/60:.1f} min")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🧠 SKULL STRIPPING COMPLETE")
    print("="*60)
    print(f"✅ Successfully skull-stripped: {len(progress['processed'])} files")
    print(f"❌ Failed: {len(progress['failed'])} files")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"📁 Output directory: {args.output}")
    print(f"   Structure: {args.output}/patient_id/filename_registered_skull_stripped.nii.gz")
    print(f"📄 Progress saved to: {progress_file}")

    if progress['failed']:
        print(f"\n⚠️  Failed files:")
        for failed in progress['failed'][:10]:  # Show first 10
            print(f"   - {failed}")
        if len(progress['failed']) > 10:
            print(f"   ... and {len(progress['failed'])-10} more")

if __name__ == '__main__':
    main()
