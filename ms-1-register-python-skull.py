import os
import logging
import subprocess
from tqdm import tqdm
import time
import warnings

# Disable all warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define base directory and other paths
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"
INPUT_DIR = os.path.join(BASE_DIR, "ms-1-register")
OUTPUT_DIR = os.path.join(BASE_DIR, "ms-1-synthstrip-skull")

def check_docker_availability():
    """
    Check if Docker is installed and running.
    """
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        logger.info(f"Docker version: {result.stdout.strip()}")
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, check=True)
        logger.info("Docker daemon is running")
        return True
        
    except subprocess.CalledProcessError:
        logger.error("Docker is not installed or not running")
        logger.error("Please install Docker Desktop for Mac from: https://docker.com/products/docker-desktop")
        return False
    except FileNotFoundError:
        logger.error("Docker command not found")
        logger.error("Please install Docker Desktop for Mac from: https://docker.com/products/docker-desktop")
        return False

def pull_synthstrip_image():
    """
    Pull the official SynthStrip Docker image.
    """
    try:
        logger.info("Pulling SynthStrip Docker image...")
        result = subprocess.run(['docker', 'pull', 'freesurfer/synthstrip:latest'], 
                              capture_output=True, text=True, check=True)
        logger.info("Successfully pulled SynthStrip Docker image")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to pull Docker image: {e.stderr}")
        return False

def docker_synthstrip_skull_strip(input_file_path, output_file_path, subject_id):
    """
    Performs skull stripping using SynthStrip Docker container.
    """
    logger.info(f"Starting SynthStrip skull stripping for {subject_id}")
    
    try:
        # Get absolute paths
        input_abs = os.path.abspath(input_file_path)
        output_abs = os.path.abspath(output_file_path)
        
        # Get directories to mount
        input_dir = os.path.dirname(input_abs)
        output_dir = os.path.dirname(output_abs)
        
        # Get filenames
        input_filename = os.path.basename(input_abs)
        output_filename = os.path.basename(output_abs)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Docker command for SynthStrip
        docker_cmd = [
            'docker', 'run', '--rm',
            '-v', f'{input_dir}:/input',
            '-v', f'{output_dir}:/output',
            'freesurfer/synthstrip:latest',
            '-i', f'/input/{input_filename}',
            '-o', f'/output/{output_filename}',
            '--no-csf'  # Exclude CSF for better MS lesion analysis
        ]
        
        logger.debug(f"Running: {' '.join(docker_cmd)}")
        
        # Run the Docker command
        result = subprocess.run(docker_cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logger.info(f"Successfully processed {subject_id}")
            return True
        else:
            logger.error(f"SynthStrip failed for {subject_id}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"SynthStrip timed out for {subject_id}")
        return False
    except Exception as e:
        logger.error(f"Error during skull stripping for {subject_id}: {str(e)}")
        return False

def main():
    """
    Main pipeline for skull stripping registered NIfTI files using Docker SynthStrip.
    """
    logger.info("Starting SynthStrip skull stripping pipeline")
    
    # Check if Docker is available
    if not check_docker_availability():
        logger.error("Docker is not available. Please install Docker Desktop for Mac.")
        logger.error("Download from: https://www.docker.com/products/docker-desktop")
        return

    # Pull the SynthStrip Docker image
    if not pull_synthstrip_image():
        logger.error("Failed to pull SynthStrip Docker image.")
        return

    logger.info("SynthStrip Docker image ready. Starting processing...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Check input directory exists
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    # Find files to process
    try:
        files_to_process = [f for f in os.listdir(INPUT_DIR) 
                          if f.endswith('.nii.gz') and '_registered_n4' in f]
    except FileNotFoundError:
        logger.error(f"Cannot access input directory: {INPUT_DIR}")
        return

    if not files_to_process:
        logger.warning(f"No matching files found in {INPUT_DIR}")
        logger.warning("Looking for files ending with '_registered_n4.nii.gz'")
        return

    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Check for already processed files
    already_processed = set()
    try:
        already_processed = {
            f.replace('_synthstrip_skull.nii.gz', '') 
            for f in os.listdir(OUTPUT_DIR) 
            if f.endswith('_synthstrip_skull.nii.gz')
        }
        if already_processed:
            logger.info(f"Found {len(already_processed)} already processed files")
    except FileNotFoundError:
        pass

    # Process files
    processed_count = 0
    successful_count = 0
    start_time = time.time()

    with tqdm(total=len(files_to_process), desc="SynthStrip Processing") as pbar:
        for filename in files_to_process:
            subject_id = filename.replace('_registered_n4.nii.gz', '')
            pbar.set_description(f"Processing {subject_id}")

            # Skip if already processed
            if subject_id in already_processed:
                logger.debug(f"Skipping {subject_id} (already processed)")
                processed_count += 1
                pbar.update(1)
                continue

            # Set file paths
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"{subject_id}_synthstrip_skull.nii.gz")
            
            # Process file
            try:
                success = docker_synthstrip_skull_strip(input_path, output_path, subject_id)
                if success:
                    # Verify output file exists and has reasonable size
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                        successful_count += 1
                    else:
                        logger.error(f"Output file missing or too small for {subject_id}")
                        success = False
            except Exception as e:
                logger.error(f"Unexpected error processing {subject_id}: {str(e)}")
                success = False
            
            processed_count += 1
            
            # Update progress bar
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / processed_count if processed_count > 0 else 0
            remaining_files = len(files_to_process) - processed_count
            eta = avg_time_per_file * remaining_files

            pbar.set_postfix({
                'Success': f"{successful_count}/{processed_count}",
                'ETA': f"{eta/60:.1f}min" if eta > 0 else "0min",
                'Avg': f"{avg_time_per_file:.1f}s/file"
            })
            pbar.update(1)

    # Final summary
    total_time = time.time() - start_time
    logger.info("\n" + "="*50)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*50)
    logger.info(f"Total files found: {len(files_to_process)}")
    logger.info(f"Already processed: {len(already_processed)}")
    logger.info(f"Newly processed: {processed_count - len(already_processed)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {processed_count - successful_count - len(already_processed)}")
    logger.info(f"Total time: {total_time/60:.2f} minutes")
    logger.info(f"Average time per file: {total_time/max(processed_count, 1):.1f} seconds")
    logger.info("="*50)

if __name__ == "__main__":
    main() 