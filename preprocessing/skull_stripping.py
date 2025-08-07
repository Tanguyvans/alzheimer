import os
import logging
import subprocess
from typing import Optional, List
import glob

logger = logging.getLogger(__name__)

def check_docker_availability() -> bool:
    """
    Check if Docker is installed and running.
    
    Returns:
        True if Docker is available, False otherwise
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
        logger.error("Please install Docker Desktop from: https://docker.com/products/docker-desktop")
        return False
    except FileNotFoundError:
        logger.error("Docker command not found")
        logger.error("Please install Docker Desktop from: https://docker.com/products/docker-desktop")
        return False

def pull_synthstrip_image() -> bool:
    """
    Pull the official SynthStrip Docker image.
    
    Returns:
        True if successful, False otherwise
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

def synthstrip_skull_strip(input_file_path: str, output_file_path: str, subject_id: str = None, exclude_csf: bool = True) -> bool:
    """
    Perform skull stripping using SynthStrip Docker container.
    
    Args:
        input_file_path: Path to input NIfTI file
        output_file_path: Path to save skull-stripped file
        subject_id: Subject identifier for logging
        exclude_csf: Whether to exclude CSF (better for MS lesion analysis)
        
    Returns:
        True if successful, False otherwise
    """
    if subject_id is None:
        subject_id = os.path.basename(input_file_path).replace('.nii.gz', '')
    
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
            '-o', f'/output/{output_filename}'
        ]
        
        # Add CSF exclusion flag if requested
        if exclude_csf:
            docker_cmd.append('--no-csf')
        
        logger.debug(f"Running: {' '.join(docker_cmd)}")
        
        # Run the Docker command
        result = subprocess.run(docker_cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Verify output file exists and has reasonable size
            if os.path.exists(output_abs) and os.path.getsize(output_abs) > 1024:
                logger.info(f"Successfully skull-stripped {subject_id}")
                return True
            else:
                logger.error(f"Output file missing or too small for {subject_id}")
                return False
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

def skull_strip_directory(input_dir: str, output_dir: str, file_pattern: str = "*enhanced.nii.gz") -> dict:
    """
    Skull strip all enhanced NIfTI files in a directory.
    
    Args:
        input_dir: Input directory containing enhanced NIfTI files
        output_dir: Output directory for skull-stripped files
        file_pattern: File pattern to match (default: *enhanced.nii.gz)
        
    Returns:
        Dictionary with processing results
    """
    # Check Docker availability first
    if not check_docker_availability():
        logger.error("Docker is not available")
        return {'successful': [], 'failed': [], 'total': 0, 'error': 'docker_unavailable'}
    
    # Pull SynthStrip image
    if not pull_synthstrip_image():
        logger.error("Failed to pull SynthStrip Docker image")
        return {'successful': [], 'failed': [], 'total': 0, 'error': 'image_pull_failed'}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all enhanced NIfTI files
    search_pattern = os.path.join(input_dir, file_pattern)
    nifti_files = glob.glob(search_pattern)
    
    logger.info(f"Found {len(nifti_files)} enhanced files to skull strip")
    
    results = {
        'successful': [],
        'failed': [],
        'total': len(nifti_files)
    }
    
    for input_file in nifti_files:
        filename = os.path.basename(input_file)
        subject_id = filename.replace('_enhanced.nii.gz', '').replace('.nii.gz', '')
        output_file = os.path.join(output_dir, f"{subject_id}_skull_stripped.nii.gz")
        
        # Skip if already exists
        if os.path.exists(output_file):
            logger.info(f"Skull-stripped file already exists: {output_file}")
            results['successful'].append(subject_id)
            continue
        
        success = synthstrip_skull_strip(input_file, output_file, subject_id)
        
        if success:
            results['successful'].append(subject_id)
        else:
            results['failed'].append(subject_id)
    
    logger.info(f"Skull stripping complete: {len(results['successful'])} successful, {len(results['failed'])} failed")
    return results

def setup_synthstrip_docker() -> bool:
    """
    Setup SynthStrip Docker environment (check Docker + pull image).
    
    Returns:
        True if setup successful, False otherwise
    """
    logger.info("Setting up SynthStrip Docker environment...")
    
    if not check_docker_availability():
        return False
    
    if not pull_synthstrip_image():
        return False
    
    logger.info("SynthStrip Docker environment ready")
    return True