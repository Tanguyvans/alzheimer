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
    level=logging.INFO,  # Keep INFO for detailed processing messages
    format='%(asctime)s - %(levelname)s - %(message)s' # More informative format
)

# Define base directory and other paths
BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"  # Your project's BASE_DIR
INPUT_DIR = os.path.join(BASE_DIR, "ms-1-register")
OUTPUT_DIR = os.path.join(BASE_DIR, "ms-1-skullstripped")

def skull_strip_hd_bet(input_file_path, output_file_path, subject_id):
    """
    Performs skull stripping using HD-BET, adapted from user's template.
    """
    logger.info(f"Starting HD-BET skull stripping for {subject_id} from {input_file_path}")
    
    # Construct the command string, similar to the user's template.
    # Changed device to "cpu". If you have a compatible GPU, you might use "cuda" or "cuda:0" (for NVIDIA)
    # or "mps" (for Apple M-series with appropriate PyTorch support).
    cmd_str = f"hd-bet -i \"{input_file_path}\" -o \"{output_file_path}\" -device cpu"
    # If you need to specify the output directory instead of the full file path for -o:
    # output_dir_for_hd_bet = os.path.dirname(output_file_path)
    # cmd_str = f"hd-bet -i \"{input_file_path}\" -o \"{output_dir_for_hd_bet}\" -device cpu"


    try:
        logger.info(f"Executing HD-BET command: {cmd_str}")
        # Using shell=True as in the user's template
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=600) # Added timeout

        if process.returncode != 0:
            error_message = f"HD-BET failed for {subject_id}. Return code: {process.returncode}\nStdout: {stdout.decode(errors='ignore')}\nStderr: {stderr.decode(errors='ignore')}"
            logger.error(error_message)
            raise Exception(error_message) # Re-raise to be caught by the main loop's try-except

        # Check if HD-BET created the file directly or appended _bet
        # If -o is a full file path, HD-BET should ideally create output_file_path directly.
        # If HD-BET saved it as output_file_path_bet.nii.gz (e.g. if -o was treated as a directory)
        expected_bet_output = output_file_path.replace('.nii.gz', '_bet.nii.gz')
        
        if os.path.exists(output_file_path):
            logger.info(f"HD-BET skull stripping successful for {subject_id}. Output directly saved to: {output_file_path}")
            return True
        elif os.path.exists(expected_bet_output):
            logger.info(f"HD-BET created {expected_bet_output}. Renaming to {output_file_path}.")
            os.rename(expected_bet_output, output_file_path)
            if os.path.exists(output_file_path):
                logger.info(f"Rename successful. Output saved to: {output_file_path}")
                return True
            else:
                logger.error(f"Rename failed. Expected file {output_file_path} not found after attempting rename.")
                return False
        else:
            logger.error(f"HD-BET completed for {subject_id} but expected output file ({output_file_path} or {expected_bet_output}) not found.\nStdout: {stdout.decode(errors='ignore')}\nStderr: {stderr.decode(errors='ignore')}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"HD-BET command timed out for {subject_id} after 10 minutes.")
        process.kill()
        stdout, stderr = process.communicate()
        logger.error(f"Timeout stdout: {stdout.decode(errors='ignore')}\nTimeout stderr: {stderr.decode(errors='ignore')}")
        return False
    except FileNotFoundError: # Should not happen with shell=True if hd-bet is in PATH
        logger.error("HD-BET command not found. Make sure it is installed and in your system's PATH.")
        raise # Re-raise critical error
    except Exception as e:
        logger.error(f"An error occurred during HD-BET skull stripping for {subject_id}: {str(e)}")
        return False # Captured by main loop, will increment processed_count, not successful_stripping_count


def main():
    """
    Main pipeline for skull stripping registered NIfTI files.
    """
    if not BASE_DIR: # Should be defined globally
        logger.error("BASE_DIR is not set.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Skull-stripped output directory set to: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input NIfTI directory not found: {INPUT_DIR}. Please ensure 'ms-1-register' exists and contains files.")
        return

    try:
        files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz') and '_registered_n4' in f]
    except FileNotFoundError: # Should be caught by the check above
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    if not files_to_process:
        logger.warning(f"No '.nii.gz' files found in {INPUT_DIR} matching the expected pattern (e.g., *_registered_n4.nii.gz)")
        return

    logger.info(f"Found {len(files_to_process)} NIfTI files to process from {INPUT_DIR}.")
    
    processed_count = 0
    successful_stripping_count = 0
    start_time_total = time.time()

    # Using a set for faster lookups for already processed files
    try:
        already_skull_stripped_ids = {
            f.replace('_skullstripped.nii.gz', '') 
            for f in os.listdir(OUTPUT_DIR) 
            if f.endswith('_skullstripped.nii.gz')
        }
        logger.info(f"Found {len(already_skull_stripped_ids)} already skull-stripped files in {OUTPUT_DIR}.")
    except FileNotFoundError:
        logger.warning(f"Output directory {OUTPUT_DIR} not found for checking existing files. Will create it.")
        already_skull_stripped_ids = set()


    with tqdm(total=len(files_to_process), desc="Skull Stripping") as pbar:
        for nifti_filename in files_to_process:
            subject_id = nifti_filename.replace('_registered_n4.nii.gz', '')
            pbar.set_description(f"Processing {subject_id}")

            if subject_id in already_skull_stripped_ids:
                logger.info(f"Subject {subject_id} already processed. Skipping.")
                processed_count += 1 # Count as processed for progress bar
                pbar.update(1)
                continue

            input_file_path = os.path.join(INPUT_DIR, nifti_filename)
            output_file_path = os.path.join(OUTPUT_DIR, f"{subject_id}_skullstripped.nii.gz")
            
            success = False # Default to False
            try:
            success = skull_strip_hd_bet(input_file_path, output_file_path, subject_id)
            except Exception as e: # Catch exceptions raised from skull_strip_hd_bet
                logger.error(f"Critical error during skull stripping for {subject_id}: {str(e)}. Moving to next file.")
                # success remains False
            
            if success:
                successful_stripping_count += 1
            
            processed_count += 1 # Increment for every attempt
            
            elapsed_time_total = time.time() - start_time_total
            avg_time_per_file = elapsed_time_total / max(processed_count, 1)
            remaining_files = len(files_to_process) - processed_count
            estimated_time_remaining = avg_time_per_file * remaining_files

            pbar.set_postfix({
                'Processed': f"{processed_count}/{len(files_to_process)}",
                'Successful': successful_stripping_count,
                'ETA': f"{estimated_time_remaining/3600:.1f}h" if estimated_time_remaining > 0 else "0.0h",
                'Avg Time/file': f"{avg_time_per_file:.1f}s"
            })
            pbar.update(1)

    logger.info(f"\nSkull stripping completed:")
    logger.info(f"Total files initially targeted: {len(files_to_process)}")
    logger.info(f"Total attempts made (processed_count): {processed_count}")
    logger.info(f"Total successfully skull-stripped and saved: {successful_stripping_count}")
    
    # Recalculate skipped based on current state of OUTPUT_DIR and files_to_process
    final_skipped_count = 0
    final_failed_count = 0
    for nifti_filename in files_to_process:
        subject_id = nifti_filename.replace('_registered_n4.nii.gz', '')
        expected_output_file = os.path.join(OUTPUT_DIR, f"{subject_id}_skullstripped.nii.gz")
        if subject_id in already_skull_stripped_ids and not os.path.exists(expected_output_file):
            # This case indicates it was "skipped" initially but the file doesn't actually exist now.
            # Could happen if script was interrupted after marking as skipped but before actual processing.
            # For simplicity, we count based on what was *attempted* and what *succeeded*.
            pass 
        elif subject_id in already_skull_stripped_ids and os.path.exists(expected_output_file):
            final_skipped_count +=1


    # A file attempt is failed if it was processed but did not result in a successful output.
    # total_attempts_not_skipped_initially = processed_count - (len(files_to_process) - len(already_skull_stripped_ids))

    logger.info(f"Total skipped (already existed at start): {len(already_skull_stripped_ids) if already_skull_stripped_ids else 0}")
    # Number of files that an attempt was made on = processed_count - number_of_files_skipped_because_they_existed_at_start
    actual_attempts = processed_count - (len(files_to_process) - (len(files_to_process) - len(already_skull_stripped_ids)))
    
    # A more direct way to count failed: total attempts - successful
    # However, processed_count includes those skipped because they existed.
    # So, we need to count files for which an *attempt* was made but did *not* succeed.
    
    files_attempted_processing = processed_count - len(already_skull_stripped_ids)
    total_failed = files_attempted_processing - successful_stripping_count

    logger.info(f"Total failed during skull stripping (among attempted files): {max(0, total_failed)}")
    logger.info(f"Total time: {(time.time() - start_time_total)/3600:.2f} hours")

if __name__ == "__main__":
    main()