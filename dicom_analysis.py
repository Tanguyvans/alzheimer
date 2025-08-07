import SimpleITK as sitk
import os

def check_dicom_sequences(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    
    if not dicom_names:
        print(f"No DICOM series found in {dicom_folder}")
        return
    
    # Read the first DICOM file to get metadata
    reader.SetFileNames([dicom_names[0]])
    image = reader.Execute()
    
    # Get DICOM metadata
    metadata = {}
    for key in image.GetMetaDataKeys():
        metadata[key] = image.GetMetaData(key)
    
    # Print relevant sequence information
    print("\nDICOM Sequence Information:")
    print(f"Series Description: {metadata.get('0008|103e', 'Not found')}")  # Series Description
    print(f"Sequence Name: {metadata.get('0018|0024', 'Not found')}")      # Sequence Name
    print(f"Scanning Sequence: {metadata.get('0018|0020', 'Not found')}")  # Scanning Sequence
    print(f"Sequence Variant: {metadata.get('0018|0021', 'Not found')}")   # Sequence Variant
    print(f"Scan Options: {metadata.get('0018|0022', 'Not found')}")       # Scan Options
    print(f"MR Acquisition Type: {metadata.get('0018|0023', 'Not found')}") # MR Acquisition Type
    print(f"Repetition Time: {metadata.get('0018|0080', 'Not found')}")    # Repetition Time
    print(f"Echo Time: {metadata.get('0018|0081', 'Not found')}")          # Echo Time
    print(f"Inversion Time: {metadata.get('0018|0082', 'Not found')}")     # Inversion Time

# Usage
dicom_folder = "/Users/tanguyvans/Desktop/umons/code/alzheimer/irm_sep_etude/SEP-MRI-001/T0"
check_dicom_sequences(dicom_folder)