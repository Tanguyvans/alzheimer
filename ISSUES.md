# Medical Image Processing Issues & Solutions

## issue_1: Brain Split Visualization
**Problem**: Brain appeared as two disconnected halves in all anatomical views
**Root Cause**: DICOM to NIfTI conversion was sorting slices alphabetically by filename instead of by spatial position
**Solution**: Modified `preprocessing/dicom_to_nifti.py` to use `GetGDCMSeriesFileNames()` which reads DICOM headers and sorts slices by actual spatial location
**Files Changed**: `preprocessing/dicom_to_nifti.py` (lines 57-88)

## issue_2: Incorrect Anatomical View Orientation  
**Problem**: Sagittal and axial views were displayed in wrong orientations
**Root Cause**: Missing transpose operations for proper neurological display convention
**Solution**: Added `.T` transpose to axial and coronal views in visualization code, kept proper `origin='lower'` and `aspect='equal'`
**Files Changed**: `utils/visualize.py` (lines 38-42, 55-57)