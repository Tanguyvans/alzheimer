import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches
from pathlib import Path
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

class BrainImageViewer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.original_dir = self.base_dir / "ms-1-register"
        self.synthstrip_dir = self.base_dir / "ms-1-synthstrip-skull"
        self.hdbet_dir = self.base_dir / "ms-1-skull"  # Updated path
        
        # Find all available subjects
        self.subjects = self.find_subjects()
        self.current_subject_idx = 0
        
        if not self.subjects:
            raise ValueError("No processed subjects found!")
        
        print(f"Found {len(self.subjects)} subjects")
        
        # Initialize slice indices for each view
        self.current_slice_axial = None
        self.current_slice_coronal = None
        self.current_slice_sagittal = None
        
        # Set up the figure
        self.setup_figure()
        
        # Load and display first subject
        self.update_display()
    
    def find_subjects(self):
        """Find subjects that have both SynthStrip and HD-BET processed images"""
        subjects = []
        
        if not self.original_dir.exists():
            print(f"Original directory not found: {self.original_dir}")
            return subjects
            
        if not self.synthstrip_dir.exists():
            print(f"SynthStrip directory not found: {self.synthstrip_dir}")
            return subjects
        
        # Get registered files
        registered_files = list(self.original_dir.glob("*_registered_n4.nii.gz"))
        
        for reg_file in registered_files:
            # Extract subject ID from registered file
            subject_id = reg_file.name.replace("_registered_n4.nii.gz", "")
            
            # Look for corresponding SynthStrip file
            synthstrip_file = self.synthstrip_dir / f"{subject_id}_synthstrip_skull.nii.gz"
            
            # Look for corresponding HD-BET file (same name as registered but in different dir)
            hdbet_file = self.hdbet_dir / f"{subject_id}_registered_n4.nii.gz"
            
            # Check if at least SynthStrip exists
            if synthstrip_file.exists():
                subjects.append(subject_id)
                if not hdbet_file.exists():
                    print(f"Note: HD-BET result not found for {subject_id}")
                    print(f"Expected: {hdbet_file}")
        
        return sorted(subjects)
    
    def setup_figure(self):
        """Set up the matplotlib figure with subplots and buttons"""
        self.fig = plt.figure(figsize=(18, 14))
        self.fig.suptitle("Brain Image Multi-View Comparison: HD-BET vs SynthStrip", fontsize=16, fontweight='bold')
        
        # Create 2x3 grid: 2 rows (HD-BET, SynthStrip) x 3 columns (Axial, Coronal, Sagittal)
        # Adjust subplot positions to make room for sliders
        self.ax_hdbet_axial = plt.subplot2grid((3, 3), (0, 0))
        self.ax_hdbet_coronal = plt.subplot2grid((3, 3), (0, 1))
        self.ax_hdbet_sagittal = plt.subplot2grid((3, 3), (0, 2))
        
        self.ax_synthstrip_axial = plt.subplot2grid((3, 3), (1, 0))
        self.ax_synthstrip_coronal = plt.subplot2grid((3, 3), (1, 1))
        self.ax_synthstrip_sagittal = plt.subplot2grid((3, 3), (1, 2))
        
        # Set titles
        self.ax_hdbet_axial.set_title("HD-BET - Axial", fontsize=12, fontweight='bold')
        self.ax_hdbet_coronal.set_title("HD-BET - Coronal", fontsize=12, fontweight='bold')
        self.ax_hdbet_sagittal.set_title("HD-BET - Sagittal", fontsize=12, fontweight='bold')
        
        self.ax_synthstrip_axial.set_title("SynthStrip - Axial", fontsize=12, fontweight='bold')
        self.ax_synthstrip_coronal.set_title("SynthStrip - Coronal", fontsize=12, fontweight='bold')
        self.ax_synthstrip_sagittal.set_title("SynthStrip - Sagittal", fontsize=12, fontweight='bold')
        
        # Remove axes ticks
        all_axes = [self.ax_hdbet_axial, self.ax_hdbet_coronal, self.ax_hdbet_sagittal,
                   self.ax_synthstrip_axial, self.ax_synthstrip_coronal, self.ax_synthstrip_sagittal]
        
        for ax in all_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add navigation buttons and sliders
        self.setup_controls()
    
    def setup_controls(self):
        """Set up navigation buttons and sliders"""
        # Subject navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_subject)
        
        ax_next = plt.axes([0.2, 0.02, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_subject)
        
        # Subject info text
        self.ax_info = plt.axes([0.35, 0.02, 0.3, 0.06])
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
        
        # Slice sliders - positioned below the images
        # Axial slider
        ax_slider_axial = plt.axes([0.1, 0.12, 0.25, 0.03])
        self.slider_axial = Slider(ax_slider_axial, 'Axial', 0, 100, valinit=50, valfmt='%d')
        self.slider_axial.on_changed(self.update_axial_slice)
        
        # Coronal slider  
        ax_slider_coronal = plt.axes([0.4, 0.12, 0.25, 0.03])
        self.slider_coronal = Slider(ax_slider_coronal, 'Coronal', 0, 100, valinit=50, valfmt='%d')
        self.slider_coronal.on_changed(self.update_coronal_slice)
        
        # Sagittal slider
        ax_slider_sagittal = plt.axes([0.7, 0.12, 0.25, 0.03])
        self.slider_sagittal = Slider(ax_slider_sagittal, 'Sagittal', 0, 100, valinit=50, valfmt='%d')
        self.slider_sagittal.on_changed(self.update_sagittal_slice)
        
        # Store sliders for easy access
        self.sliders = {
            'axial': self.slider_axial,
            'coronal': self.slider_coronal,
            'sagittal': self.slider_sagittal
        }
    
    def update_slider_ranges(self, data_shape):
        """Update slider ranges based on current image dimensions"""
        # Update axial slider (Z axis)
        self.slider_axial.valmax = data_shape[2] - 1
        self.slider_axial.ax.set_xlim(0, data_shape[2] - 1)
        if self.current_slice_axial is None:
            self.current_slice_axial = data_shape[2] // 2
        self.slider_axial.set_val(self.current_slice_axial)
        
        # Update coronal slider (Y axis)
        self.slider_coronal.valmax = data_shape[1] - 1
        self.slider_coronal.ax.set_xlim(0, data_shape[1] - 1)
        if self.current_slice_coronal is None:
            self.current_slice_coronal = data_shape[1] // 2
        self.slider_coronal.set_val(self.current_slice_coronal)
        
        # Update sagittal slider (X axis)
        self.slider_sagittal.valmax = data_shape[0] - 1
        self.slider_sagittal.ax.set_xlim(0, data_shape[0] - 1)
        if self.current_slice_sagittal is None:
            self.current_slice_sagittal = data_shape[0] // 2
        self.slider_sagittal.set_val(self.current_slice_sagittal)
    
    def update_axial_slice(self, val):
        """Update axial slice when slider changes"""
        self.current_slice_axial = int(val)
        self.update_images()
    
    def update_coronal_slice(self, val):
        """Update coronal slice when slider changes"""
        self.current_slice_coronal = int(val)
        self.update_images()
    
    def update_sagittal_slice(self, val):
        """Update sagittal slice when slider changes"""
        self.current_slice_sagittal = int(val)
        self.update_images()
    
    def load_images(self, subject_id):
        """Load images for a specific subject"""
        try:
            # Load SynthStrip image
            synthstrip_path = self.synthstrip_dir / f"{subject_id}_synthstrip_skull.nii.gz"
            synthstrip_img = nib.load(synthstrip_path)
            synthstrip_data = synthstrip_img.get_fdata()
            
            # Try to load HD-BET image
            hdbet_path = self.hdbet_dir / f"{subject_id}_registered_n4.nii.gz"
            if hdbet_path.exists():
                hdbet_img = nib.load(hdbet_path)
                hdbet_data = hdbet_img.get_fdata()
                print(f"Loaded HD-BET result for {subject_id}")
            else:
                # If HD-BET doesn't exist, use the original registered image as comparison
                reg_path = self.original_dir / f"{subject_id}_registered_n4.nii.gz"
                hdbet_img = nib.load(reg_path)
                hdbet_data = hdbet_img.get_fdata()
                print(f"Using original image for {subject_id} (HD-BET not found)")
            
            # Normalize images to [0, 1] for display
            synthstrip_data = self.normalize_image(synthstrip_data)
            hdbet_data = self.normalize_image(hdbet_data)
            
            return synthstrip_data, hdbet_data
            
        except Exception as e:
            print(f"Error loading images for {subject_id}: {e}")
            return None, None
    
    def normalize_image(self, img_data):
        """Normalize image data for display"""
        # Remove background
        img_data = img_data.copy()
        img_data[img_data < 0] = 0
        
        # Normalize to [0, 1]
        if img_data.max() > 0:
            img_data = img_data / img_data.max()
        
        return img_data
    
    def get_slices(self, data):
        """Extract axial, coronal, and sagittal slices from 3D data"""
        # Initialize slice indices if they're None
        if self.current_slice_axial is None:
            self.current_slice_axial = data.shape[2] // 2
        if self.current_slice_coronal is None:
            self.current_slice_coronal = data.shape[1] // 2
        if self.current_slice_sagittal is None:
            self.current_slice_sagittal = data.shape[0] // 2
            
        # Ensure slice indices are within bounds
        axial_idx = min(self.current_slice_axial, data.shape[2] - 1)
        coronal_idx = min(self.current_slice_coronal, data.shape[1] - 1)
        sagittal_idx = min(self.current_slice_sagittal, data.shape[0] - 1)
        
        # Extract slices
        axial = data[:, :, axial_idx]           # XY plane
        coronal = data[:, coronal_idx, :]       # XZ plane
        sagittal = data[sagittal_idx, :, :]     # YZ plane
        
        return axial, coronal, sagittal
    
    def update_display(self):
        """Update the display with current subject"""
        if not self.subjects:
            return
        
        subject_id = self.subjects[self.current_subject_idx]
        print(f"Loading subject: {subject_id}")
        
        # Load images
        self.synthstrip_data, self.hdbet_data = self.load_images(subject_id)
        
        if self.synthstrip_data is None or self.hdbet_data is None:
            return
        
        # Update slider ranges based on new image
        self.update_slider_ranges(self.synthstrip_data.shape)
        
        # Update images
        self.update_images()
        
        # Update info text
        self.update_info_display(subject_id, self.synthstrip_data.shape)
    
    def update_images(self):
        """Update only the images (called by sliders)"""
        if not hasattr(self, 'synthstrip_data') or self.synthstrip_data is None:
            return
        
        # Get slices for both datasets
        synthstrip_axial, synthstrip_coronal, synthstrip_sagittal = self.get_slices(self.synthstrip_data)
        hdbet_axial, hdbet_coronal, hdbet_sagittal = self.get_slices(self.hdbet_data)
        
        # Clear all axes
        all_axes = [self.ax_hdbet_axial, self.ax_hdbet_coronal, self.ax_hdbet_sagittal,
                   self.ax_synthstrip_axial, self.ax_synthstrip_coronal, self.ax_synthstrip_sagittal]
        
        for ax in all_axes:
            ax.clear()
        
        # Display HD-BET images
        self.ax_hdbet_axial.imshow(hdbet_axial.T, cmap='gray', origin='lower')
        self.ax_hdbet_axial.set_title("HD-BET - Axial", fontsize=12, fontweight='bold')
        
        self.ax_hdbet_coronal.imshow(hdbet_coronal.T, cmap='gray', origin='lower')
        self.ax_hdbet_coronal.set_title("HD-BET - Coronal", fontsize=12, fontweight='bold')
        
        self.ax_hdbet_sagittal.imshow(hdbet_sagittal.T, cmap='gray', origin='lower')
        self.ax_hdbet_sagittal.set_title("HD-BET - Sagittal", fontsize=12, fontweight='bold')
        
        # Display SynthStrip images
        self.ax_synthstrip_axial.imshow(synthstrip_axial.T, cmap='gray', origin='lower')
        self.ax_synthstrip_axial.set_title("SynthStrip - Axial", fontsize=12, fontweight='bold')
        
        self.ax_synthstrip_coronal.imshow(synthstrip_coronal.T, cmap='gray', origin='lower')
        self.ax_synthstrip_coronal.set_title("SynthStrip - Coronal", fontsize=12, fontweight='bold')
        
        self.ax_synthstrip_sagittal.imshow(synthstrip_sagittal.T, cmap='gray', origin='lower')
        self.ax_synthstrip_sagittal.set_title("SynthStrip - Sagittal", fontsize=12, fontweight='bold')
        
        # Remove ticks
        for ax in all_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.draw()
    
    def update_info_display(self, subject_id, data_shape):
        """Update the information display"""
        self.ax_info.clear()
        
        info_text = (f"Subject: {subject_id} ({self.current_subject_idx + 1}/{len(self.subjects)})\n"
                    f"Axial: {self.current_slice_axial + 1}/{data_shape[2]} | "
                    f"Coronal: {self.current_slice_coronal + 1}/{data_shape[1]} | "
                    f"Sagittal: {self.current_slice_sagittal + 1}/{data_shape[0]}")
        
        self.ax_info.text(0.5, 0.5, info_text,
                         ha='center', va='center', fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
    
    def prev_subject(self, event):
        """Go to previous subject"""
        if self.current_subject_idx > 0:
            self.current_subject_idx -= 1
            # Reset slice indices for new subject
            self.current_slice_axial = None
            self.current_slice_coronal = None
            self.current_slice_sagittal = None
            self.update_display()
    
    def next_subject(self, event):
        """Go to next subject"""
        if self.current_subject_idx < len(self.subjects) - 1:
            self.current_subject_idx += 1
            # Reset slice indices for new subject
            self.current_slice_axial = None
            self.current_slice_coronal = None
            self.current_slice_sagittal = None
            self.update_display()
    
    def show(self):
        """Show the viewer"""
        plt.show()

def main():
    # Set the base directory (same as your processing script)
    BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"
    
    try:
        # Create and show the viewer
        viewer = BrainImageViewer(BASE_DIR)
        
        print("\n" + "="*70)
        print("BRAIN IMAGE MULTI-VIEW VIEWER: HD-BET vs SynthStrip")
        print("="*70)
        print("Controls:")
        print("• Previous/Next: Navigate between subjects")
        print("• Axial/Coronal/Sagittal sliders: Navigate slices independently")
        print("• Each slider controls slices for that specific view")
        print("• Close window to exit")
        print("\nLayout:")
        print("• Top row: HD-BET results (Axial, Coronal, Sagittal)")
        print("• Bottom row: SynthStrip results (Axial, Coronal, Sagittal)")
        print("• Sliders at bottom control each view independently")
        print("="*70)
        
        viewer.show()
        
    except Exception as e:
        print(f"Error starting viewer: {e}")
        print("Make sure you have processed images in the expected directories:")
        print(f"- SynthStrip: {BASE_DIR}/ms-1-synthstrip-skull/")
        print(f"- HD-BET: {BASE_DIR}/ms-1-skull/")

if __name__ == "__main__":
    main() 