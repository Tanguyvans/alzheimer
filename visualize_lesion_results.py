#!/usr/bin/env python3
"""
MS Lesion Visualization Tool
Visualizes FLAMeS lesion segmentation results overlaid on original skull-stripped images
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

class LesionViewer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.skull_stripped_dir = self.base_dir / "ms-1-synthstrip-skull"
        self.lesion_dir = self.base_dir / "ms-2-flames-lesions"
        
        # Find available subjects with lesion results
        self.subjects = self.find_subjects_with_lesions()
        
        if not self.subjects:
            raise ValueError("No subjects with lesion results found!")
        
        print(f"Found {len(self.subjects)} subjects with lesion results")
        
        # Initialize display state
        self.current_subject_idx = 0
        self.current_slice = None
        
        # Setup the figure
        self.setup_figure()
        self.setup_controls()
        
        # Load first subject
        self.update_display()
    
    def find_subjects_with_lesions(self):
        """Find subjects that have both skull-stripped images and lesion results"""
        subjects = []
        
        if not self.lesion_dir.exists():
            return subjects
        
        for lesion_file in self.lesion_dir.glob("*_lesions.nii.gz"):
            # Extract subject ID from lesion filename 
            # lesion_file.name = "SEP-MRI-043_T2_6194_lesions.nii.gz"
            # We want: "SEP-MRI-043_T2_6194"
            subject_id = lesion_file.name.replace("_lesions.nii.gz", "")
            
            # Check if corresponding skull-stripped image exists
            skull_file = self.skull_stripped_dir / f"{subject_id}_synthstrip_skull.nii.gz"
            if skull_file.exists():
                subjects.append(subject_id)
                print(f"Found matching pair: {subject_id}")
            else:
                print(f"No skull-stripped file found for: {subject_id}")
                print(f"  Looking for: {skull_file}")
        
        return sorted(subjects)
    
    def setup_figure(self):
        """Setup the matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("MS Lesion Segmentation Results (FLAMeS)", fontsize=16, fontweight='bold')
        
        # Create grid layout: 2 rows, 3 columns for each view
        # Top row: Original images, Bottom row: Lesion overlays
        
        # Original images (top row)
        self.ax_orig_axial = plt.subplot(2, 3, 1)
        self.ax_orig_coronal = plt.subplot(2, 3, 2)
        self.ax_orig_sagittal = plt.subplot(2, 3, 3)
        
        # Lesion overlays (bottom row)
        self.ax_lesion_axial = plt.subplot(2, 3, 4)
        self.ax_lesion_coronal = plt.subplot(2, 3, 5)
        self.ax_lesion_sagittal = plt.subplot(2, 3, 6)
        
        # Info panel
        self.ax_info = plt.axes([0.02, 0.02, 0.96, 0.08])
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, top=0.9)
    
    def setup_controls(self):
        """Setup interactive controls"""
        # Navigation buttons
        ax_prev = plt.axes([0.1, 0.15, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.15, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_subject)
        self.btn_next.on_clicked(self.next_subject)
        
        # Slice slider (will be updated when image is loaded)
        ax_slice = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.slider_slice = Slider(ax_slice, 'Axial Slice', 0, 100, valinit=50, valfmt='%d')
        self.slider_slice.on_changed(self.update_slice)
    
    def load_images(self, subject_id):
        """Load skull-stripped image and lesion mask for a subject"""
        try:
            # Load skull-stripped image
            skull_path = self.skull_stripped_dir / f"{subject_id}_synthstrip_skull.nii.gz"
            skull_img = nib.load(skull_path)
            skull_data = skull_img.get_fdata()
            
            # Load lesion mask
            lesion_path = self.lesion_dir / f"{subject_id}_lesions.nii.gz"
            lesion_img = nib.load(lesion_path)
            lesion_data = lesion_img.get_fdata()
            
            # Normalize skull image for display
            skull_data = self.normalize_image(skull_data)
            
            # Ensure lesion mask is binary
            lesion_data = (lesion_data > 0).astype(float)
            
            print(f"Loaded images for {subject_id}")
            print(f"  Skull image shape: {skull_data.shape}")
            print(f"  Lesion mask shape: {lesion_data.shape}")
            print(f"  Number of lesion voxels: {np.sum(lesion_data)}")
            
            return skull_data, lesion_data
            
        except Exception as e:
            print(f"Error loading images for {subject_id}: {e}")
            return None, None
    
    def normalize_image(self, img_data):
        """Normalize image data for display"""
        img_data = img_data.copy()
        img_data[img_data < 0] = 0
        
        if img_data.max() > 0:
            img_data = img_data / img_data.max()
        
        return img_data
    
    def get_slices(self, data, slice_idx):
        """Get axial, coronal, and sagittal slices at given indices"""
        # Axial slice (XY plane)
        axial = data[:, :, slice_idx]
        
        # Coronal slice (XZ plane) - middle slice
        coronal_idx = data.shape[1] // 2
        coronal = data[:, coronal_idx, :]
        
        # Sagittal slice (YZ plane) - middle slice
        sagittal_idx = data.shape[0] // 2
        sagittal = data[sagittal_idx, :, :]
        
        return axial, coronal, sagittal
    
    def create_overlay(self, background, mask, alpha=0.3):
        """Create colored overlay of lesion mask on background image"""
        # Create RGB image
        overlay = np.zeros((*background.shape, 3))
        
        # Background in grayscale
        overlay[:, :, 0] = background
        overlay[:, :, 1] = background
        overlay[:, :, 2] = background
        
        # Add red lesions
        lesion_mask = mask > 0
        overlay[lesion_mask, 0] = 1.0  # Red channel
        overlay[lesion_mask, 1] = alpha * background[lesion_mask]  # Reduce green
        overlay[lesion_mask, 2] = alpha * background[lesion_mask]  # Reduce blue
        
        return overlay
    
    def update_display(self):
        """Update the display with current subject"""
        if not self.subjects:
            return
        
        subject_id = self.subjects[self.current_subject_idx]
        print(f"\nLoading subject: {subject_id}")
        
        # Load images
        self.skull_data, self.lesion_data = self.load_images(subject_id)
        
        if self.skull_data is None or self.lesion_data is None:
            return
        
        # Update slider range
        max_slice = self.skull_data.shape[2] - 1
        self.slider_slice.valmin = 0
        self.slider_slice.valmax = max_slice
        self.current_slice = max_slice // 2
        self.slider_slice.set_val(self.current_slice)
        
        # Update images
        self.update_images()
        
        # Update info
        self.update_info_display(subject_id)
    
    def update_images(self):
        """Update the displayed images"""
        if not hasattr(self, 'skull_data') or self.skull_data is None:
            return
        
        # Get slices
        skull_axial, skull_coronal, skull_sagittal = self.get_slices(self.skull_data, self.current_slice)
        lesion_axial, lesion_coronal, lesion_sagittal = self.get_slices(self.lesion_data, self.current_slice)
        
        # Create overlays
        overlay_axial = self.create_overlay(skull_axial, lesion_axial)
        overlay_coronal = self.create_overlay(skull_coronal, lesion_coronal)
        overlay_sagittal = self.create_overlay(skull_sagittal, lesion_sagittal)
        
        # Clear all axes
        axes = [self.ax_orig_axial, self.ax_orig_coronal, self.ax_orig_sagittal,
                self.ax_lesion_axial, self.ax_lesion_coronal, self.ax_lesion_sagittal]
        
        for ax in axes:
            ax.clear()
        
        # Display original images (top row)
        self.ax_orig_axial.imshow(skull_axial.T, cmap='gray', origin='lower')
        self.ax_orig_axial.set_title("Original - Axial", fontsize=12, fontweight='bold')
        
        self.ax_orig_coronal.imshow(skull_coronal.T, cmap='gray', origin='lower')
        self.ax_orig_coronal.set_title("Original - Coronal", fontsize=12, fontweight='bold')
        
        self.ax_orig_sagittal.imshow(skull_sagittal.T, cmap='gray', origin='lower')
        self.ax_orig_sagittal.set_title("Original - Sagittal", fontsize=12, fontweight='bold')
        
        # Display lesion overlays (bottom row)
        self.ax_lesion_axial.imshow(overlay_axial.transpose(1, 0, 2), origin='lower')
        self.ax_lesion_axial.set_title("Lesions - Axial", fontsize=12, fontweight='bold', color='red')
        
        self.ax_lesion_coronal.imshow(overlay_coronal.transpose(1, 0, 2), origin='lower')
        self.ax_lesion_coronal.set_title("Lesions - Coronal", fontsize=12, fontweight='bold', color='red')
        
        self.ax_lesion_sagittal.imshow(overlay_sagittal.transpose(1, 0, 2), origin='lower')
        self.ax_lesion_sagittal.set_title("Lesions - Sagittal", fontsize=12, fontweight='bold', color='red')
        
        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.draw()
    
    def update_info_display(self, subject_id):
        """Update the information display"""
        self.ax_info.clear()
        
        # Calculate lesion statistics
        total_lesion_voxels = np.sum(self.lesion_data > 0)
        total_brain_voxels = np.sum(self.skull_data > 0)
        lesion_percentage = (total_lesion_voxels / total_brain_voxels * 100) if total_brain_voxels > 0 else 0
        
        # Get current slice lesion count
        current_lesion_voxels = np.sum(self.lesion_data[:, :, self.current_slice] > 0)
        
        info_text = (f"Subject: {subject_id} ({self.current_subject_idx + 1}/{len(self.subjects)}) | "
                    f"Slice: {self.current_slice + 1}/{self.skull_data.shape[2]} | "
                    f"Total Lesion Voxels: {total_lesion_voxels:,} | "
                    f"Lesion Load: {lesion_percentage:.2f}% | "
                    f"Current Slice Lesions: {current_lesion_voxels}")
        
        self.ax_info.text(0.5, 0.5, info_text,
                         ha='center', va='center', fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis('off')
    
    def update_slice(self, val):
        """Update slice when slider changes"""
        self.current_slice = int(val)
        self.update_images()
        self.update_info_display(self.subjects[self.current_subject_idx])
    
    def prev_subject(self, event):
        """Go to previous subject"""
        if self.current_subject_idx > 0:
            self.current_subject_idx -= 1
            self.update_display()
    
    def next_subject(self, event):
        """Go to next subject"""
        if self.current_subject_idx < len(self.subjects) - 1:
            self.current_subject_idx += 1
            self.update_display()
    
    def show(self):
        """Show the viewer"""
        plt.show()

def main():
    # Set the base directory
    BASE_DIR = "/Users/tanguyvans/Desktop/umons/code/alzheimer"
    
    try:
        # Create and show the lesion viewer
        viewer = LesionViewer(BASE_DIR)
        
        print("\n" + "="*70)
        print("MS LESION SEGMENTATION VIEWER (FLAMeS Results)")
        print("="*70)
        print("Controls:")
        print("• Previous/Next: Navigate between subjects")
        print("• Axial Slice slider: Navigate through axial slices")
        print("• Red areas indicate detected MS lesions")
        print("• Close window to exit")
        print("\nLayout:")
        print("• Top row: Original skull-stripped images")
        print("• Bottom row: Lesion overlays (red = lesions)")
        print("• Info panel shows lesion statistics")
        print("="*70)
        
        viewer.show()
        
    except Exception as e:
        print(f"Error starting lesion viewer: {e}")
        print("Make sure you have:")
        print(f"- Skull-stripped images in: {BASE_DIR}/ms-1-synthstrip-skull/")
        print(f"- Lesion results in: {BASE_DIR}/ms-2-flames-lesions/")

if __name__ == "__main__":
    main() 