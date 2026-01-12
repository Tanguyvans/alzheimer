#!/usr/bin/env python3
"""
Interactive slice viewer to find matching slices between native and registered images
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
from pathlib import Path

# Paths to the preprocessed images
OUTPUT_DIR = Path(__file__).parent / "preprocessing_stages"
PATIENT_ID = "029_S_0836"

paths = {
    'nifti': OUTPUT_DIR / f"{PATIENT_ID}_01_nifti.nii.gz",
    'registered': OUTPUT_DIR / f"{PATIENT_ID}_03_registered.nii.gz",
}

# Load images
print("Loading images...")
native_img = nib.load(str(paths['nifti']))
registered_img = nib.load(str(paths['registered']))

native_data = native_img.get_fdata()
registered_data = registered_img.get_fdata()

print(f"Native shape: {native_data.shape}")
print(f"Registered shape: {registered_data.shape}")

def normalize(s):
    s = s.astype(float)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return s

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Initial slices
init_native = native_data.shape[2] // 2
init_registered = 63

# Display initial slices
native_slice = normalize(np.rot90(native_data[:, :, init_native]))
registered_slice = normalize(np.rot90(registered_data[:, :, init_registered]))

im_native = axes[0].imshow(native_slice, cmap='gray')
axes[0].set_title(f'Original (slice {init_native})')
axes[0].axis('off')

im_registered = axes[1].imshow(registered_slice, cmap='gray')
axes[1].set_title(f'Registered (slice {init_registered})')
axes[1].axis('off')

# Create sliders
ax_native = plt.axes([0.15, 0.1, 0.3, 0.03])
ax_registered = plt.axes([0.55, 0.1, 0.3, 0.03])

slider_native = Slider(ax_native, 'Native', 0, native_data.shape[2]-1, valinit=init_native, valstep=1)
slider_registered = Slider(ax_registered, 'Registered', 0, registered_data.shape[2]-1, valinit=init_registered, valstep=1)

def update_native(val):
    idx = int(slider_native.val)
    native_slice = normalize(np.rot90(native_data[:, :, idx]))
    im_native.set_data(native_slice)
    axes[0].set_title(f'Original (slice {idx})')
    fig.canvas.draw_idle()

def update_registered(val):
    idx = int(slider_registered.val)
    registered_slice = normalize(np.rot90(registered_data[:, :, idx]))
    im_registered.set_data(registered_slice)
    axes[1].set_title(f'Registered (slice {idx})')
    fig.canvas.draw_idle()

slider_native.on_changed(update_native)
slider_registered.on_changed(update_registered)

plt.suptitle('Scroll to find matching slices\nNote the slice numbers for create_preprocessing_fig.py', fontsize=12)
plt.show()

print(f"\nFinal values:")
print(f"  NATIVE_SLICE = {int(slider_native.val)}")
print(f"  REGISTERED_SLICE = {int(slider_registered.val)}")
