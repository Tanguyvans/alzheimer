import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Charger le fichier
#img = nib.load('normal.nii.gz')
#img = nib.load('/Users/tanguyvans/Desktop/umons/code/alzheimer/new_irm_output/SEP-MRI-001_T0_6f1.nii.gz')
#img = nib.load('/Users/tanguyvans/Desktop/umons/code/alzheimer/new_irm_output/SEP-MRI-001_T1_be7d.nii.gz')
#img = nib.load('/Volumes/KINGSTON/ADNI_nifti/002_S_2010/MPRAGE_2010-06-24_14_21_28.0_I180310_180310.nii.gz')
img = nib.load('/Volumes/KINGSTON/ADNI_nifti/006_S_4153/MPRAGE_2011-08-03_08_12_01.0_I248517_248517.nii.gz')

data = img.get_fdata()

# Squeeze out extra dimensions (e.g., 4D with shape[3]==1 -> 3D)
import numpy as np
data = np.squeeze(data)

# Afficher les dimensions et orientation
print("Image shape:", data.shape)
print("Image spacing:", img.header.get_zooms())
print("Image orientation:", img.header.get_best_affine())

# Créer la figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25)

# Position initiale des coupes
z_pos = data.shape[2]//2
y_pos = data.shape[1]//2
x_pos = data.shape[0]//2

# Axial view: slice through z-axis (horizontal brain slices)
img1 = ax1.imshow(data[:, :, z_pos].T, cmap='gray', origin='lower', aspect='equal')
# Coronal view: slice through y-axis (front-back brain slices)  
img2 = ax2.imshow(data[:, y_pos, :].T, cmap='gray', origin='lower', aspect='equal')
# Sagittal view: slice through x-axis (left-right brain slices)
img3 = ax3.imshow(data[x_pos, :, :], cmap='gray', origin='lower', aspect='equal')

ax1.set_title('Vue axiale')
ax2.set_title('Vue coronale')
ax3.set_title('Vue sagittale')

# Ajouter les sliders
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Coupe', 0, data.shape[2]-1, valinit=z_pos, valstep=1)

def update(val):
    pos = int(slider.val)
    # Update with correct orientations
    img1.set_array(data[:, :, pos].T)
    img2.set_array(data[:, pos, :].T) 
    img3.set_array(data[pos, :, :])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()