import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Charger le fichier
#img = nib.load('normal.nii.gz')
img = nib.load('output/register/normal/normal_stripped.nii.gz')
data = img.get_fdata()

# Afficher les dimensions
print("Image shape:", data.shape)
print("Image spacing:", img.header.get_zooms())

# Créer la figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25)

# Position initiale des coupes
z_pos = data.shape[2]//2
y_pos = data.shape[1]//2
x_pos = data.shape[0]//2

# Afficher les images initiales
img1 = ax1.imshow(data[:, :, z_pos], cmap='gray')
img2 = ax2.imshow(data[:, y_pos, :], cmap='gray')
img3 = ax3.imshow(data[x_pos, :, :], cmap='gray')

ax1.set_title('Vue axiale')
ax2.set_title('Vue coronale')
ax3.set_title('Vue sagittale')

# Ajouter les sliders
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Coupe', 0, data.shape[2]-1, valinit=z_pos, valstep=1)

def update(val):
    pos = int(slider.val)
    img1.set_array(data[:, :, pos])
    img2.set_array(data[:, pos, :])
    img3.set_array(data[pos, :, :])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()