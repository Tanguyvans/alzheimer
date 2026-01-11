#!/usr/bin/env python3
"""
Create preprocessing pipeline figure for the paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis('off')

# Colors
box_color = '#E3F2FD'  # Light blue
arrow_color = '#1976D2'  # Dark blue
text_color = '#0D47A1'  # Darker blue

# Define pipeline steps
steps = [
    ('DICOM\nInput', 0.5),
    ('NIfTI\nConversion', 2.5),
    ('N4 Bias\nCorrection', 4.5),
    ('MNI-152\nRegistration', 6.5),
    ('Skull\nStripping', 8.5),
    ('128³\nVolume', 10.5),
]

# Draw boxes and labels
box_width = 1.6
box_height = 1.8

for label, x in steps:
    # Draw box
    box = FancyBboxPatch(
        (x - box_width/2, 0.6),
        box_width, box_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=box_color,
        edgecolor=arrow_color,
        linewidth=2
    )
    ax.add_patch(box)

    # Add label
    ax.text(x, 1.5, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=text_color)

# Draw arrows between boxes
for i in range(len(steps) - 1):
    x1 = steps[i][1] + box_width/2 + 0.05
    x2 = steps[i+1][1] - box_width/2 - 0.05

    ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color,
                               lw=2, mutation_scale=15))

# Add subtitle descriptions
descriptions = [
    'Raw scan',
    'dicom2nifti',
    'ANTs N4',
    '1.75mm iso',
    'SynthStrip',
    'Final'
]

for (label, x), desc in zip(steps, descriptions):
    ax.text(x, 0.25, desc, ha='center', va='center',
            fontsize=8, style='italic', color='gray')

# Title
ax.text(6, 2.7, 'MRI Preprocessing Pipeline', ha='center', va='center',
        fontsize=14, fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig('preprocessing_pipeline.pdf', bbox_inches='tight', dpi=300)
plt.savefig('preprocessing_pipeline.png', bbox_inches='tight', dpi=300)
print("Saved preprocessing_pipeline.pdf and preprocessing_pipeline.png")
