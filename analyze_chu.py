import os
import pydicom
from collections import defaultdict, Counter

# Dossier source
source_folder = "../../../shared/databases/alzheimer/CHU_refactored"

# Fichier de sortie
output_file = "summary_CHU_refactored.txt"

# Fonction pour compter les slices d'un type d'image
def count_slices(image_folder):
    slices = 0
    for file in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file)
        try:
            ds = pydicom.dcmread(file_path)
            slices += 1  # Chaque fichier DICOM est une coupe
        except Exception as e:
            print(f"Erreur de lecture : {file_path}, {e}")
    return slices

# Résumé des types d'images par patient
patient_data = defaultdict(dict)
image_type_counts = Counter()

for patient_folder in os.listdir(source_folder):
    patient_path = os.path.join(source_folder, patient_folder)
    if os.path.isdir(patient_path):  # Vérifier que c'est un dossier
        print(f"Traitement du patient : {patient_folder}")
        for image_type in os.listdir(patient_path):
            image_type_path = os.path.join(patient_path, image_type)
            if os.path.isdir(image_type_path):  # Vérifier que c'est un dossier
                slices = count_slices(image_type_path)
                patient_data[patient_folder][image_type] = slices
                image_type_counts[image_type] += 1

# Trouver les types d'images communs à tous les patients
all_patients = set(patient_data.keys())
common_image_types = set.intersection(*[set(images.keys()) for images in patient_data.values()])

# Écriture du fichier de sortie
with open(output_file, "w") as f:
    f.write("Résumé des types d'images par patient\n")
    f.write("=" * 50 + "\n")
    for patient, images in patient_data.items():
        f.write(f"Patient : {patient}\n")
        for image_type, slices in images.items():
            f.write(f"  - {image_type} : {slices} slices\n")
        f.write("\n")
    
    f.write("\nTypes d'images communs entre tous les patients\n")
    f.write("=" * 50 + "\n")
    for image_type in common_image_types:
        f.write(f"- {image_type}\n")
    
    f.write("\nTypes d'images non communs (et leur fréquence)\n")
    f.write("=" * 50 + "\n")
    for image_type, count in image_type_counts.items():
        if image_type not in common_image_types:
            f.write(f"- {image_type} : {count} patients\n")
