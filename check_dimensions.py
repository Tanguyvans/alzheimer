import os
import nibabel as nib
from pathlib import Path
from collections import defaultdict

def classify_image_type(dimensions):
    """
    Classifie le type d'image en comparant les dimensions entre elles
    """
    a, b, c = dimensions
    dims = list(dimensions)
    
    # Pour "all": vérifie si les dimensions sont proportionnelles
    min_dim = min(dims)
    max_dim = max(dims)
    ratio = min_dim / max_dim
    
    if ratio > 0.7 and all(d > 150 for d in dims):
        return "all"
    
    # Trier les dimensions pour trouver les deux plus grandes
    sorted_dims = sorted(enumerate(dims), key=lambda x: x[1], reverse=True)
    top_two_indices = [idx for idx, _ in sorted_dims[:2]]
    smallest_idx = [idx for idx, _ in sorted_dims[2:]][0]
    
    # Calculer le ratio entre la plus petite dimension et la moyenne des deux plus grandes
    avg_top_two = (dims[top_two_indices[0]] + dims[top_two_indices[1]]) / 2
    ratio_to_top = dims[smallest_idx] / avg_top_two
    
    # Si le ratio est suffisamment petit (< 0.5), c'est une vue 2D
    if ratio_to_top < 0.5:
        # Vérifier que les deux plus grandes dimensions sont similaires (ratio > 0.8)
        larger_dims = [dims[i] for i in top_two_indices]
        if min(larger_dims) / max(larger_dims) > 0.8:
            if smallest_idx == 0:
                return f"sagittal (ratio: {ratio_to_top:.2f})"
            elif smallest_idx == 1:
                return f"coronal (ratio: {ratio_to_top:.2f})"
            else:
                return f"axial (ratio: {ratio_to_top:.2f})"
    
    return "autre"

def get_image_category(filename):
    """
    Détermine la catégorie de l'image basée sur son nom de fichier
    """
    filename_upper = filename.upper()
    if 'T1' in filename_upper:
        return 'T1'
    elif 'FLAIR' in filename_upper:
        return 'FLAIR'
    return 'AUTRE'

def check_nifti_dimensions(directory):
    """
    Parcourt tous les fichiers .nii.gz et classifie les types d'images par patient
    """
    nifti_dir = Path(directory)
    patient_images = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Créer un fichier pour les résultats
    output_file = Path(directory) / "image_dimensions_summary.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Détails des images:\n")
        f.write("-" * 60 + "\n\n")
        
        for patient_dir in nifti_dir.iterdir():
            if not patient_dir.is_dir():
                continue
                
            for nifti_file in patient_dir.glob('*.nii.gz'):
                try:
                    img = nib.load(str(nifti_file))
                    dimensions = img.shape
                    img_type = classify_image_type(dimensions)
                    ratio = min(dimensions)/max(dimensions)
                    category = get_image_category(nifti_file.name)
                    
                    # Format condensé pour chaque image
                    details = (
                        f"Patient {patient_dir.name} - {nifti_file.name}:\n"
                        f"Catégorie: {category}\n"
                        f"Dimensions: {dimensions}\n"
                        f"Type: {img_type}\n"
                        f"Ratio min/max: {ratio:.2f}\n"
                    )
                    
                    # Écrire dans le fichier et afficher dans la console
                    print(details)
                    f.write(details)
                    
                    # Stocker les dimensions et le nom du fichier par catégorie
                    patient_images[patient_dir.name][category][img_type].append((dimensions, nifti_file.name))
                    
                except Exception as e:
                    error_msg = f"Erreur avec {nifti_file}: {str(e)}\n"
                    print(error_msg)
                    f.write(error_msg)
        
        # Écrire le résumé
        f.write("\nRésumé par patient:\n")
        f.write("-" * 40 + "\n")
        
        for patient_id, categories in sorted(patient_images.items()):
            f.write(f"\nPatient {patient_id}:\n")
            
            for category in ['T1', 'FLAIR', 'AUTRE']:  # Ordre spécifique des catégories
                if category in categories:
                    f.write(f"\n{category}:\n")
                    for img_type, dimensions_list in categories[category].items():
                        # Regrouper les dimensions identiques
                        dim_count = {}
                        for dim, filename in dimensions_list:
                            if dim not in dim_count:
                                dim_count[dim] = {'count': 0, 'files': []}
                            dim_count[dim]['count'] += 1
                            dim_count[dim]['files'].append(filename)
                        
                        # Format condensé pour chaque type d'image
                        for dim, info in dim_count.items():
                            f.write(f"  {info['count']} {img_type}: {[dim]} - {info['files']}\n")
        
    print(f"\nRésultats sauvegardés dans: {output_file}")
    return patient_images

if __name__ == "__main__":
    nifti_directory = '/Volumes/KINGSTON/CHU_nifti/'
    check_nifti_dimensions(nifti_directory) 