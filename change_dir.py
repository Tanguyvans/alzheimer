import os
import shutil

def reorganize_npy_files(base_directory):
    """
    Réorganise les fichiers NPY de:
    npy_seg/[ID]/[id]_hippo.npy -> npy_seg/[id].npy
    """
    print(f"Début de la réorganisation des fichiers dans {base_directory}")
    
    # Parcourir tous les dossiers dans le répertoire de base
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        # Vérifier si c'est un dossier
        if os.path.isdir(folder_path):
            try:
                # Chemin du fichier source
                hippo_file = os.path.join(folder_path, f"{folder_name}_hippo.npy")
                
                # Vérifier si le fichier existe
                if os.path.exists(hippo_file):
                    # Définir le nouveau chemin (sans _hippo)
                    new_file = os.path.join(base_directory, f"{folder_name}.npy")
                    
                    # Déplacer et renommer le fichier
                    shutil.move(hippo_file, new_file)
                    
                    # Supprimer le dossier vide
                    os.rmdir(folder_path)
                    
                    print(f"✓ {folder_name}: Fichier déplacé avec succès")
                else:
                    print(f"✗ {folder_name}: Fichier manquant")
                    
            except Exception as e:
                print(f"✗ {folder_name}: Erreur - {str(e)}")

if __name__ == "__main__":
    base_dir = "npy_seg"
    
    if not os.path.exists(base_dir):
        print(f"Erreur: Le répertoire {base_dir} n'existe pas")
    else:
        reorganize_npy_files(base_dir)
        print("\nRéorganisation terminée")