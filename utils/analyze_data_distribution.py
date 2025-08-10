import pandas as pd

def analyze_dataset(csv_path="dataset_MRI_cohort1_updated.csv"):
    # Lire le fichier CSV
    df = pd.read_csv(csv_path)
    
    # Afficher les informations générales
    print("\n=== Informations générales sur le dataset ===")
    print(f"Nombre total d'images: {len(df)}")
    print("\n=== Distribution des groupes de recherche ===")
    research_groups = df['research_group'].value_counts()
    print(research_groups)    

if __name__ == "__main__":
    analyze_dataset() 