from pathlib import Path
import requests

def download_csv(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Le fichier a été téléchargé et enregistré sous {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement du fichier : {e}")

BASE_DIR = Path(__file__).parent.resolve()
raw_data_dir = BASE_DIR / "../data/raw_data/raw_data.csv"
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

download_csv(url,raw_data_dir)