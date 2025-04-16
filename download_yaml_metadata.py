import os
import requests
from huggingface_hub import list_datasets, hf_hub_url

def download_yaml_metadata(base_dir="datasets"):
    # Get all datasets from the polymathic-ai org
    datasets = list_datasets(author="polymathic-ai", full=True)

    for d in datasets:
        dataset_id = d.id  # e.g. 'polymathic-ai/turbulent_radiative_layer_2D'
        dataset_name = dataset_id.split("/")[-1]
        target_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)

        for yaml_filename in [f"{dataset_name}.yaml", "stats.yaml"]:
            try:
                url = hf_hub_url(dataset_id, yaml_filename, repo_type="dataset")
                response = requests.get(url)
                if response.status_code == 200:
                    yaml_path = os.path.join(target_dir, yaml_filename)
                    with open(yaml_path, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Downloaded {yaml_filename} to {yaml_path}")
                else:
                    print(f"⚠️ Could not download {yaml_filename} (status {response.status_code})")
            except Exception as e:
                print(f"❌ Error downloading {yaml_filename} for {dataset_id}: {e}")

if __name__ == "__main__":
    base_dir = "data/datasets"
    download_yaml_metadata(base_dir=base_dir)
