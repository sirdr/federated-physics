import os
from huggingface_hub import list_models, hf_hub_download

def download_unet_benchmarks(base_dir="data/model-data/benchmarks"):
    models = list_models(author="polymathic-ai", full=True)
    unet_models = [m for m in models if "UNetConvNext" in m.modelId]

    for model in unet_models:
        model_id = model.modelId  # e.g., 'polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D'

        try:
            name_parts = model_id.split("/")[-1].split("-")
            model_name = name_parts[0]
            dataset_name = "-".join(name_parts[1:])
            target_dir = os.path.join(base_dir, f"{model_name}-{dataset_name}")
        except Exception as e:
            print(f"❌ Skipping malformed model ID: {model_id} ({e})")
            continue

        os.makedirs(target_dir, exist_ok=True)

        for filename in ["model.safetensors", "config.json"]:
            try:
                local_path = hf_hub_download(repo_id=model_id, filename=filename)
                dest_path = os.path.join(target_dir, filename)
                with open(local_path, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                print(f"✅ Downloaded {filename} to {dest_path}")
            except Exception as e:
                print(f"⚠️ Failed to download {filename} from {model_id}: {e}")

if __name__ == "__main__":
    download_unet_benchmarks()
