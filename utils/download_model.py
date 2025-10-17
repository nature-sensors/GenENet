import click
from pathlib import Path
import requests
from tqdm import tqdm

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_file(url, output_path, description=None):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    desc = description or f"Downloading {output_path.name}"
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return output_path

def download_model(model_name, model_dir):
    base_url = "https://gen-enet-kim.s3.us-east-2.amazonaws.com/model"
    print(f"Downloading pretrained model {model_name}...")

    model_dir = ensure_dir(Path(model_dir).expanduser())
    model_path = model_dir / f"{model_name}.pt"

    model_url = f"{base_url}/{model_name}.pt"
    download_file(model_url, model_path, f"Downloading {model_name}")

    print(f"Model saved to {model_path.absolute()}")
    return model_path

@click.command(help="Download pretrained model (.pt).")
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Model name without extension (e.g., mymodel_v1)",
)
def main(model_name):
    output_dir = Path.cwd() / "model_pt"
    download_model(model_name, output_dir)

if __name__ == "__main__":
    main()
