import argparse
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent


def prepare_vsfc(data_dir):
    import gdown

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    file_ids = {"train": "1eJdEStCQBf3GtbOIQfIR2iS-Wb3nlfJX", "dev": "1UWbzHFeUPu4AuofOEy0Fo0SFotVZHzUe", "test": "1g4imdabNbswIu91K6zKFGalkjDBBqaQq"}
    for split, file_id in file_ids.items():
        path = data_dir / f"UIT-VSFC-{split}.json"
        if not path.exists():
            gdown.download(id=file_id, output=str(path), quiet=False)
        if not path.is_file():
            raise FileNotFoundError(f"Download failed: {path}")


def prepare_phoner(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word"
    for split in ["train", "dev", "test"]:
        path = data_dir / f"{split}_word.conll"
        if not path.exists():
            with urlopen(f"{url}/{path.name}", timeout=60) as response:
                content = response.read()
            path.write_bytes(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vsfc", "phoner", "all"], default="all")
    args = parser.parse_args()
    if args.dataset in ["vsfc", "all"]:
        prepare_vsfc(ROOT / "datasets" / "UIT-VSFC")
    if args.dataset in ["phoner", "all"]:
        prepare_phoner(ROOT / "datasets" / "PhoNER")
