import logging
import urllib.request
import zipfile
from pathlib import Path

import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/",
    config_name="llava-150k",
)
def download_llava_instruct_150k(data_cfg: DictConfig):
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="complex_reasoning_77k.json",
        repo_type="dataset",
    )

    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="conversation_58k.json",
        repo_type="dataset",
    )

    data_dir = Path(data_cfg.data_dir)
    images_dir = Path(data_cfg.images_dir)

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract image datasets (train2017, val2017) into images/
    image_downloads = [
        (data_cfg.train.name, data_cfg.train.url),
        (data_cfg.validation.name, data_cfg.validation.url),
    ]

    for name, url in image_downloads:
        zip_path = data_dir / f"{name}.zip"
        extract_dir = images_dir / name

        if extract_dir.exists():
            logger.info(f"{name} already exists at {extract_dir}")
            continue

        if not zip_path.exists():
            logger.info(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(images_dir)
        zip_path.unlink()

        logger.info(f"Successfully extracted {name}")


if __name__ == "__main__":
    download_llava_instruct_150k()
