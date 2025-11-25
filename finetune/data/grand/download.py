"""Download script for the GranD dataset."""
import csv
import logging
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_grand(data_cfg: DictConfig) -> None:
    """Download the GranD dataset into the specified directory.

    Args:
        data_cfg: Configuration containing data directory path and download URLs.
    """
    data_dir = data_cfg.data_dir
    data_path = Path(data_dir)

    repo_dir = Path(data_cfg.grand.repo_dir)
    links_file = Path(data_cfg.grand.image_links_file)
    images_dir = Path(data_cfg.grand.images_dir)

    # Check if dataset is already downloaded
    if (
        repo_dir.exists()
        and links_file.exists()
        and images_dir.exists()
        and any(images_dir.iterdir())
    ):
        logger.info(f"GranD dataset already exists at {data_dir}")
        return

    data_path.mkdir(parents=True, exist_ok=True)

    # 1. Download the repository (metadata and links)
    repo_id = data_cfg.grand.repo_id
    
    logger.info(f"Downloading GranD repository from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            local_dir=repo_dir,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        logger.error(f"Failed to download repository: {e}")
        return

    # 2. Download the image links file
    image_list_url = data_cfg.grand.image_list_url
    
    logger.info(f"Downloading image links from {image_list_url}...")
    try:
        urllib.request.urlretrieve(image_list_url, links_file)
    except Exception as e:
        logger.error(f"Failed to download image links: {e}")
        return

    logger.info(f"Image links saved to: {links_file}")

    # 3. Download and extract images
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(links_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        rows = list(reader)
        
        for row in tqdm(rows, desc="Downloading and extracting images"):
            file_name = row.get("file_name")
            url = row.get("cdn_link")
            
            if not file_name or not url:
                continue

            tar_path = images_dir / file_name
            marker_path = images_dir / f".{file_name}.done"
            
            if marker_path.exists():
                continue

            if not tar_path.exists():
                try:
                    urllib.request.urlretrieve(url, tar_path)
                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")
                    continue
            
            # Extract
            try:
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=images_dir)
                
                # Create marker
                marker_path.touch()
                
                # Remove tar
                tar_path.unlink()
                
            except Exception as e:
                logger.error(f"Failed to extract {tar_path}: {e}")

    logger.info("GranD dataset download complete.")
    