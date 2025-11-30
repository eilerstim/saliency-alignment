"""Download script for the GranD dataset."""

import csv
import logging
import tarfile
import urllib.request
from pathlib import Path

import hydra
import requests
from huggingface_hub import HfApi, hf_hub_download
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
    annotations_dir = Path(data_cfg.grand.annotations_dir)

    data_path.mkdir(parents=True, exist_ok=True)

    # 1. Setup
    repo_id = data_cfg.grand.repo_id

    # 2. Download the image links file
    image_list_url = data_cfg.grand.image_list_url

    if not links_file.exists():
        logger.info(f"Downloading image links from {image_list_url}...")
        try:
            urllib.request.urlretrieve(image_list_url, links_file)
        except Exception as e:
            logger.error(f"Failed to download image links: {e}")
            return
        logger.info(f"Image links saved to: {links_file}")

    # 3. Download and extract images (Streaming)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    max_archives = data_cfg.grand.get("max_image_archives", None)

    with open(links_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

        if max_archives is not None:
            rows = rows[:max_archives]

        for row in tqdm(rows, desc="Downloading and extracting images"):
            file_name = row.get("file_name")
            url = row.get("cdn_link")

            if not file_name or not url:
                continue

            # We don't have a local tar path anymore since we stream
            # But we use the marker to know if we are done
            marker_path = images_dir / f".{file_name}.done"

            if marker_path.exists():
                continue

            try:
                # Stream download and extract directly without saving tar to disk
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    # mode="r|gz" is essential for streaming
                    with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                        tar.extractall(path=images_dir)

                # Delete JSON files in images_dir immediately
                for json_file in images_dir.rglob("*.json"):
                    json_file.unlink()

                # Create marker
                marker_path.touch()

            except Exception as e:
                logger.error(f"Failed to download/extract {url}: {e}")

    # 4. Process Annotations (Download one by one)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of valid image stems
    valid_stems = {p.stem for p in images_dir.rglob("*") if p.is_file() and not p.name.startswith(".")}
    
    logger.info(f"Found {len(valid_stems)} valid images. Fetching and processing annotation archives...")

    # List files in the repo to find annotation tarballs
    api = HfApi()
    try:
        all_repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        # Filter for annotation tarballs (e.g., part_1/part_1_1.tar.gz)
        annotation_tar_files = [f for f in all_repo_files if "part_" in f and f.endswith(".tar.gz")]
        annotation_tar_files.sort()
    except Exception as e:
        logger.error(f"Failed to list repo files: {e}")
        return

    for tar_filename in tqdm(annotation_tar_files, desc="Processing annotation archives"):
        local_tar_path = repo_dir / tar_filename
        
        # Download the single tarball
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=tar_filename,
                repo_type="dataset",
                local_dir=repo_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.error(f"Failed to download {tar_filename}: {e}")
            continue

        # Process and delete
        try:
            with tarfile.open(local_tar_path, "r") as tar:
                # Iterate members and extract only if matching
                # Using iterator instead of getmembers() for memory efficiency
                for member in tar:
                    if not member.isfile():
                        continue
                    
                    member_stem = Path(member.name).stem
                    if member_stem in valid_stems:
                        # Extract to annotations_dir
                        # Flatten structure
                        member.name = Path(member.name).name
                        tar.extract(member, path=annotations_dir)
        except Exception as e:
            logger.error(f"Failed to process annotation archive {local_tar_path}: {e}")
        finally:
            # Delete tar file immediately after processing
            if local_tar_path.exists():
                local_tar_path.unlink()

    logger.info("GranD dataset download complete.")


@hydra.main(
    version_base="1.3",
    config_path="/cluster/project/sachan/pmlr/saliency-alignment/configs/data/",
    config_name="grand",
)
def main(cfg: DictConfig) -> None:
    download_grand(cfg)


if __name__ == "__main__":
    main()
