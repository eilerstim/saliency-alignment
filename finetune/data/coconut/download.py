import json
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

COCONUT_SPLIT = "coconut_s"


def download_coco(data_cfg: DictConfig):
    """Download and extract COCO dataset if not already present.

    Downloads training images, validation images, and annotations from COCO dataset URLs
    specified in the configuration. Skips download if files already exist locally.

    Target structure:
        data_dir/
        |_ images/
        |  |_ train2017/
        |  |_ val2017/
        |_ annotations/
           |_ panoptic_segmentation/
           |  |_ train2017/
           |  |_ val2017/
           |_ panoptic_train2017.json
           |_ panoptic_val2017.json
           |_ instances_train2017.json
           |_ ...

    Args:
        data_cfg: Configuration containing data directory path and download URLs for
            train images, validation images, and annotations.
    """
    data_dir = Path(data_cfg.data_dir)
    images_dir = Path(data_cfg.images_dir)
    annotations_dir = Path(data_cfg.annotations_dir)
    panoptic_seg_dir = Path(data_cfg.panoptic.segmentation_dir)

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    panoptic_seg_dir.mkdir(parents=True, exist_ok=True)

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

    # Download and extract annotations
    ann_name = data_cfg.annotations.name
    ann_url = data_cfg.annotations.url
    ann_zip_path = data_dir / f"{ann_name}.zip"

    # Check if annotations are already extracted (check for a known file)
    if not (annotations_dir / "instances_train2017.json").exists():
        if not ann_zip_path.exists():
            logger.info(f"Downloading {ann_name}...")
            urllib.request.urlretrieve(ann_url, ann_zip_path)

        logger.info(f"Extracting {ann_name}...")
        with zipfile.ZipFile(ann_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        ann_zip_path.unlink()

        logger.info(f"Successfully extracted {ann_name}")
    else:
        logger.info(f"{ann_name} already exists at {annotations_dir}")

    # Download and extract panoptic annotations
    panoptic_name = data_cfg.panoptic.name
    panoptic_url = data_cfg.panoptic.url
    panoptic_zip_path = data_dir / f"{panoptic_name}.zip"

    # Check if panoptic segmentation masks are already extracted
    panoptic_train_dir = panoptic_seg_dir / "train2017"
    panoptic_val_dir = panoptic_seg_dir / "val2017"

    if not panoptic_train_dir.exists() or not panoptic_val_dir.exists():
        if not panoptic_zip_path.exists():
            logger.info(f"Downloading {panoptic_name}...")
            urllib.request.urlretrieve(panoptic_url, panoptic_zip_path)

        logger.info(f"Extracting {panoptic_name}...")
        with zipfile.ZipFile(panoptic_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        panoptic_zip_path.unlink()

        # The panoptic zip extracts to annotations/ with:
        # - panoptic_train2017.json, panoptic_val2017.json (JSON files)
        # - panoptic_train2017.zip, panoptic_val2017.zip (mask images as nested zips)
        # Extract the nested zips to panoptic_segmentation/{split}/
        for split in ["train2017", "val2017"]:
            nested_zip = annotations_dir / f"panoptic_{split}.zip"
            split_dir = panoptic_seg_dir / split

            if nested_zip.exists() and not split_dir.exists():
                logger.info(f"Extracting panoptic masks for {split}...")
                with zipfile.ZipFile(nested_zip, "r") as zip_ref:
                    zip_ref.extractall(annotations_dir)

                # Move extracted folder to panoptic_segmentation/{split}
                extracted_dir = annotations_dir / f"panoptic_{split}"
                if extracted_dir.exists():
                    extracted_dir.rename(split_dir)

                nested_zip.unlink()
                logger.info(f"Successfully extracted panoptic masks for {split}")

        logger.info(f"Successfully extracted {panoptic_name}")
    else:
        logger.info(f"{panoptic_name} already exists at {panoptic_seg_dir}")

    logger.info("COCO dataset download and extraction complete")


def download_coconut(data_cfg: DictConfig):
    """Download COCONut dataset from HuggingFace.

    Args:
        data_cfg: Data configuration containing paths and URLs for the dataset.
    """
    # Check if files already exist
    output_json_file = Path(data_cfg.coconut.ann_file)
    output_mask_dir = Path(data_cfg.coconut.masks_dir)
    output_captions_dir = Path(data_cfg.coconut.captions_dir)

    # Check if dataset is already downloaded
    if (
        output_json_file.exists()
        and output_mask_dir.exists()
        and len(list(output_mask_dir.glob("*.png"))) > 0
        and output_captions_dir.exists()
        and len(list(output_captions_dir.glob("*.txt"))) > 0
    ):
        logger.info(f"COCONut dataset already exists at {data_cfg.data_dir}")
        return

    dataset_name = f"xdeng77/{COCONUT_SPLIT}"
    logger.info(f"Downloading {COCONUT_SPLIT} from {dataset_name}...")
    dataset = load_dataset(dataset_name)

    # create output folder
    output_path = Path(data_cfg.data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_json_file = Path(data_cfg.coconut.ann_file)

    # create output mask folder
    output_mask_dir = Path(data_cfg.coconut.masks_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # collect items from huggingface dataset: annotations (segments_info) and image infos
    output_annotations = []
    output_img_infos = []

    logger.info("Saving dataset to local path...")
    for item in tqdm(dataset["train"], desc=f"Processing {COCONUT_SPLIT}"):
        anno_info = item["segments_info"]
        img_id = anno_info["file_name"].split(".")[0]

        # save PIL image object to disk
        mask_path = output_mask_dir / f"{img_id}.png"
        item["mask"].save(mask_path)

        # save anno info to output_annotations
        output_annotations.append(anno_info)

        # save image info to output_img_infos
        output_img_infos.append(item["image_info"])

    output_json: dict[str, object] = {}
    output_json["images"] = output_img_infos
    output_json["annotations"] = output_annotations
    output_json["categories"] = [
        {"supercategory": "person", "isthing": 1, "id": 1, "name": "person"},
        {"supercategory": "vehicle", "isthing": 1, "id": 2, "name": "bicycle"},
        {"supercategory": "vehicle", "isthing": 1, "id": 3, "name": "car"},
        {"supercategory": "vehicle", "isthing": 1, "id": 4, "name": "motorcycle"},
        {"supercategory": "vehicle", "isthing": 1, "id": 5, "name": "airplane"},
        {"supercategory": "vehicle", "isthing": 1, "id": 6, "name": "bus"},
        {"supercategory": "vehicle", "isthing": 1, "id": 7, "name": "train"},
        {"supercategory": "vehicle", "isthing": 1, "id": 8, "name": "truck"},
        {"supercategory": "vehicle", "isthing": 1, "id": 9, "name": "boat"},
        {"supercategory": "outdoor", "isthing": 1, "id": 10, "name": "traffic light"},
        {"supercategory": "outdoor", "isthing": 1, "id": 11, "name": "fire hydrant"},
        {"supercategory": "outdoor", "isthing": 1, "id": 13, "name": "stop sign"},
        {"supercategory": "outdoor", "isthing": 1, "id": 14, "name": "parking meter"},
        {"supercategory": "outdoor", "isthing": 1, "id": 15, "name": "bench"},
        {"supercategory": "animal", "isthing": 1, "id": 16, "name": "bird"},
        {"supercategory": "animal", "isthing": 1, "id": 17, "name": "cat"},
        {"supercategory": "animal", "isthing": 1, "id": 18, "name": "dog"},
        {"supercategory": "animal", "isthing": 1, "id": 19, "name": "horse"},
        {"supercategory": "animal", "isthing": 1, "id": 20, "name": "sheep"},
        {"supercategory": "animal", "isthing": 1, "id": 21, "name": "cow"},
        {"supercategory": "animal", "isthing": 1, "id": 22, "name": "elephant"},
        {"supercategory": "animal", "isthing": 1, "id": 23, "name": "bear"},
        {"supercategory": "animal", "isthing": 1, "id": 24, "name": "zebra"},
        {"supercategory": "animal", "isthing": 1, "id": 25, "name": "giraffe"},
        {"supercategory": "accessory", "isthing": 1, "id": 27, "name": "backpack"},
        {"supercategory": "accessory", "isthing": 1, "id": 28, "name": "umbrella"},
        {"supercategory": "accessory", "isthing": 1, "id": 31, "name": "handbag"},
        {"supercategory": "accessory", "isthing": 1, "id": 32, "name": "tie"},
        {"supercategory": "accessory", "isthing": 1, "id": 33, "name": "suitcase"},
        {"supercategory": "sports", "isthing": 1, "id": 34, "name": "frisbee"},
        {"supercategory": "sports", "isthing": 1, "id": 35, "name": "skis"},
        {"supercategory": "sports", "isthing": 1, "id": 36, "name": "snowboard"},
        {"supercategory": "sports", "isthing": 1, "id": 37, "name": "sports ball"},
        {"supercategory": "sports", "isthing": 1, "id": 38, "name": "kite"},
        {"supercategory": "sports", "isthing": 1, "id": 39, "name": "baseball bat"},
        {"supercategory": "sports", "isthing": 1, "id": 40, "name": "baseball glove"},
        {"supercategory": "sports", "isthing": 1, "id": 41, "name": "skateboard"},
        {"supercategory": "sports", "isthing": 1, "id": 42, "name": "surfboard"},
        {"supercategory": "sports", "isthing": 1, "id": 43, "name": "tennis racket"},
        {"supercategory": "kitchen", "isthing": 1, "id": 44, "name": "bottle"},
        {"supercategory": "kitchen", "isthing": 1, "id": 46, "name": "wine glass"},
        {"supercategory": "kitchen", "isthing": 1, "id": 47, "name": "cup"},
        {"supercategory": "kitchen", "isthing": 1, "id": 48, "name": "fork"},
        {"supercategory": "kitchen", "isthing": 1, "id": 49, "name": "knife"},
        {"supercategory": "kitchen", "isthing": 1, "id": 50, "name": "spoon"},
        {"supercategory": "kitchen", "isthing": 1, "id": 51, "name": "bowl"},
        {"supercategory": "food", "isthing": 1, "id": 52, "name": "banana"},
        {"supercategory": "food", "isthing": 1, "id": 53, "name": "apple"},
        {"supercategory": "food", "isthing": 1, "id": 54, "name": "sandwich"},
        {"supercategory": "food", "isthing": 1, "id": 55, "name": "orange"},
        {"supercategory": "food", "isthing": 1, "id": 56, "name": "broccoli"},
        {"supercategory": "food", "isthing": 1, "id": 57, "name": "carrot"},
        {"supercategory": "food", "isthing": 1, "id": 58, "name": "hot dog"},
        {"supercategory": "food", "isthing": 1, "id": 59, "name": "pizza"},
        {"supercategory": "food", "isthing": 1, "id": 60, "name": "donut"},
        {"supercategory": "food", "isthing": 1, "id": 61, "name": "cake"},
        {"supercategory": "furniture", "isthing": 1, "id": 62, "name": "chair"},
        {"supercategory": "furniture", "isthing": 1, "id": 63, "name": "couch"},
        {"supercategory": "furniture", "isthing": 1, "id": 64, "name": "potted plant"},
        {"supercategory": "furniture", "isthing": 1, "id": 65, "name": "bed"},
        {"supercategory": "furniture", "isthing": 1, "id": 67, "name": "dining table"},
        {"supercategory": "furniture", "isthing": 1, "id": 70, "name": "toilet"},
        {"supercategory": "electronic", "isthing": 1, "id": 72, "name": "tv"},
        {"supercategory": "electronic", "isthing": 1, "id": 73, "name": "laptop"},
        {"supercategory": "electronic", "isthing": 1, "id": 74, "name": "mouse"},
        {"supercategory": "electronic", "isthing": 1, "id": 75, "name": "remote"},
        {"supercategory": "electronic", "isthing": 1, "id": 76, "name": "keyboard"},
        {"supercategory": "electronic", "isthing": 1, "id": 77, "name": "cell phone"},
        {"supercategory": "appliance", "isthing": 1, "id": 78, "name": "microwave"},
        {"supercategory": "appliance", "isthing": 1, "id": 79, "name": "oven"},
        {"supercategory": "appliance", "isthing": 1, "id": 80, "name": "toaster"},
        {"supercategory": "appliance", "isthing": 1, "id": 81, "name": "sink"},
        {"supercategory": "appliance", "isthing": 1, "id": 82, "name": "refrigerator"},
        {"supercategory": "indoor", "isthing": 1, "id": 84, "name": "book"},
        {"supercategory": "indoor", "isthing": 1, "id": 85, "name": "clock"},
        {"supercategory": "indoor", "isthing": 1, "id": 86, "name": "vase"},
        {"supercategory": "indoor", "isthing": 1, "id": 87, "name": "scissors"},
        {"supercategory": "indoor", "isthing": 1, "id": 88, "name": "teddy bear"},
        {"supercategory": "indoor", "isthing": 1, "id": 89, "name": "hair drier"},
        {"supercategory": "indoor", "isthing": 1, "id": 90, "name": "toothbrush"},
        {"supercategory": "textile", "isthing": 0, "id": 92, "name": "banner"},
        {"supercategory": "textile", "isthing": 0, "id": 93, "name": "blanket"},
        {"supercategory": "building", "isthing": 0, "id": 95, "name": "bridge"},
        {"supercategory": "raw-material", "isthing": 0, "id": 100, "name": "cardboard"},
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 107,
            "name": "counter",
        },
        {"supercategory": "textile", "isthing": 0, "id": 109, "name": "curtain"},
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 112,
            "name": "door-stuff",
        },
        {"supercategory": "floor", "isthing": 0, "id": 118, "name": "floor-wood"},
        {"supercategory": "plant", "isthing": 0, "id": 119, "name": "flower"},
        {"supercategory": "food-stuff", "isthing": 0, "id": 122, "name": "fruit"},
        {"supercategory": "ground", "isthing": 0, "id": 125, "name": "gravel"},
        {"supercategory": "building", "isthing": 0, "id": 128, "name": "house"},
        {"supercategory": "furniture-stuff", "isthing": 0, "id": 130, "name": "light"},
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 133,
            "name": "mirror-stuff",
        },
        {"supercategory": "structural", "isthing": 0, "id": 138, "name": "net"},
        {"supercategory": "textile", "isthing": 0, "id": 141, "name": "pillow"},
        {"supercategory": "ground", "isthing": 0, "id": 144, "name": "platform"},
        {"supercategory": "ground", "isthing": 0, "id": 145, "name": "playingfield"},
        {"supercategory": "ground", "isthing": 0, "id": 147, "name": "railroad"},
        {"supercategory": "water", "isthing": 0, "id": 148, "name": "river"},
        {"supercategory": "ground", "isthing": 0, "id": 149, "name": "road"},
        {"supercategory": "building", "isthing": 0, "id": 151, "name": "roof"},
        {"supercategory": "ground", "isthing": 0, "id": 154, "name": "sand"},
        {"supercategory": "water", "isthing": 0, "id": 155, "name": "sea"},
        {"supercategory": "furniture-stuff", "isthing": 0, "id": 156, "name": "shelf"},
        {"supercategory": "ground", "isthing": 0, "id": 159, "name": "snow"},
        {"supercategory": "furniture-stuff", "isthing": 0, "id": 161, "name": "stairs"},
        {"supercategory": "building", "isthing": 0, "id": 166, "name": "tent"},
        {"supercategory": "textile", "isthing": 0, "id": 168, "name": "towel"},
        {"supercategory": "wall", "isthing": 0, "id": 171, "name": "wall-brick"},
        {"supercategory": "wall", "isthing": 0, "id": 175, "name": "wall-stone"},
        {"supercategory": "wall", "isthing": 0, "id": 176, "name": "wall-tile"},
        {"supercategory": "wall", "isthing": 0, "id": 177, "name": "wall-wood"},
        {"supercategory": "water", "isthing": 0, "id": 178, "name": "water-other"},
        {"supercategory": "window", "isthing": 0, "id": 180, "name": "window-blind"},
        {"supercategory": "window", "isthing": 0, "id": 181, "name": "window-other"},
        {"supercategory": "plant", "isthing": 0, "id": 184, "name": "tree-merged"},
        {
            "supercategory": "structural",
            "isthing": 0,
            "id": 185,
            "name": "fence-merged",
        },
        {"supercategory": "ceiling", "isthing": 0, "id": 186, "name": "ceiling-merged"},
        {"supercategory": "sky", "isthing": 0, "id": 187, "name": "sky-other-merged"},
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 188,
            "name": "cabinet-merged",
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 189,
            "name": "table-merged",
        },
        {
            "supercategory": "floor",
            "isthing": 0,
            "id": 190,
            "name": "floor-other-merged",
        },
        {"supercategory": "ground", "isthing": 0, "id": 191, "name": "pavement-merged"},
        {"supercategory": "solid", "isthing": 0, "id": 192, "name": "mountain-merged"},
        {"supercategory": "plant", "isthing": 0, "id": 193, "name": "grass-merged"},
        {"supercategory": "ground", "isthing": 0, "id": 194, "name": "dirt-merged"},
        {
            "supercategory": "raw-material",
            "isthing": 0,
            "id": 195,
            "name": "paper-merged",
        },
        {
            "supercategory": "food-stuff",
            "isthing": 0,
            "id": 196,
            "name": "food-other-merged",
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 197,
            "name": "building-other-merged",
        },
        {"supercategory": "solid", "isthing": 0, "id": 198, "name": "rock-merged"},
        {"supercategory": "wall", "isthing": 0, "id": 199, "name": "wall-other-merged"},
        {"supercategory": "textile", "isthing": 0, "id": 200, "name": "rug-merged"},
    ]
    output_json["licenses"] = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
            "id": 3,
            "name": "Attribution-NonCommercial-NoDerivs License",
        },
        {
            "url": "http://creativecommons.org/licenses/by/2.0/",
            "id": 4,
            "name": "Attribution License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-sa/2.0/",
            "id": 5,
            "name": "Attribution-ShareAlike License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nd/2.0/",
            "id": 6,
            "name": "Attribution-NoDerivs License",
        },
        {
            "url": "http://flickr.com/commons/usage/",
            "id": 7,
            "name": "No known copyright restrictions",
        },
        {
            "url": "http://www.usa.gov/copyright.shtml",
            "id": 8,
            "name": "United States Government Work",
        },
    ]
    output_json["info"] = {
        "description": "COCO 2018 Panoptic Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2018,
        "contributor": "https://arxiv.org/abs/1801.00868",
        "date_created": "2018-06-01 00:00:00.0",
    }

    with open(output_json_file, "w") as f:
        json.dump(output_json, f, indent=4)

    logger.info(f"Downloaded {COCONUT_SPLIT} successfully!")
    logger.info("Now working on captions...")

    # Download and extract captions
    captions_url = data_cfg.coconut.captions_url
    captions_tar_path = Path(data_cfg.coconut.captions_tar_path)

    # Download captions tar file if not already present
    if not captions_tar_path.exists():
        logger.info(f"Downloading captions from {captions_url}...")
        urllib.request.urlretrieve(captions_url, captions_tar_path)
        logger.info("Captions tar downloaded successfully!")
    else:
        logger.info(f"Captions tar already exists at {captions_tar_path}")

    # Extract captions to the specified directory
    if (
        not output_captions_dir.exists()
        or len(list(output_captions_dir.glob("*.txt"))) == 0
    ):
        logger.info(f"Extracting captions to {output_captions_dir}...")

        # Extract to data_dir
        extract_root = Path(data_cfg.data_dir)

        with tarfile.open(captions_tar_path, "r") as tar:
            tar.extractall(path=extract_root)
        captions_tar_path.unlink()  # remove the tar file after extraction

        # Rename the extracted folder
        extracted_folder = extract_root / "coconut_pancap_50k"
        if extracted_folder.exists():
            if output_captions_dir.exists():
                shutil.rmtree(output_captions_dir)
            extracted_folder.rename(output_captions_dir)

        logger.info("Captions extracted successfully!")
    else:
        logger.info(f"Captions already extracted at {output_captions_dir}")

def download_png(data_cfg: DictConfig):
    """Download Panoptic Narrative Grounding (PNG) dataset.

    Downloads annotations and features, reusing COCO images and panoptic segmentation.
    For train features (not available for direct download), clones the PNG repo
    and runs feature extraction using the pretrained model.

    Args:
        data_cfg: Data configuration containing paths and URLs for PNG dataset.

    Target structure (in same data_dir as COCO):
        data_dir/
        |_ images/
        |  |_ train2017/        (from COCO)
        |  |_ val2017/          (from COCO)
        |_ features/
        |  |_ train2017/
        |  |  |_ mask_features/
        |  |  |_ sem_seg_features/
        |  |  |_ panoptic_seg_predictions/
        |  |_ val2017/
        |     |_ mask_features/
        |     |_ sem_seg_features/
        |     |_ panoptic_seg_predictions/
        |_ annotations/
           |_ png_coco_train2017.json
           |_ png_coco_val2017.json
           |_ panoptic_segmentation/  (from COCO)
           |  |_ train2017/
           |  |_ val2017/
           |_ panoptic_train2017.json (from COCO)
           |_ panoptic_val2017.json   (from COCO)
    """
    png_cfg = data_cfg.png
    data_dir = Path(data_cfg.data_dir)
    features_dir = Path(data_cfg.features_dir)

    features_dir.mkdir(parents=True, exist_ok=True)

    # Create feature subdirectories
    for split in ["train2017", "val2017"]:
        (features_dir / split).mkdir(parents=True, exist_ok=True)

    # Download PNG annotation files
    ann_train_path = Path(png_cfg.ann_file_train)
    ann_val_path = Path(png_cfg.ann_file_val)

    if not ann_train_path.exists():
        logger.info("Downloading PNG train annotations...")
        urllib.request.urlretrieve(png_cfg.ann_file_train_url, ann_train_path)
        logger.info(f"Downloaded PNG train annotations to {ann_train_path}")
    else:
        logger.info(f"PNG train annotations already exist at {ann_train_path}")

    if not ann_val_path.exists():
        logger.info("Downloading PNG val annotations...")
        urllib.request.urlretrieve(png_cfg.ann_file_val_url, ann_val_path)
        logger.info(f"Downloaded PNG val annotations to {ann_val_path}")
    else:
        logger.info(f"PNG val annotations already exist at {ann_val_path}")

    # Download val2017 features
    val_features_dir = features_dir / "val2017"
    val_features_complete = (
        (val_features_dir / "mask_features").exists()
        and len(list((val_features_dir / "mask_features").glob("*"))) > 0
        and (val_features_dir / "sem_seg_features").exists()
        and len(list((val_features_dir / "sem_seg_features").glob("*"))) > 0
        and (val_features_dir / "panoptic_seg_predictions").exists()
        and len(list((val_features_dir / "panoptic_seg_predictions").glob("*"))) > 0
    )

    if not val_features_complete:
        features_zip_path = data_dir / "val2017_features.zip"
        if not features_zip_path.exists():
            logger.info("Downloading val2017 features...")
            urllib.request.urlretrieve(png_cfg.features_val_url, features_zip_path)
            logger.info("Downloaded val2017 features")

        logger.info("Extracting val2017 features...")
        with zipfile.ZipFile(features_zip_path, "r") as zip_ref:
            zip_ref.extractall(features_dir)
        features_zip_path.unlink()
        logger.info("Extracted val2017 features")
    else:
        logger.info("Val2017 features already exist")

    # For train2017 features, we need to run feature extraction using the PNG repo
    train_features_dir = features_dir / "train2017"
    train_features_complete = (
        (train_features_dir / "mask_features").exists()
        and len(list((train_features_dir / "mask_features").glob("*"))) > 0
        and (train_features_dir / "sem_seg_features").exists()
        and len(list((train_features_dir / "sem_seg_features").glob("*"))) > 0
        and (train_features_dir / "panoptic_seg_predictions").exists()
        and len(list((train_features_dir / "panoptic_seg_predictions").glob("*"))) > 0
    )

    if not train_features_complete:
        logger.info("Remember to generate train2017 features using the PNG repo.")
    else:
        logger.info("Train2017 features already exist")

    logger.info("PNG dataset download and setup complete!")


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/",
    config_name="coconut",
)
def main(cfg: DictConfig) -> None:
    download_coco(cfg)
    download_png(cfg)


if __name__ == "__main__":
    main()