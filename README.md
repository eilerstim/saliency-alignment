# Saliency Alignment

This project finetunes llava using a custom saliency loss.

## Dataset Structure

After running the download scripts, your data directory will have the following structure:

```
$SCRATCH/finetune/data/coco/
│
├── train2017.zip                          # Downloaded COCO train images (temp)
├── val2017.zip                            # Downloaded COCO val images (temp)
├── annotations.zip                        # Downloaded COCO annotations (temp)
├── coconut_pancap_50k.tar                # Downloaded COCONut captions archive (temp)
│
├── train2017/                             # Extracted from train2017.zip
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (118,287 training images)
│
├── val2017/                               # Extracted from val2017.zip
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ... (5,000 validation images)
│
├── annotations/                           # Extracted from annotations.zip
│   ├── instances_train2017.json          # COCO instance segmentation (train)
│   ├── instances_val2017.json            # COCO instance segmentation (val)
│   ├── captions_train2017.json           # COCO captions (train)
│   ├── captions_val2017.json             # COCO captions (val)
│   ├── panoptic_train2017.json           # COCONut panoptic annotations
│   └── ... (other COCO annotation files)
│
├── panoptic_train2017_masks/             # COCONut masks from HuggingFace
│   ├── 000000000009.png                  # Panoptic segmentation masks
│   ├── 000000000025.png
│   └── ... (subset of COCO images with panoptic annotations)
│
└── panoptic_train2017_captions/          # Extracted from coconut_pancap_50k.tar
    ├── <extracted caption files>
    └── ... (caption data)
```
