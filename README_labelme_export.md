# LabelMe Export Script Usage

## Overview
`labelme_405YOLO_checkpoint.py` loads a trained checkpoint and exports predictions in LabelMe JSON format for a given image folder.

## Features
- Loads images from a specified folder
- Runs model inference with configurable confidence threshold
- Exports results as LabelMe JSON files (one per image)
- Copies original images to the output folder
- Disease IDs match those in `prepare_data405.py`:
  - ID 1: Caries
  - ID 2: Calculus  
  - ID 3: Mouth_Ulcer
  - ID 4: Tooth_Discoloration

## Usage

```bash
python src/labelme_405YOLO_checkpoint.py \
    --checkpoint path/to/checkpoint.pth \
    --image-folder path/to/images \
    --conf-threshold 0.30 \
    --output-name labelme_result
```

## Arguments

- `--checkpoint` (required): Path to the model checkpoint (.pth file)
- `--image-folder` (required): Folder containing images to process
- `--conf-threshold` (optional): Confidence threshold for predictions (default: 0.30)
- `--output-name` (optional): Name of output folder created under checkpoint directory (default: "labelme_result")

## Output Structure

The script creates a folder named `labelme_result` (or custom name) under the checkpoint directory containing:
- Original images (copied from input folder)
- JSON files with same name as images (e.g., `image001.json` for `image001.jpg`)

Each JSON file contains:
- `shapes`: List of detected bounding boxes with labels and confidence scores
- `imagePath`: Original image filename
- `imageWidth`, `imageHeight`: Original image dimensions

## Example

```bash
python src/labelme_405YOLO_checkpoint.py \
    --checkpoint res_checkpoints/multi_disease_562_expt_vit_base_dental_backbone/best_map.pth \
    --image-folder /path/to/test/images \
    --conf-threshold 0.35
```

This will create:
```
res_checkpoints/multi_disease_562_expt_vit_base_dental_backbone/labelme_result/
├── image001.jpg
├── image001.json
├── image002.jpg
├── image002.json
└── ...
```

## Notes

- Images and JSON files are placed in the same directory for direct use with LabelMe
- The script preserves original image names
- Confidence scores are included in each shape's metadata
- Only 4 disease categories from prepare_data405.py are supported
