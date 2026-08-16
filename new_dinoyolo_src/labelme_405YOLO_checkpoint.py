"""Export model predictions to LabelMe format for a given image folder."""

import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from train_detector_405YOLO import Config, build_model, infer_num_classes
from data.model_data import letterbox_image


# ========== Configuration ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Checkpoint path
CHECKPOINT_PATH = (
    PROJECT_ROOT / "res_checkpoints"
    / "multi_disease_767_expt_v3_1_highsize"
    / "best_map.pth"
)

# Input image folder
IMAGE_FOLDER = PROJECT_ROOT.parent / "10" 

# Output folder name (will be created under checkpoint directory)
OUTPUT_FOLDER_NAME = f"{IMAGE_FOLDER}_labelme_result"

# Image size (must match training)
IMG_SIZE = Config.IMG_SIZE  # 640

# ========== Category Definitions ==========
# Categories from prepare_data405.py - only 4 classes
CATEGORIES = {
    "Caries": {"id": 1, "name": "caries"},
    "Calculus": {"id": 2, "name": "calculus"},
    "Mouth_Ulcer": {"id": 3, "name": "mouth_ulcer"},
    "Tooth_Discoloration": {"id": 4, "name": "tooth_discoloration"},
}

# Per-disease confidence thresholds for this LabelMe export script only.
# Keep this mapping local: it is intentionally independent from the evaluation
# script's EvalConfig.VAL_CLASS_THRESHOLDS.
DISEASE_CONF_THRESHOLDS = {
    "Caries": 0.30,
    "Calculus": 0.30,
    "Mouth_Ulcer": 0.30,
    "Tooth_Discoloration": 0.30,
}

MODEL_CLASS_TO_DISEASE = {
    0: "Caries",
    1: "Calculus",
    2: "Mouth_Ulcer",
    3: "Tooth_Discoloration",
}

# Map model output class indices (0-indexed) to category IDs used in prepare_data405.py
# This assumes the model was trained with the 4-class dataset
MODEL_CLASS_TO_CATEGORY_ID = {
    0: 1,  # caries
    1: 2,  # calculus
    2: 3,  # mouth_ulcer
    3: 4,  # tooth_discoloration
}

# Reverse lookup from category ID to category name
CATEGORY_ID_TO_NAME = {
    category["id"]: category["name"]
    for category in CATEGORIES.values()
}

# Map category name back to original LabelMe format (capitalized with underscores)
CATEGORY_NAME_TO_LABELME = {
    "caries": "Caries",
    "calculus": "Calculus",
    "mouth_ulcer": "Mouth_Ulcer",
    "tooth_discoloration": "Tooth_Discoloration",
}


def create_labelme_json(image_path, predictions, image_width, image_height, ratio, pad_x, pad_y):
    """Create a LabelMe JSON annotation from model predictions."""
    shapes = []

    for x1, y1, x2, y2, score, label in predictions.detach().cpu().tolist():
        # Convert model class index to category ID
        model_class = int(label)
        category_id = MODEL_CLASS_TO_CATEGORY_ID.get(model_class)
        if category_id is None:
            continue

        # Get category name and convert to LabelMe format
        category_name_normalized = CATEGORY_ID_TO_NAME.get(category_id, f"class_{category_id}")
        category_name = CATEGORY_NAME_TO_LABELME.get(category_name_normalized, category_name_normalized)

        # Undo letterbox: remove padding offset, then rescale to original size
        x1_original = (x1 - pad_x) / ratio
        y1_original = (y1 - pad_y) / ratio
        x2_original = (x2 - pad_x) / ratio
        y2_original = (y2 - pad_y) / ratio

        # Create LabelMe shape (rectangle format)
        shape = {
            "label": category_name,
            "points": [
                [float(x1_original), float(y1_original)],
                [float(x2_original), float(y2_original)]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "score": float(score)
        }
        shapes.append(shape)

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    return labelme_data


def process_image(model, image_path, device, disease_conf_thresholds, img_size):
    """Run inference on a single image and return predictions with letterbox info.

    Uses the same preprocessing as training:
    - Letterbox resize to (img_size, img_size), preserving aspect ratio
    - pil_to_tensor followed by division by 255
    - No normalization with mean/std (the model normalizes internally)
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size

        # Letterbox (same as training) — preserves lesion shape
        image_resized, ratio, pad_x, pad_y = letterbox_image(
            image, img_size, img_size, pad_value=Config.PAD_VALUE
        )

        # Convert to tensor and scale to [0, 1] (same as training)
        image_tensor = pil_to_tensor(image_resized).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference
        # Use the lowest class threshold for model-side filtering, then apply
        # each disease's threshold below so no class is discarded prematurely.
        model_conf_threshold = min(disease_conf_thresholds.values())
        with torch.no_grad():
            predictions = model(image_tensor, conf_threshold=model_conf_threshold)

        prediction = predictions[0]
        if len(prediction):
            keep = torch.tensor(
                [
                    float(score) >= disease_conf_thresholds.get(
                        MODEL_CLASS_TO_DISEASE.get(int(label)),
                        model_conf_threshold,
                    )
                    for score, label in zip(prediction[:, 4], prediction[:, 5])
                ],
                dtype=torch.bool,
                device=prediction.device,
            )
            prediction = prediction[keep]

        return prediction, original_width, original_height, ratio, pad_x, pad_y


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not IMAGE_FOLDER.is_dir():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")
    invalid_thresholds = {
        disease: threshold
        for disease, threshold in DISEASE_CONF_THRESHOLDS.items()
        if not 0.0 <= threshold <= 1.0
    }
    if invalid_thresholds:
        raise ValueError(
            f"Disease confidence thresholds must be in [0, 1]: {invalid_thresholds}"
        )

    # Setup output directory
    output_dir = CHECKPOINT_PATH.parent / OUTPUT_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device(Config.DEVICE)
    num_classes = infer_num_classes(PROJECT_ROOT / Config.TRAIN_JSON)
    # The current model builder requires the training configuration explicitly.
    # This keeps architecture, preprocessing, and DINOv3 settings aligned with
    # the checkpoint produced by train_detector_405YOLO.py.
    model = build_model(num_classes=num_classes, config=Config).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # Older checkpoints may not contain the class_weights buffer. Any other
    # mismatch indicates that the checkpoint and model architecture disagree.
    real_missing = [key for key in load_result.missing_keys if key != "class_weights"]
    if real_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch.\n"
            f"  Missing keys   : {real_missing}\n"
            f"  Unexpected keys: {load_result.unexpected_keys}"
        )
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Disease confidence thresholds: {DISEASE_CONF_THRESHOLDS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Processing images from: {IMAGE_FOLDER}")
    print(f"Output directory: {output_dir}")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [
        f for f in sorted(IMAGE_FOLDER.iterdir())
        if f.suffix in image_extensions
    ]

    if not image_files:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    print(f"Found {len(image_files)} images")

    # Process each image
    total_predictions = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run inference
            predictions, img_width, img_height, ratio, pad_x, pad_y = process_image(
                model, image_path, device, DISEASE_CONF_THRESHOLDS, IMG_SIZE
            )

            # Create LabelMe JSON
            labelme_data = create_labelme_json(
                image_path, predictions, img_width, img_height, ratio, pad_x, pad_y
            )

            # Save JSON file
            json_path = output_dir / f"{image_path.stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)

            # Copy original image to output directory
            output_image_path = output_dir / image_path.name
            shutil.copy2(image_path, output_image_path)

            total_predictions += len(labelme_data["shapes"])

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            continue

    print(f"\nProcessing complete. Results saved to: {output_dir}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average predictions per image: {total_predictions / len(image_files):.2f}")


if __name__ == "__main__":
    main()
