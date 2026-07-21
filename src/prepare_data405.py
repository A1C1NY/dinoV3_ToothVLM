"""Build a single, lossless multi-disease COCO dataset from LabelMe files."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "562"
IMAGE_DIR = DATA_ROOT / "image_filtered"
LABEL_DIR = DATA_ROOT / "label_filtered"
OUTPUT_DIR = PROJECT_ROOT / "coco" / "All_Diseases"

# Keep category IDs contiguous because train_detector_405YOLO.py expects 1..N.
CATEGORIES = {
    "Caries": {"id": 1, "name": "caries"},
    "Calculus": {"id": 2, "name": "calculus"},
    "Mouth_Ulcer": {"id": 3, "name": "mouth_ulcer"},
    "Tooth_Discoloration": {"id": 4, "name": "tooth_discoloration"},
}

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
IMAGE_ID_OFFSETS = {"train": 0, "val": 50000}
ANNOTATION_ID_OFFSETS = {"train": 0, "val": 500000}


def normalize_label(label):
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


CATEGORY_ID_BY_LABEL = {
    normalize_label(category_name): category["id"]
    for category_name, category in CATEGORIES.items()
}


def resolve_image_path(label_path):
    """Resolve images only from the standardized LabelMe JSON filename."""
    stem = label_path.stem
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = IMAGE_DIR / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_records():
    """Load every LabelMe file once and group records by physical image name."""
    records_by_filename = defaultdict(list)
    missing_images = []

    for label_path in sorted(LABEL_DIR.glob("*.json")):
        image_path = resolve_image_path(label_path)
        if image_path is None:
            missing_images.append(label_path.name)
            continue

        with label_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        records_by_filename[image_path.name].append((label_path, image_path, data))

    return records_by_filename, missing_images


def split_records(records_by_filename):
    """Split by image filename so a duplicated LabelMe record cannot leak splits."""
    filenames = sorted(records_by_filename)
    random.Random(SPLIT_SEED).shuffle(filenames)
    split_index = int(len(filenames) * TRAIN_RATIO)
    return {
        "train": filenames[:split_index],
        "val": filenames[split_index:],
    }


def points_to_bbox(points):
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width, height = int(x2 - x1), int(y2 - y1)
    if width <= 0 or height <= 0:
        return None
    return [int(x1), int(y1), width, height]


def build_subset(subset, filenames, records_by_filename):
    images = []
    annotations = []
    exported_labels = Counter()
    skipped_labels = Counter()
    invalid_boxes = 0
    annotation_id = ANNOTATION_ID_OFFSETS[subset]

    for image_index, filename in enumerate(tqdm(filenames, desc=f"All diseases {subset}"), start=1):
        records = records_by_filename[filename]
        image_path = records[0][1]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        image_id = IMAGE_ID_OFFSETS[subset] + image_index
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        # Every shape from every LabelMe record for this physical image is kept.
        for label_path, _, data in records:
            for shape in data.get("shapes", []):
                normalized_label = normalize_label(shape.get("label", ""))
                category_id = CATEGORY_ID_BY_LABEL.get(normalized_label)
                if category_id is None:
                    skipped_labels[normalized_label or "<empty>"] += 1
                    continue

                bbox = points_to_bbox(shape.get("points", []))
                if bbox is None:
                    invalid_boxes += 1
                    continue

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                })
                annotation_id += 1
                exported_labels[normalized_label] += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": list(CATEGORIES.values()),
    }, exported_labels, skipped_labels, invalid_boxes


def validate_subset(subset, coco_data):
    image_ids = {image["id"] for image in coco_data["images"]}
    annotation_ids = [annotation["id"] for annotation in coco_data["annotations"]]
    bad_image_refs = sum(
        annotation["image_id"] not in image_ids
        for annotation in coco_data["annotations"]
    )
    duplicate_annotation_ids = len(annotation_ids) - len(set(annotation_ids))
    if bad_image_refs or duplicate_annotation_ids:
        raise ValueError(
            f"{subset} integrity failure: bad_image_refs={bad_image_refs}, "
            f"duplicate_annotation_ids={duplicate_annotation_ids}"
        )


def main():
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Expected image_dir={IMAGE_DIR}, label_dir={LABEL_DIR}")

    records_by_filename, missing_images = load_records()
    split_filenames = split_records(records_by_filename)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_exported_labels = Counter()
    all_skipped_labels = Counter()
    total_invalid_boxes = 0

    for subset in ("train", "val"):
        coco_data, exported_labels, skipped_labels, invalid_boxes = build_subset(
            subset,
            split_filenames[subset],
            records_by_filename,
        )
        validate_subset(subset, coco_data)
        output_path = OUTPUT_DIR / f"{subset}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(coco_data, file, ensure_ascii=False, indent=2)

        all_exported_labels.update(exported_labels)
        all_skipped_labels.update(skipped_labels)
        total_invalid_boxes += invalid_boxes
        print(
            f"All diseases [{subset}]: images={len(coco_data['images'])}, "
            f"annotations={len(coco_data['annotations'])}, output={output_path}"
        )

    print("Exported annotations by label:")
    for label, count in sorted(all_exported_labels.items()):
        print(f"  {label}: {count}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Invalid boxes: {total_invalid_boxes}")
    print(f"Unsupported labels: {sum(all_skipped_labels.values())}")
    for label, count in sorted(all_skipped_labels.items()):
        print(f"  {label}: {count}")

    if missing_images or total_invalid_boxes or all_skipped_labels:
        raise RuntimeError(
            "Conversion did not preserve every source annotation. "
            "Resolve the reported missing images, invalid boxes, or unsupported labels."
        )


if __name__ == "__main__":
    main()
