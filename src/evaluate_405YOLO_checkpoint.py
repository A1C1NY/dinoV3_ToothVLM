"""Evaluate a saved DINOv3-YOLOv10 checkpoint at a chosen confidence threshold."""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from train_detector_405YOLO import (
    Config,
    build_dataloaders,
    build_model,
    infer_num_classes,
)


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "res_checkpoints"
    / "multi_disease_562_expt"
    / "best_map.pth"
)

# RGB equivalents of the BGR colors in visualize_annotations.py.
CATEGORY_COLORS = {
    "caries": (0, 255, 0),
    "calculus": (0, 0, 255),
    "mouth_ulcer": (255, 165, 0),
    "periodontal_disease": (255, 0, 0),
    "tooth_discoloration": (0, 255, 255),
}
DEFAULT_COLOR = (128, 128, 128)
DEFAULT_METRIC_IOU_THRESHOLD = 0.5


def category_display(category_id, categories):
    name = categories.get(category_id, f"class_{category_id}")
    return name, CATEGORY_COLORS.get(name.lower(), DEFAULT_COLOR)


def calculate_detection_metrics(prediction, target, iou_threshold):
    """Calculate class-aware one-to-one TP/FP/FN counts for one image."""
    predictions = prediction.detach().cpu().tolist()
    ground_truth = [
        (box, int(label))
        for box, label in zip(target["boxes"].detach().cpu().tolist(), target["labels"].detach().cpu().tolist())
    ]
    matched_ground_truth = set()
    true_positives = 0

    # Predictions are already ordered by confidence from the model.
    for x1, y1, x2, y2, _, label in predictions:
        best_iou = 0.0
        best_index = None
        for index, (ground_truth_box, ground_truth_label) in enumerate(ground_truth):
            if index in matched_ground_truth or int(label) + 1 != ground_truth_label:
                continue
            gx1, gy1, gx2, gy2 = ground_truth_box
            intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(0.0, min(y2, gy2) - max(y1, gy1))
            prediction_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ground_truth_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
            union = prediction_area + ground_truth_area - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truth.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def draw_ground_truth(image, target, categories, fill_alpha=80):
    """Overlay original COCO boxes with disease-matched transparent fills."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    scale_x = target["scale_x"]
    scale_y = target["scale_y"]

    for box, category_id in zip(target["boxes"].tolist(), target["labels"].tolist()):
        x1, y1, x2, y2 = box
        original_box = [x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y]
        name, color = category_display(int(category_id), categories)
        overlay_draw.rectangle(
            original_box,
            fill=(*color, fill_alpha),
        )

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    label_draw = ImageDraw.Draw(image)
    for box, category_id in zip(target["boxes"].tolist(), target["labels"].tolist()):
        x1, y1, _, _ = box
        name, color = category_display(int(category_id), categories)
        label_draw.text((x1 / scale_x, max(0, y1 / scale_y - 12)), f"GT {name}", fill=color)
    return image


def audit_empty_samples(dataset, output_dir):
    """Create a visual audit of all images with no COCO annotations."""
    all_ids = list(dataset.images.keys())
    empty_ids = [image_id for image_id in all_ids if not dataset.annotations.get(image_id)]
    positive_ids = [image_id for image_id in all_ids if dataset.annotations.get(image_id)]
    positive_filenames = {dataset.images[image_id]["file_name"] for image_id in positive_ids}
    duplicate_empty_ids = [
        image_id for image_id in empty_ids
        if dataset.images[image_id]["file_name"] in positive_filenames
    ]

    print(
        f"Empty-image audit: total={len(all_ids)}, empty={len(empty_ids)}, "
        f"annotated={len(positive_ids)}"
    )
    print(
        "Empty images sharing file_name with annotated images: "
        f"{len(duplicate_empty_ids)}"
    )
    if duplicate_empty_ids:
        print("Duplicate file_name examples:")
        for image_id in duplicate_empty_ids[:10]:
            print(f"  image_id={image_id}, file_name={dataset.images[image_id]['file_name']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_items = []
    for image_id in empty_ids:
        info = dataset.images[image_id]
        audit_items.append({
            "image_id": image_id,
            "file_name": info["file_name"],
            "path": str(dataset.image_dir / info["file_name"]),
            "duplicates_annotated_file_name": info["file_name"] in positive_filenames,
        })
    manifest_path = output_dir / "empty_sample_audit.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(audit_items, file, ensure_ascii=False, indent=2)

    if empty_ids:
        columns, thumb_width, thumb_height, label_height = 5, 240, 180, 40
        rows = (len(empty_ids) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(audit_items):
            image_path = Path(item["path"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_width, thumb_height))
                x = (index % columns) * thumb_width + (thumb_width - image.width) // 2
                y = (index // columns) * (thumb_height + label_height)
                sheet.paste(image, (x, y))
            label = f"id={item['image_id']} {Path(item['file_name']).name}"
            draw.text((index % columns * thumb_width + 4, y + thumb_height + 4), label[:42], fill="black")
        sheet_path = output_dir / "empty_sample_audit.png"
        sheet.save(sheet_path)
        print(f"Empty-image contact sheet: {sheet_path}")

    print(f"Empty-image sample manifest: {manifest_path}")
    print("Review the contact sheet before treating these samples as true negatives.")


def visualize_predictions(model, val_loader, device, conf_threshold, sample_count=20, seed=42, output_dir=None):
    """Visualize predictions and original COCO annotations on validation samples."""
    if output_dir is None:
        output_dir = Path("inference_results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        int(category["id"]): category["name"]
        for category in val_loader.dataset.coco_data.get("categories", [])
    }

    all_image_ids = list(range(len(val_loader.dataset)))
    sample_indices = random.Random(seed).sample(all_image_ids, min(sample_count, len(all_image_ids)))

    model.eval()
    with torch.no_grad():
        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Visualizing predictions")):
            images, targets = val_loader.dataset[sample_idx]
            images_batch = images.unsqueeze(0).to(device, non_blocking=True)
            predictions = model(images_batch, conf_threshold=conf_threshold)
            prediction = predictions[0]

            scale_x = targets["scale_x"]
            scale_y = targets["scale_y"]
            image_id = targets["image_id"]

            image_path = val_loader.dataset.image_dir / val_loader.dataset.images[image_id]["file_name"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = draw_ground_truth(image, targets, categories)
                draw = ImageDraw.Draw(image)

                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    x1_scaled = x1 / scale_x
                    y1_scaled = y1 / scale_y
                    x2_scaled = x2 / scale_x
                    y2_scaled = y2 / scale_y

                    category_id = int(label) + 1
                    category_name, color = category_display(category_id, categories)
                    draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline=color, width=2)
                    label_text = f"Pred {category_name} {score:.2f}"
                    draw.text((x1_scaled, y1_scaled - 10), label_text, fill=color)

                result_path = output_dir / Path(image_path).name
                image.save(result_path)

    print(f"Inference visualizations saved to {output_dir}")


def evaluate(model, val_loader, device, conf_threshold, metric_iou_threshold=DEFAULT_METRIC_IOU_THRESHOLD):
    model.eval()
    coco_results = []
    total_predictions = 0
    score_values = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            predictions = model(images, conf_threshold=conf_threshold)

            for prediction, target in zip(predictions, targets):
                image_tp, image_fp, image_fn = calculate_detection_metrics(
                    prediction, target, metric_iou_threshold
                )
                true_positives += image_tp
                false_positives += image_fp
                false_negatives += image_fn
                total_predictions += len(prediction)
                if len(prediction):
                    score_values.extend(prediction[:, 4].detach().cpu().tolist())

                scale_x = target["scale_x"]
                scale_y = target["scale_y"]
                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    coco_results.append({
                        "image_id": target["image_id"],
                        "category_id": int(label) + 1,
                        "bbox": [
                            x1 / scale_x,
                            y1 / scale_y,
                            (x2 - x1) / scale_x,
                            (y2 - y1) / scale_y,
                        ],
                        "score": float(score),
                    })

    average_score = sum(score_values) / len(score_values) if score_values else 0.0
    max_score = max(score_values) if score_values else 0.0
    print(
        f"Validation predictions: total={total_predictions}, "
        f"average/image={total_predictions / max(1, len(val_loader.dataset)):.2f}, "
        f"max_score={max_score:.6f}, mean_score={average_score:.6f}"
    )

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    print(
        f"Detection metrics (IoU>={metric_iou_threshold:.2f}): "
        f"TP={true_positives}, FP={false_positives}, FN={false_negatives}, "
        f"precision={precision:.6f}, recall={recall:.6f}, F1={f1:.6f}"
    )

    if not coco_results:
        return {
            "map": 0.0, "map50": 0.0, "map75": 0.0,
            "precision": precision, "recall": recall, "f1": f1,
        }

    coco_gt = COCO(str(val_loader.dataset.annotation_file))
    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--metric-iou-threshold", type=float, default=DEFAULT_METRIC_IOU_THRESHOLD)
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf-threshold must be in [0, 1]")
    if not 0.0 <= args.metric_iou_threshold <= 1.0:
        parser.error("--metric-iou-threshold must be in [0, 1]")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(Config.DEVICE)
    project_root = Path(__file__).resolve().parent.parent
    num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    _, val_loader = build_dataloaders()
    audit_output_dir = args.audit_output_dir or args.checkpoint.parent
    audit_empty_samples(
        val_loader.dataset,
        output_dir=audit_output_dir,
    )
    model = build_model(num_classes=num_classes).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Confidence threshold: {args.conf_threshold}")

    inference_output_dir = audit_output_dir / "inference_results"
    visualize_predictions(model, val_loader, device, args.conf_threshold, sample_count=113, output_dir=inference_output_dir)

    metrics = evaluate(model, val_loader, device, args.conf_threshold, args.metric_iou_threshold)
    print(
        f"Metrics: mAP@[.5:.95]={metrics['map']:.6f}, "
        f"mAP@.5={metrics['map50']:.6f}, mAP@.75={metrics['map75']:.6f}"
    )


if __name__ == "__main__":
    main()
