"""Evaluate a saved DINOv3-YOLOv10 checkpoint at a chosen confidence threshold."""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from train_detector_405YOLO import (
    build_dataloaders,
    build_model,
    infer_num_classes,
)


# ────────────────────────────────────────────────────────────────────────────────
# Evaluation Configuration (independent from training config)
# ────────────────────────────────────────────────────────────────────────────────


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "res_checkpoints"
    / "multi_disease_Sonata_expt_v3_1"
    / "best_map.pth"
)
class EvalConfig:
    """独立的评估配置，不受训练脚本 Config 修改的影响"""
    # 路径配置
    REPO_DIR = "."
    IMAGE_DIR = "../Sonata/image"
    TRAIN_JSON = "coco/All_Diseases_Sonata/train.json"
    VAL_JSON = "coco/All_Diseases_Sonata/val.json"
    SINGLE_CAT_ID = None
    OUTPUT_DIR = "res_checkpoints/multi_disease_Sonata_expt_v3_1"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # 数据集配置
    # True = 验证时计入空标注图；False = 过滤空标注图。
    # 这个变量就是实际开关，直接改它即可，不要依赖命令行覆盖。
    INCLUDE_EMPTY_ANNOTATIONS = False
    DROP_EMPTY = not INCLUDE_EMPTY_ANNOTATIONS

    # 数据增强配置（评估时不使用，但配置结构需要完整）
    AUG_HFLIP = 0.5
    AUG_AFFINE = 0.7
    AUG_SCALE = 0.25
    AUG_TRANSLATE = 0.10
    AUG_ROTATE = 7.0
    AUG_MIN_BOX_SIZE = 4.0
    AUG_MIN_BOX_KEEP = 0.25
    PAD_VALUE = 114

    # Small-lesion augmentation（评估时不使用）
    MOSAIC_PROB = 0.35
    MOSAIC_CENTER_RANGE = (0.45, 0.55)
    COPY_PASTE_PROB = 0.30
    COPY_PASTE_MAX_BOX_AREA_RATIO = 0.02
    COPY_PASTE_MAX_OBJECTS = 2
    COPY_PASTE_CONTEXT_RATIO = 0.20
    COPY_PASTE_MAX_IOU = 0.10
    OVERSAMPLE_CATEGORY_ID = 3
    OVERSAMPLE_FACTOR = 1.75

    # 梯度裁剪（评估时不使用）
    CLIP_GRAD_NORM = 200.0

    # 训练超参数（评估时不使用，但模型构建需要）
    BATCH_SIZE = 8
    EPOCHS = 70
    LR = 0.001
    BACKBONE_LR = 0.0001
    WARMUP_EPOCHS = 5
    UNFREEZE_BLOCKS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Backbone 配置
    BACKBONE_OUT_INDICES = (5, 8, 11)

    # 继续训练配置（评估时不使用）
    RESUME_CHECKPOINT = None
    START_EPOCH = 1

    # 验证与评估参数
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5

    # 模型参数
    MIN_SIZE = 1200
    MAX_SIZE = 1200
    NUM_CLASSES = None
    CONF_THRESHOLD = 0.001

    # 类别自适应阈值
    VAL_CLASS_THRESHOLDS = {
        0: 0.30,  # Caries
        1: 0.30,  # Calculus
        2: 0.30,  # Mouth_Ulcer
        3: 0.30,  # Tooth_Discoloration
    }
    VAL_CONF_THRESHOLD_DEFAULT = 0.3

    # 类别权重
    CLASS_WEIGHTS = [1.2, 1.3, 2.5, 1.1]

    # DINOv3 归一化参数
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)

    # 数据加载配置
    IMG_SIZE = 768
    NUM_WORKERS = 4
    SEED = 42



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


def box_iou(box_a, box_b):
    """Return IoU for two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def update_confusion_matrix(confusion_matrix, prediction, target, category_to_index, iou_threshold):
    """Match boxes by IoU and accumulate GT-row/prediction-column counts."""
    predictions = prediction.detach().cpu().tolist()
    ground_truth = list(zip(
        target["boxes"].detach().cpu().tolist(),
        target["labels"].detach().cpu().tolist(),
    ))
    false_positive_index = len(category_to_index)  # FP row: predictions without GT match
    missed_index = len(category_to_index)  # Missed column: GT without prediction match
    matched_ground_truth = set()

    # A cross-class match is kept in the matrix to expose classification errors.
    for prediction_box in predictions:
        best_iou = 0.0
        best_index = None
        for index, (ground_truth_box, _) in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue
            iou = box_iou(prediction_box[:4], ground_truth_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        predicted_category_id = int(prediction_box[5]) + 1
        predicted_index = category_to_index.get(predicted_category_id)
        if predicted_index is None:
            continue
        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truth.add(best_index)
            ground_truth_category_id = int(ground_truth[best_index][1])
            ground_truth_index = category_to_index.get(ground_truth_category_id)
            if ground_truth_index is not None:
                confusion_matrix[ground_truth_index, predicted_index] += 1
        else:
            # False positive: prediction has no matching GT
            confusion_matrix[false_positive_index, predicted_index] += 1

    # Missed detections: GT boxes that were not matched
    for index, (_, ground_truth_category_id) in enumerate(ground_truth):
        if index not in matched_ground_truth:
            ground_truth_index = category_to_index.get(int(ground_truth_category_id))
            if ground_truth_index is not None:
                confusion_matrix[ground_truth_index, missed_index] += 1


def save_confusion_matrix(
    confusion_matrix,
    category_names,
    output_path,
    iou_threshold,
    normalize=False,
):
    """Save a raw-count or row-normalized validation detection confusion matrix."""
    # Updated labels: "Missed" for unmatched GT, "FP" for false positives
    x_labels = [*category_names, "background"]
    y_labels = [*category_names, "FP"]

    display_matrix = confusion_matrix
    colorbar_label = "Count"
    title_suffix = ""

    if normalize:
        row_totals = confusion_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            confusion_matrix,
            row_totals,
            out=np.zeros_like(confusion_matrix, dtype=float),
            where=row_totals != 0,
        )
        colorbar_label = "Proportion"
        title_suffix = " (Normalized)"

    figure, axis = plt.subplots(figsize=(max(8, len(x_labels) * 1.5), max(6, len(y_labels) * 1.25)))
    image = axis.imshow(display_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    axis.set(
        xticks=np.arange(len(x_labels)),
        yticks=np.arange(len(y_labels)),
        xticklabels=x_labels,
        yticklabels=y_labels,
        xlabel="Predicted class",
        ylabel="Ground-truth class",
        title=f"Detection Confusion Matrix (IoU >= {iou_threshold:.2f}){title_suffix}",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for row, column in np.ndindex(display_matrix.shape):
        if normalize:
            value = f"{display_matrix[row, column]:.1%}"
        else:
            value = str(int(confusion_matrix[row, column]))
        axis.text(
            column, row, value,
            ha="center", va="center",
            color="white" if display_matrix[row, column] > threshold else "black",
            fontsize=9,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Validation confusion matrix saved to {output_path}")


def draw_ground_truth(image, target, categories, fill_alpha=80):
    """Overlay original COCO boxes with disease-matched transparent fills."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    ratio = target["letterbox_ratio"]
    pad_x = target["pad_x"]
    pad_y = target["pad_y"]

    for box, category_id in zip(target["boxes"].tolist(), target["labels"].tolist()):
        x1, y1, x2, y2 = box
        original_box = [
            (x1 - pad_x) / ratio,
            (y1 - pad_y) / ratio,
            (x2 - pad_x) / ratio,
            (y2 - pad_y) / ratio,
        ]
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
        label_draw.text(
            ((x1 - pad_x) / ratio, max(0, (y1 - pad_y) / ratio - 12)),
            f"GT {name}",
            fill=color,
        )
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


def visualize_predictions(model, val_loader, device, conf_threshold, sample_count=20, seed=42, output_dir=None, use_class_thresholds=True):
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

            # Use class-specific thresholds if enabled
            if use_class_thresholds:
                predictions = model(images_batch, conf_threshold=0.001)
                # Apply class-specific thresholds（使用优化后的阈值）
                filtered_predictions = []
                for pred in predictions:
                    if len(pred) == 0:
                        filtered_predictions.append(pred)
                        continue
                    mask = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
                    for i, p in enumerate(pred):
                        cls_id = int(p[5].item())
                        cls_thresh = EvalConfig.VAL_CLASS_THRESHOLDS.get(
                            cls_id, EvalConfig.VAL_CONF_THRESHOLD_DEFAULT
                        )
                        if p[4].item() >= cls_thresh:
                            mask[i] = True
                    filtered_predictions.append(pred[mask])
                predictions = filtered_predictions
            else:
                predictions = model(images_batch, conf_threshold=conf_threshold)

            prediction = predictions[0]

            ratio = targets["letterbox_ratio"]
            pad_x = targets["pad_x"]
            pad_y = targets["pad_y"]
            image_id = targets["image_id"]

            image_path = val_loader.dataset.image_dir / val_loader.dataset.images[image_id]["file_name"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = draw_ground_truth(image, targets, categories)
                draw = ImageDraw.Draw(image)

                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    x1_scaled = (x1 - pad_x) / ratio
                    y1_scaled = (y1 - pad_y) / ratio
                    x2_scaled = (x2 - pad_x) / ratio
                    y2_scaled = (y2 - pad_y) / ratio

                    category_id = int(label) + 1
                    category_name, color = category_display(category_id, categories)
                    draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline=color, width=2)
                    label_text = f"Pred {category_name} {score:.2f}"
                    draw.text((x1_scaled, y1_scaled - 10), label_text, fill=color)

                result_path = output_dir / Path(image_path).name
                image.save(result_path)

    print(f"Inference visualizations saved to {output_dir}")


def evaluate(
    model,
    val_loader,
    device,
    conf_threshold,
    metric_iou_threshold=DEFAULT_METRIC_IOU_THRESHOLD,
    confusion_matrix_output=None,
    use_class_thresholds=True,
):
    model.eval()
    categories = sorted(
        val_loader.dataset.coco_data.get("categories", []),
        key=lambda category: int(category["id"]),
    )
    category_to_index = {
        int(category["id"]): index for index, category in enumerate(categories)
    }
    confusion_matrix = np.zeros((len(categories) + 1, len(categories) + 1), dtype=np.int64)
    coco_results = []
    total_predictions = 0
    score_values = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)

            # Use class-specific thresholds if enabled
            if use_class_thresholds:
                predictions = model(images, conf_threshold=0.001)  # 先用低阈值获取所有预测
                # 后处理：应用类别自适应阈值（使用优化后的阈值）
                filtered_predictions = []
                for pred in predictions:
                    if len(pred) == 0:
                        filtered_predictions.append(pred)
                        continue
                    mask = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
                    for i, p in enumerate(pred):
                        cls_id = int(p[5].item())
                        cls_thresh = EvalConfig.VAL_CLASS_THRESHOLDS.get(
                            cls_id, EvalConfig.VAL_CONF_THRESHOLD_DEFAULT
                        )
                        if p[4].item() >= cls_thresh:
                            mask[i] = True
                    filtered_predictions.append(pred[mask])
                predictions = filtered_predictions
            else:
                predictions = model(images, conf_threshold=conf_threshold)

            for prediction, target in zip(predictions, targets):
                update_confusion_matrix(
                    confusion_matrix,
                    prediction,
                    target,
                    category_to_index,
                    metric_iou_threshold,
                )
                image_tp, image_fp, image_fn = calculate_detection_metrics(
                    prediction, target, metric_iou_threshold
                )
                true_positives += image_tp
                false_positives += image_fp
                false_negatives += image_fn
                total_predictions += len(prediction)
                if len(prediction):
                    score_values.extend(prediction[:, 4].detach().cpu().tolist())

                ratio = target["letterbox_ratio"]
                pad_x = target["pad_x"]
                pad_y = target["pad_y"]
                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    coco_results.append({
                        "image_id": target["image_id"],
                        "category_id": int(label) + 1,
                        "bbox": [
                            (x1 - pad_x) / ratio,
                            (y1 - pad_y) / ratio,
                            (x2 - x1) / ratio,
                            (y2 - y1) / ratio,
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

    if confusion_matrix_output is not None:
        save_confusion_matrix(
            confusion_matrix,
            [category["name"] for category in categories],
            Path(confusion_matrix_output),
            metric_iou_threshold,
        )
        normalized_output = Path(confusion_matrix_output).with_name(
            f"{Path(confusion_matrix_output).stem}_normalized.png"
        )
        save_confusion_matrix(
            confusion_matrix,
            [category["name"] for category in categories],
            normalized_output,
            metric_iou_threshold,
            normalize=True,
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
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=EvalConfig.VAL_CONF_THRESHOLD_DEFAULT,
        help="Single confidence threshold (only used when --no-class-thresholds is specified)",
    )
    parser.add_argument("--metric-iou-threshold", type=float, default=DEFAULT_METRIC_IOU_THRESHOLD)
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=None,
        help="Output PNG path for the validation detection confusion matrix.",
    )
    parser.add_argument(
        "--no-class-thresholds",
        action="store_true",
        default=False,
        help="Disable class-specific thresholds and use a single --conf-threshold for all classes.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf-threshold must be in [0, 1]")
    if not 0.0 <= args.metric_iou_threshold <= 1.0:
        parser.error("--metric-iou-threshold must be in [0, 1]")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 默认使用类别自适应阈值，除非用户指定 --no-class-thresholds
    use_class_thresholds = not args.no_class_thresholds
    EvalConfig.DROP_EMPTY = not EvalConfig.INCLUDE_EMPTY_ANNOTATIONS

    device = torch.device(EvalConfig.DEVICE)
    project_root = Path(__file__).resolve().parent.parent
    num_classes = infer_num_classes(project_root / EvalConfig.TRAIN_JSON)
    _, val_loader = build_dataloaders(config=EvalConfig)
    audit_output_dir = args.audit_output_dir or args.checkpoint.parent
    audit_empty_samples(
        val_loader.dataset,
        output_dir=audit_output_dir,
    )
    model = build_model(num_classes=num_classes, config=EvalConfig).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # 兼容旧版 checkpoint（不含 class_weights buffer）：
    # 用 strict=False 加载，但对非 class_weights 的 missing/unexpected key 仍报错。
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    ignored_missing = {"class_weights"}
    real_missing = [k for k in load_result.missing_keys if k not in ignored_missing]
    if real_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch.\n"
            f"  Missing keys   : {real_missing}\n"
            f"  Unexpected keys: {load_result.unexpected_keys}"
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(
        "Empty-annotation handling: "
        + ("included in validation" if EvalConfig.INCLUDE_EMPTY_ANNOTATIONS else "excluded from validation")
    )
    if use_class_thresholds:
        print(f"Using class-specific thresholds: {EvalConfig.VAL_CLASS_THRESHOLDS}")
    else:
        print(f"Using single confidence threshold: {args.conf_threshold}")

    inference_output_dir = audit_output_dir / "inference_results"
    visualize_predictions(
        model, val_loader, device, args.conf_threshold,
        sample_count=113, output_dir=inference_output_dir,
        use_class_thresholds=use_class_thresholds
    )

    confusion_matrix_output = args.confusion_matrix_output or audit_output_dir / "validation_confusion_matrix.png"
    metrics = evaluate(
        model,
        val_loader,
        device,
        args.conf_threshold,
        args.metric_iou_threshold,
        confusion_matrix_output,
        use_class_thresholds=use_class_thresholds,
    )
    print(
        f"Metrics: mAP@[.5:.95]={metrics['map']:.6f}, "
        f"mAP@.5={metrics['map50']:.6f}, mAP@.75={metrics['map75']:.6f}"
    )


if __name__ == "__main__":
    main()
