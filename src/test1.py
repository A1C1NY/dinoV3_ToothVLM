import torch
import json
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from train_detector_405YOLO import (
    Config,
    DinoPANNeck,
    build_model,
)
from ultralytics.nn.modules.head import v10Detect


def check_neck_and_head(device):
    print("\n[1/4] Checking Neck and v10Detect shapes")

    neck = DinoPANNeck().to(device)
    head = v10Detect(
        nc=Config.NUM_CLASSES,
        ch=(256, 512, 1024),
    ).to(device)
    head.stride = torch.tensor([8.0, 16.0, 32.0], device=device)
    head.bias_init()

    p3 = torch.randn(2, 256, 32, 32, device=device)
    p4 = torch.randn(2, 256, 16, 16, device=device)
    p5 = torch.randn(2, 256, 8, 8, device=device)

    features = neck([p3, p4, p5])
    expected_shapes = [
        (2, 256, 32, 32),
        (2, 512, 16, 16),
        (2, 1024, 8, 8),
    ]

    for index, (feature, expected) in enumerate(zip(features, expected_shapes)):
        print(f"feature[{index}]: {tuple(feature.shape)}")
        assert tuple(feature.shape) == expected

    head.train()
    raw = head(features)
    print("one2many boxes:", tuple(raw["one2many"]["boxes"].shape))
    print("one2many scores:", tuple(raw["one2many"]["scores"].shape))
    return True


def load_real_batch(device, count=2, image_size=(256, 256)):
    """Load a tiny real COCO batch and resize images/boxes for the smoke test."""
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root.parent / "405"
    image_dir = data_root / "image_filtered"
    annotation_path = project_root / "coco" / "All_Diseases" / "train.json"

    with annotation_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    annotations_by_image = {}
    for annotation in coco_data.get("annotations", []):
        if annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    selected_images = [
        image for image in coco_data.get("images", [])
        if image["id"] in annotations_by_image
    ][:count]
    if len(selected_images) < count:
        raise RuntimeError("Not enough annotated images found in the training JSON")

    tensors = []
    targets = []
    target_height, target_width = image_size

    for image_info in selected_images:
        image_path = image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((target_width, target_height))
        tensors.append(pil_to_tensor(image).float() / 255.0)

        scale_x = target_width / original_width
        scale_y = target_height / original_height
        boxes, labels = [], []
        for annotation in annotations_by_image[image_info["id"]]:
            x, y, width, height = annotation["bbox"]
            boxes.append([
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ])
            # COCO source categories are 1-based; the model converts them to 0-based.
            labels.append(annotation["category_id"])

        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        })

    return torch.stack(tensors).to(device), targets


def check_full_model(device):
    print("\n[2/4] Building the complete DINOv3 + YOLOv10 model")
    model = build_model().to(device)
    print("model built")
    print("num_classes:", Config.NUM_CLASSES)
    print("stride:", model.detect_head.stride.detach().cpu().tolist())

    images, targets = load_real_batch(device, count=2, image_size=(256, 256))
    print("real batch:", tuple(images.shape))
    print("target boxes:", [target["boxes"].shape[0] for target in targets])

    print("\n[3/4] Checking inference and postprocess")
    model.eval()
    with torch.no_grad():
        detections_without_filter = model(images, conf_threshold=0.0)
        detections_with_filter = model(
            images,
            conf_threshold=Config.CONF_THRESHOLD,
        )

    for index, detections in enumerate(detections_without_filter):
        print(
            f"image {index}: raw detections={tuple(detections.shape)}, "
            f"max_score={detections[:, 4].max().item() if len(detections) else 0.0:.6f}"
        )
        if len(detections):
            assert detections.shape[1] == 6
            assert torch.isfinite(detections).all()
            assert (detections[:, 2] >= detections[:, 0]).all()
            assert (detections[:, 3] >= detections[:, 1]).all()

    for index, detections in enumerate(detections_with_filter):
        print(f"image {index}: detections after threshold={tuple(detections.shape)}")

    print("\n[4/4] Checking target conversion, E2E loss, and backward")
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(images, targets)
    loss = output["loss"]
    print("loss:", float(loss.detach().cpu()))
    print("loss_items:", output["loss_items"])
    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()

    groups = {
        "dino_backbone": [],
        "neck": [],
        "detect_head": [],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("backbone.backbone"):
            groups["dino_backbone"].append(parameter.grad.detach().norm().item())
        elif name.startswith("neck"):
            groups["neck"].append(parameter.grad.detach().norm().item())
        elif name.startswith("detect_head"):
            groups["detect_head"].append(parameter.grad.detach().norm().item())

    for group, values in groups.items():
        print(
            f"{group}: tensors_with_grad={len(values)}, "
            f"max_grad={max(values) if values else 0.0:.6e}"
        )
        assert values, f"No gradients found for {group}"

    print("\n[5/5] Single-batch real-data overfit test")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        output = model(images, targets)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"overfit step={step}, loss={loss.item():.6f}, "
                f"box/cls/dfl={output['loss_items'].detach().cpu().tolist()}"
            )

    print("\nFinal predictions after real-data overfit:")
    model.eval()
    with torch.no_grad():
        final_detections = model(images, conf_threshold=0.0)

    for image_index, (prediction, target) in enumerate(zip(final_detections, targets)):
        print(f"image {image_index}: predictions={tuple(prediction.shape)}")
        if len(prediction) == 0:
            print("  no predictions")
            continue

        top_k = prediction[prediction[:, 4].argsort(descending=True)[:10]]
        print("  top predictions [x1,y1,x2,y2,score,class]:")
        print(top_k.detach().cpu())

        gt_boxes = target["boxes"]
        pred_boxes = prediction[:, :4]
        intersection_left_top = torch.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
        intersection_right_bottom = torch.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
        intersection_size = (intersection_right_bottom - intersection_left_top).clamp(min=0)
        intersection_area = intersection_size[..., 0] * intersection_size[..., 1]
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
        iou = intersection_area / (pred_area[:, None] + gt_area[None, :] - intersection_area + 1e-7)
        print(f"  max IoU with GT: {iou.max().item():.6f}")

    print("\nAll model checks passed.")


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    check_neck_and_head(device)
    check_full_model(device)
