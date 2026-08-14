import json
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config.config import Config
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

def infer_num_classes(annotation_file):
    """使用 COCO JSON 注释文件推断前景类别数量。

    Args:
        annotation_file: COCO JSON 注释文件路径。

    Returns:
        int: 前景类别数量。

    Raises:
        ValueError: 如果 COCO 类别 ID 不连续或未找到任何注释类别 ID。
    """
    with Path(annotation_file).open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    used_ids = sorted({int(item["category_id"]) for item in coco_data.get("annotations", [])})
    expected_ids = list(range(1, len(used_ids) + 1))
    if not used_ids:
        raise ValueError(f"No annotated category IDs found in {annotation_file}")
    if used_ids != expected_ids:
        raise ValueError(f"COCO category IDs must be contiguous 1..N, got {used_ids}")
    return len(used_ids)




def letterbox_params(original_width, original_height, target_width, target_height):
    """计算保持纵横比的缩放比例与居中填充偏移。

    返回 (ratio, pad_x, pad_y)。原图坐标 -> 网络输入坐标的映射为
    ``x_in = x_orig * ratio + pad_x``，反变换为 ``x_orig = (x_in - pad_x) / ratio``。
    """
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = round(original_width * ratio)
    new_height = round(original_height * ratio)
    pad_x = (target_width - new_width) / 2.0
    pad_y = (target_height - new_height) / 2.0
    return ratio, pad_x, pad_y


def letterbox_image(image, target_width, target_height, pad_value=114):
    """把 PIL 图缩放到能放进目标尺寸的最大比例，再居中填充成目标尺寸。"""
    original_width, original_height = image.size
    ratio, pad_x, pad_y = letterbox_params(
        original_width, original_height, target_width, target_height
    )
    new_width = round(original_width * ratio)
    new_height = round(original_height * ratio)
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (pad_value,) * 3)
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, ratio, pad_x, pad_y


def random_affine(
    image_tensor,
    boxes,
    labels,
    degrees=7.0,
    translate=0.10,
    scale=0.25,
    pad_value=114,
    min_box_size=4.0,
    min_box_keep=0.25,
):
    """对已 letterbox 的图做随机缩放/平移/旋转，并同步变换边界框。

    只做几何变换，不触碰像素强度。填充区域用 ``pad_value`` 补齐，与 letterbox 一致。
    框由四角点变换后重新取轴对齐外接框，因此旋转角应保持较小，否则框会明显膨胀。
    """
    _, height, width = image_tensor.shape
    center_x, center_y = width / 2.0, height / 2.0

    angle = math.radians(random.uniform(-degrees, degrees))
    ratio = random.uniform(1 - scale, 1 + scale)
    tx = random.uniform(-translate, translate) * width
    ty = random.uniform(-translate, translate) * height

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # 正向矩阵：绕图心旋转+缩放，再平移。用于变换框坐标。
    a = ratio * cos_a
    b = -ratio * sin_a
    c = ratio * sin_a
    d = ratio * cos_a
    e = center_x - a * center_x - b * center_y + tx
    f = center_y - c * center_x - d * center_y + ty

    # grid_sample 需要 output->input 的逆映射。
    determinant = a * d - b * c
    if abs(determinant) < 1e-8:
        return image_tensor, boxes, labels
    inv_a = d / determinant
    inv_b = -b / determinant
    inv_c = -c / determinant
    inv_d = a / determinant
    inv_e = -(inv_a * e + inv_b * f)
    inv_f = -(inv_c * e + inv_d * f)

    device = image_tensor.device
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    src_x = inv_a * xs + inv_b * ys + inv_e
    src_y = inv_c * xs + inv_d * ys + inv_f
    # 归一化到 [-1, 1]（align_corners=False 的像素中心约定）
    grid_x = (src_x + 0.5) / width * 2 - 1
    grid_y = (src_y + 0.5) / height * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    # grid_sample 的 zeros padding 会补黑；先减去灰度基线再加回，得到灰色填充。
    fill = pad_value / 255.0
    shifted = (image_tensor - fill).unsqueeze(0)
    warped = F.grid_sample(
        shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped = (warped.squeeze(0) + fill).clamp_(0.0, 1.0)

    if not len(boxes):
        return warped, boxes, labels

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners_x = torch.stack([x1, x2, x1, x2], dim=1)
    corners_y = torch.stack([y1, y1, y2, y2], dim=1)
    new_x = a * corners_x + b * corners_y + e
    new_y = c * corners_x + d * corners_y + f

    original_areas = ((x2 - x1) * (y2 - y1)).clamp(min=1e-6)
    new_boxes = torch.stack([
        new_x.min(dim=1).values.clamp(0, width),
        new_y.min(dim=1).values.clamp(0, height),
        new_x.max(dim=1).values.clamp(0, width),
        new_y.max(dim=1).values.clamp(0, height),
    ], dim=1)

    widths = new_boxes[:, 2] - new_boxes[:, 0]
    heights = new_boxes[:, 3] - new_boxes[:, 1]
    # 用缩放后的期望面积作分母，避免把"因放大而变大"的框误判为需保留/丢弃。
    keep = (
        (widths > min_box_size)
        & (heights > min_box_size)
        & ((widths * heights) / (original_areas * ratio * ratio) > min_box_keep)
    )
    return warped, new_boxes[keep], labels[keep]


class CocoYOLODataset(Dataset):
    """COCO-style dataset for the custom model.

    Args:
        annotation_file: COCO JSON 的路径，包含图像和注释信息。
        image_dir: 图像文件所在的目录。
        image_size: 输入图像的目标大小，格式为 (height, width)。
        drop_empty: 如果为 True，将丢弃没有标注的图像。
        augment: 如果为 True，将应用几何数据增强（翻转 + 随机仿射）。

    图像用 letterbox 缩放：保持纵横比，不足处居中填充，避免病灶形状被拉伸。
    target 中提供 ``letterbox_ratio`` / ``pad_x`` / ``pad_y`` 供坐标反变换使用。
    """

    def __init__(self, annotation_file, image_dir, image_size, drop_empty=False, augment=False, config=None):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.image_height, self.image_width = image_size
        self.augment = augment
        # config.config 里的 Config 只做“声明”；实际取值由训练/评估脚本传入。
        self.config = config or Config
        with self.annotation_file.open("r", encoding="utf-8") as file:
            self.coco_data = json.load(file)

        self.images = {item["id"]: item for item in self.coco_data.get("images", [])}
        self.annotations = {}
        for annotation in self.coco_data.get("annotations", []):
            x, y, width, height = annotation["bbox"]
            if width > 0 and height > 0:
                self.annotations.setdefault(annotation["image_id"], []).append(annotation)

        self.ids = list(self.images)
        if drop_empty:
            self.ids = [image_id for image_id in self.ids if self.annotations.get(image_id)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        image, ratio, pad_x, pad_y = letterbox_image(
            image, self.image_width, self.image_height, pad_value=self.config.PAD_VALUE
        )
        image_tensor = pil_to_tensor(image).float() / 255.0

        boxes, labels = [], []
        for annotation in self.annotations.get(image_id, []):
            x, y, width, height = annotation["bbox"]
            boxes.append([
                x * ratio + pad_x,
                y * ratio + pad_y,
                (x + width) * ratio + pad_x,
                (y + height) * ratio + pad_y,
            ])
            labels.append(annotation["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.augment:
            if random.random() < self.config.AUG_HFLIP:
                image_tensor = image_tensor.flip(-1)
                if len(boxes):
                    left = boxes[:, 0].clone()
                    boxes[:, 0] = self.image_width - boxes[:, 2]
                    boxes[:, 2] = self.image_width - left
            if random.random() < self.config.AUG_AFFINE:
                warped, new_boxes, new_labels = random_affine(
                    image_tensor,
                    boxes,
                    labels,
                    degrees=self.config.AUG_ROTATE,
                    translate=self.config.AUG_TRANSLATE,
                    scale=self.config.AUG_SCALE,
                    pad_value=self.config.PAD_VALUE,
                    min_box_size=self.config.AUG_MIN_BOX_SIZE,
                    min_box_keep=self.config.AUG_MIN_BOX_KEEP,
                )
                # 若增强把全部框都裁掉了，退回未增强版本，避免产出空标注样本。
                if len(new_boxes) or not len(boxes):
                    image_tensor, boxes, labels = warped, new_boxes, new_labels

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_width": original_width,
            "original_height": original_height,
            "letterbox_ratio": ratio,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }
        return image_tensor, target


def detection_collate(batch):
    """
    Stack fixed-size images while preserving the per-image target dictionaries.
    """
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def build_dataloaders(config=None):
    """Build train/validation loaders from the generated All_Diseases COCO files.

    Args:
        config: 填好实际取值的 Config 类（来自训练/评估脚本）。
                config.config 里的 Config 只做“声明”，直接使用会因缺少取值而报错。
    """
    config = config or Config
    # 相对路径以“仓库根目录”（dinoV3_ToothVLM）为基准：
    #   - coco/All_Diseases_957n/train.json     -> dinoV3_ToothVLM/coco/...
    #   - ../957n/image_filtered                 -> dinoV3_ToothVLM/../957n/image_filtered
    # 本文件位于 new_dinoyolo_src/data/ 下，向上三级才是仓库根目录。
    # （注意：不能只回退两级 parent，那只会到 new_dinoyolo_src，导致路径错位。）
    project_root = Path(__file__).resolve().parents[2]
    image_dir = (project_root / config.IMAGE_DIR).resolve()
    train_json = project_root / config.TRAIN_JSON
    val_json = project_root / config.VAL_JSON
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Configured IMAGE_DIR does not exist: {image_dir}")

    train_dataset = CocoYOLODataset(
        train_json,
        image_dir,
        (config.IMG_SIZE, config.IMG_SIZE),
        drop_empty=config.DROP_EMPTY,
        augment=True,
        config=config,
    )
    val_dataset = CocoYOLODataset(
        val_json,
        image_dir,
        (config.IMG_SIZE, config.IMG_SIZE),
        drop_empty=config.DROP_EMPTY,
        augment=False,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

