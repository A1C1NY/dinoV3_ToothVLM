import json
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

    def _load_base_sample(self, index):
        """Load one letterboxed image and its boxes without random augmentation."""
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

        return image_tensor, boxes, labels, {
            "image_id": image_id,
            "original_width": original_width,
            "original_height": original_height,
            "letterbox_ratio": ratio,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }

    def _make_mosaic(self, index):
        """
        mosaic增强方法，针对单张图像，随机选择另外三张图像，将四张图像拼接成一张大图。
        用来增加对较小目标的检测能力。
        Args:
            index: 当前图像的索引。
        Returns:
            canvas: 拼接后的图像张量。
            boxes: 拼接后图像的边界框张量。
            labels: 拼接后图像的标签张量。
            metadata: 包含原始图像信息的字典。
        """
        indices = [index] + [random.randrange(len(self.ids)) for _ in range(3)]
        min_center, max_center = self.config.MOSAIC_CENTER_RANGE
        center_x = round(random.uniform(min_center, max_center) * self.image_width)
        center_y = round(random.uniform(min_center, max_center) * self.image_height)
        canvas = torch.full(
            (3, self.image_height, self.image_width),
            self.config.PAD_VALUE / 255.0,
            dtype=torch.float32,
        )
        all_boxes, all_labels = [], []
        quadrants = (
            (0, 0, center_x, center_y),
            (center_x, 0, self.image_width, center_y),
            (0, center_y, center_x, self.image_height),
            (center_x, center_y, self.image_width, self.image_height),
        )

        for source_index, (left, top, right, bottom) in zip(indices, quadrants):
            image, boxes, labels, _ = self._load_base_sample(source_index)
            tile_width, tile_height = right - left, bottom - top
            resized = F.interpolate(
                image.unsqueeze(0), size=(tile_height, tile_width),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            canvas[:, top:bottom, left:right] = resized
            if len(boxes):
                scale_x = tile_width / self.image_width
                scale_y = tile_height / self.image_height
                new_boxes = boxes.clone()
                new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * scale_x + left
                new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * scale_y + top
                all_boxes.append(new_boxes)
                all_labels.append(labels)

        boxes = torch.cat(all_boxes) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)
        metadata = {
            "image_id": self.ids[index],
            "original_width": self.image_width,
            "original_height": self.image_height,
            "letterbox_ratio": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        return canvas, boxes, labels, metadata

    @staticmethod
    def _max_iou(candidate, boxes):
        if not len(boxes):
            return 0.0
        intersection_left = torch.maximum(candidate[0], boxes[:, 0])
        intersection_top = torch.maximum(candidate[1], boxes[:, 1])
        intersection_right = torch.minimum(candidate[2], boxes[:, 2])
        intersection_bottom = torch.minimum(candidate[3], boxes[:, 3])
        intersection = (intersection_right - intersection_left).clamp(min=0) * (
            intersection_bottom - intersection_top
        ).clamp(min=0)
        candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return float((intersection / (candidate_area + box_areas - intersection).clamp(min=1e-6)).max())

    def _copy_paste_small_objects(self, image, boxes, labels):
        """Paste small annotated objects, with surrounding context, at free locations."""
        max_area = self.config.COPY_PASTE_MAX_BOX_AREA_RATIO * self.image_width * self.image_height
        pasted_boxes, pasted_labels = [], []
        result = image.clone()
        occupied_boxes = boxes.clone()

        for _ in range(self.config.COPY_PASTE_MAX_OBJECTS):
            donor_index = random.randrange(len(self.ids))
            donor_image, donor_boxes, donor_labels, _ = self._load_base_sample(donor_index)
            donor_areas = (donor_boxes[:, 2] - donor_boxes[:, 0]) * (donor_boxes[:, 3] - donor_boxes[:, 1])
            eligible = torch.where(donor_areas <= max_area)[0]
            if not len(eligible):
                continue
            box_index = int(eligible[random.randrange(len(eligible))])
            source_box = donor_boxes[box_index]
            source_width = max(1, round(float(source_box[2] - source_box[0])))
            source_height = max(1, round(float(source_box[3] - source_box[1])))
            context_x = round(source_width * self.config.COPY_PASTE_CONTEXT_RATIO)
            context_y = round(source_height * self.config.COPY_PASTE_CONTEXT_RATIO)
            crop_left = max(0, round(float(source_box[0])) - context_x)
            crop_top = max(0, round(float(source_box[1])) - context_y)
            crop_right = min(self.image_width, round(float(source_box[2])) + context_x)
            crop_bottom = min(self.image_height, round(float(source_box[3])) + context_y)
            crop_width, crop_height = crop_right - crop_left, crop_bottom - crop_top
            if crop_width <= 0 or crop_height <= 0 or crop_width > self.image_width or crop_height > self.image_height:
                continue

            for _ in range(20):
                destination_left = random.randint(0, self.image_width - crop_width)
                destination_top = random.randint(0, self.image_height - crop_height)
                new_box = source_box.clone()
                new_box[[0, 2]] += destination_left - crop_left
                new_box[[1, 3]] += destination_top - crop_top
                if self._max_iou(new_box, occupied_boxes) <= self.config.COPY_PASTE_MAX_IOU:
                    result[:, destination_top:destination_top + crop_height, destination_left:destination_left + crop_width] = donor_image[:, crop_top:crop_bottom, crop_left:crop_right]
                    pasted_boxes.append(new_box.unsqueeze(0))
                    pasted_labels.append(donor_labels[box_index].reshape(1))
                    occupied_boxes = torch.cat([occupied_boxes, new_box.unsqueeze(0)])
                    break

        if pasted_boxes:
            boxes = torch.cat([boxes, *pasted_boxes])
            labels = torch.cat([labels, *pasted_labels])
        return result, boxes, labels

    def __getitem__(self, index):
        if self.augment and random.random() < self.config.MOSAIC_PROB:
            image_tensor, boxes, labels, metadata = self._make_mosaic(index)
        else:
            image_tensor, boxes, labels, metadata = self._load_base_sample(index)

        if self.augment and random.random() < self.config.COPY_PASTE_PROB:
            image_tensor, boxes, labels = self._copy_paste_small_objects(image_tensor, boxes, labels)

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
            **metadata,
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

    # 训练和验证可以分别控制：训练默认过滤空标注，验证可以单独保留空标注样本。
    train_drop_empty = getattr(config, "DROP_EMPTY", True)
    val_include_empty = getattr(config, "VAL_INCLUDE_EMPTY_ANNOTATIONS", None)
    if val_include_empty is not None:
        val_drop_empty = not val_include_empty
    else:
        val_drop_empty = getattr(config, "VAL_DROP_EMPTY", train_drop_empty)

    include_empty = getattr(config, "INCLUDE_EMPTY_ANNOTATIONS", None)
    if include_empty is not None:
        train_drop_empty = not include_empty
        val_drop_empty = not include_empty
    elif getattr(config, "DROP_EMPTY", None) is None:
        train_drop_empty = True
        val_drop_empty = True

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
        drop_empty=train_drop_empty,
        augment=True,
        config=config,
    )
    val_dataset = CocoYOLODataset(
        val_json,
        image_dir,
        (config.IMG_SIZE, config.IMG_SIZE),
        drop_empty=val_drop_empty,
        augment=False,
        config=config,
    )

    oversample_category_id = config.OVERSAMPLE_CATEGORY_ID
    oversample_factor = config.OVERSAMPLE_FACTOR
    if oversample_factor < 1.0:
        raise ValueError("OVERSAMPLE_FACTOR must be at least 1.0")
    sample_weights = torch.tensor([
        oversample_factor
        if any(
            annotation["category_id"] == oversample_category_id
            for annotation in train_dataset.annotations.get(image_id, [])
        )
        else 1.0
        for image_id in train_dataset.ids
    ], dtype=torch.double)
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    oversampled_images = int((sample_weights > 1.0).sum().item())
    print(
        f"Train sampling: category_id={oversample_category_id}, "
        f"factor={oversample_factor:.2f}, affected_images={oversampled_images}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
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

