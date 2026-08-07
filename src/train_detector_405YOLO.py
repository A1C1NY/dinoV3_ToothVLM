import os
import json
import re
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from types import SimpleNamespace
from ultralytics.nn.modules.head import v10Detect
from ultralytics.utils.loss import E2ELoss
from pycocotools.coco import COCO
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from dinov3_backbone import Dinov3Backbone

class Config:
    # 路径配置
    REPO_DIR = "."
    
    # # --- 选项 B：所有疾病混合训练 (All Diseases) ---
    IMAGE_DIR = "../562/image_filtered"
    TRAIN_JSON = "coco/All_Diseases/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    VAL_JSON = "coco/All_Diseases/val.json"
    SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR = "res_checkpoints/multi_disease_562_expt87"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # 数据集配置
    DROP_EMPTY = True     # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 50  # <--- 增加总轮次到35，给微调留足空间
    LR = 0.001
    BACKBONE_LR = 0.0001
    WARMUP_EPOCHS = 5
    UNFREEZE_BLOCKS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 继续训练 (可选)
    RESUME_CHECKPOINT = None  # 填写 .pth 文件路径以继续训练，例如 r"..."
    START_EPOCH = 1           # 继续训练时的起始 epoch

    # 验证与评估参数
    IOU_THRESHOLD = 0.5       # 用于评估时判断正样本的 IoU 阈值
    SCORE_THRESHOLD = 0.5     # 用于过滤低置信度预测的阈值

    # 模型参数
    MIN_SIZE = 1200
    MAX_SIZE = 1200
    NUM_CLASSES = None
    CONF_THRESHOLD = 0.001

    # 对目标的自适应阈值（类别ID从0开始）
    VAL_CLASS_THRESHOLDS = {
        0: 0.30,  # Caries
        1: 0.50,  # Calculus
        2: 0.20,  # Mouth_Ulcer
        3: 0.30,  # Tooth_Discoloration
    }

    VAL_CONF_THRESHOLD_DEFAULT = 0.3 # 不在 VAL_CLASS_THRESHOLDS 中的类别使用此默认阈值

    # Set to a sequence with NUM_CLASSES entries when class reweighting is needed.
    CLASS_WEIGHTS = [1.2, 1.3, 2.5, 1.1]
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)
    IMG_SIZE = 640
    NUM_WORKERS = 0
    SEED = 42


class DinoV3Adapter(nn.Module):
    """
    #### 现有的DinoV3Backbone输出的特征图是兼容Faster R-CNN的，但YOLOv10需要一个特定的输出格式。
    #### 这个类将DinoV3Backbone的输出转换为YOLOv10所需的格式。

    **输入输出：**
    - 输入： 由DinoV3Backbone输出的特征图字典。
    - 输出： 一个列表，包含三个特征图，分别对应YOLOv10所需的不同尺度(stride 8, 16, 32)。其中丢弃了最大的stride 64。
    """

    def __init__(self, backbone_model, embed_dim = 768):
        super().__init__()
        self.backbone = Dinov3Backbone(backbone_model, embed_dim=embed_dim, out_channels=256)

    def forward(self, x):
        # 获取DinoV3Backbone的特征图
        features = self.backbone(x)
        # 返回一个列表，包含三个特征图，分别对应YOLOv10所需的不同尺度
        return [
            features['0'],   # Stride 8 特征图
            features['1'],  # Stride 16 特征图
            features['2']   # Stride 32 特征图
        ]
    
class YOLOv10WithDinoV3(nn.Module):

    def __init__(self, backbone_modle, neck, detect_head, loss_fn=None, embed_dim=768):
        super().__init__()
        self.backbone = DinoV3Adapter(backbone_modle, embed_dim=embed_dim)
        self.neck = neck
        self.detect_head = detect_head
        self.loss_fn = loss_fn

    def forward(self, images, targets=None):
        feature_group = self.backbone(images)
        features = self.neck(feature_group)
        predictions = self.detect_head(features)
        

        if self.training and targets is not None:
            if self.loss_fn is None:
                raise ValueError("Loss function is not defined.")
            
            loss, loss_items = self.loss_fn(predictions, targets,)
        
            return {
                "loss": loss,
                "loss_items": loss_items,
                "predictions": predictions,
            }
        
        return self.postprocess(predictions)
    
    def postprocess(self, predictions):
        # 解码 bbox、类别分数，并执行 NMS 或 YOLOv10 的 NMS-free 推理
        return predictions


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation block.

    Args:
        c1: 输入通道数。
        c2: 输出通道数。
        kernel_size: 卷积核大小。
        stride: 卷积步幅。
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DinoPANNeck(nn.Module):
    """Three-scale PAN/FPN neck for P3/P4/P5 from DinoV3Adapter."""

    def __init__(self, in_channels=(256, 256, 256)):
        super().__init__()
        self.p3_lateral = ConvBNAct(in_channels[0], 128, kernel_size=1)
        self.p4_lateral = ConvBNAct(in_channels[1], 256, kernel_size=1)
        self.p5_lateral = ConvBNAct(in_channels[2], 512, kernel_size=1)
        self.p4_top_down = ConvBNAct(512 + 256, 512)
        self.p3_top_down = ConvBNAct(512 + 128, 256)
        self.p3_downsample = ConvBNAct(256, 256, stride=2)
        self.p4_bottom_up = ConvBNAct(256 + 512, 512)
        self.p4_downsample = ConvBNAct(512, 512, stride=2)
        self.p5_bottom_up = ConvBNAct(512 + 512, 1024)
        self.p3_out = ConvBNAct(256, 256)
        self.p4_out = ConvBNAct(512, 512)
        self.p5_out = ConvBNAct(1024, 1024)

    def forward(self, features):
        p3, p4, p5 = features
        p3 = self.p3_lateral(p3)
        p4 = self.p4_lateral(p4)
        p5 = self.p5_lateral(p5)

        p4_top_down = self.p4_top_down(torch.cat([
            F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4
        ], dim=1))
        p3_top_down = self.p3_top_down(torch.cat([
            F.interpolate(p4_top_down, size=p3.shape[-2:], mode="nearest"), p3
        ], dim=1))
        p4_bottom_up = self.p4_bottom_up(torch.cat([
            self.p3_downsample(p3_top_down), p4_top_down
        ], dim=1))
        p5_bottom_up = self.p5_bottom_up(torch.cat([
            self.p4_downsample(p4_bottom_up), p5
        ], dim=1))

        return [
            self.p3_out(p3_top_down),
            self.p4_out(p4_bottom_up),
            self.p5_out(p5_bottom_up),
        ]


class YOLOv10WithDinoV3(nn.Module):
    """
    ### YOLOv10模型，使用DinoV3作为骨干网络。

    **模型结构：**
    - Backbone: 使用DinoV3作为特征提取器。
      - 结合DinoV3Adapter，将DinoV3的输出特征图转换为YOLOv10所需的格式。
    - Neck: 使用YOLOv10的特征融合模块。
    - Head: 使用YOLOv10的检测头，输出最终的边界框和类别预测。

    **输入输出：**
    - 输入: [B, 3, H, W] 的图像张量。
    - 输出: YOLOv10的预测结果，包括边界框坐标、类别概率等。

    """

    def __init__(self, backbone_model, embed_dim=768, num_classes=None):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided or inferred before model construction")
        self.backbone = DinoV3Adapter(backbone_model, embed_dim=embed_dim)
        self.neck = DinoPANNeck(in_channels=(256, 256, 256))
        self.detect_head = v10Detect(nc=num_classes, ch=(256, 512, 1024))

        # E2ELoss 计算， 需要在 forward 中传入 targets，返回 loss 和 loss_items
        self.model = nn.ModuleList([self.backbone, self.neck, self.detect_head])
        self.args = SimpleNamespace(box=7.5, cls=1.5, dfl=1.5, epochs=Config.EPOCHS)
        self.focal_loss_gamma = 2.0
        if Config.CLASS_WEIGHTS is None:
            self.class_weights = None
        else:
            if len(Config.CLASS_WEIGHTS) != num_classes:
                raise ValueError(
                    "CLASS_WEIGHTS must have exactly NUM_CLASSES entries"
                )
            self.register_buffer(
                "class_weights",
                torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32),
            )
        self.register_buffer(
            "dino_mean",
            torch.tensor(Config.DINO_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "dino_std",
            torch.tensor(Config.DINO_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.detect_head.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect_head.bias_init()
        self.criterion = None  # 初始化为 None，在 forward 中创建 E2ELoss

    @staticmethod
    def targets_to_yolo_batch(images, targets):
        """
        将 COCO 风格的目标字典转换为 YOLOv10 所需的批处理格式。
        Args:
            images: 输入图像张量，形状为 [B, C, H, W]。
            targets: COCO 风格的目标列表，每个目标是一个字典，包含 "boxes" 和 "labels"。
        Returns:
            一个字典，包含以下键：
                - "batch_idx": 每个目标对应的图像索引。
                - "cls": 每个目标的类别标签。
                - "bboxes": 每个目标的边界框，格式为 [x_center, y_center, width, height]，归一化到 [0, 1]。
        
        """
        height, width = images.shape[-2:]
        device = images.device
        batch_indices, classes, boxes = [], [], []
        for image_index, target in enumerate(targets):
            target_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            target_labels = target["labels"].to(device=device, dtype=torch.long)

            if target_boxes.numel() == 0:
                continue

            xywh = target_boxes.clone()
            xywh[:, 0] = (target_boxes[:, 0] + target_boxes[:, 2]) / (2 * width)
            xywh[:, 1] = (target_boxes[:, 1] + target_boxes[:, 3]) / (2 * height)
            xywh[:, 2] = (target_boxes[:, 2] - target_boxes[:, 0]) / width
            xywh[:, 3] = (target_boxes[:, 3] - target_boxes[:, 1]) / height

            batch_indices.append(torch.full((len(target_labels),), image_index, device=device, dtype=torch.long))
            classes.append(target_labels - 1)
            boxes.append(xywh.clamp_(0, 1))

        if not boxes:
            return {
                "batch_idx": torch.zeros(0, device=device, dtype=torch.long),
                "cls": torch.zeros(0, device=device, dtype=torch.long),
                "bboxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
            }
        return {
            "batch_idx": torch.cat(batch_indices),
            "cls": torch.cat(classes),
            "bboxes": torch.cat(boxes),
        }

    def forward_features(self, images):
        images = (images - self.dino_mean) / self.dino_std
        return self.neck(self.backbone(images))

    def forward(self, images, targets=None, conf_threshold=Config.CONF_THRESHOLD):
        self.detect_head.stride = self.detect_head.stride.to(images.device)
        predictions = self.detect_head(self.forward_features(images))
        if self.training:
            if targets is None:
                return predictions
            if self.criterion is None:
                self.criterion = E2ELoss(self)
                # E2ELoss 内部的 v8DetectionLoss 会自动读取 self.class_weights

            batch = self.targets_to_yolo_batch(images, targets)
            loss_items, loss_detached = self.criterion(predictions, batch)
            # E2ELoss returns [box, cls, dfl]; backward needs one scalar.
            loss = loss_items.sum()

            return {
                "loss": loss, 
                "loss_items": loss_detached, 
                "predictions": predictions
            }
        
        return self.postprocess(predictions, conf_threshold)

    @staticmethod
    def postprocess(predictions, conf_threshold):
        """v10Detect eval mode has already decoded boxes and applied its end-to-end top-k selection."""
        decoded = predictions[0] if isinstance(predictions, tuple) else predictions
        return [image_predictions[image_predictions[:, 4] >= conf_threshold] for image_predictions in decoded]


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


def build_model(num_classes=None):
    """
    加载 DinoV3 backbone 并构建 YOLOv10 模型。

    Args:
        num_classes: 前景类别数量。如果为 None，将从 COCO JSON 注释文件推断类别数量。

    Returns:
        YOLOv10WithDinoV3: 构建的 YOLOv10模型，使用 DinoV3 作为 backbone。
    """
    if num_classes is None:
        project_root = Path(__file__).resolve().parent.parent
        num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    backbone_model = torch.hub.load(
        Config.REPO_DIR,
        "dinov3_vitb16",
        source="local",
        weights=Config.WEIGHTS,
    )
    for parameter in backbone_model.parameters():
        parameter.requires_grad = False
    for parameter in backbone_model.blocks[-Config.UNFREEZE_BLOCKS:].parameters():
        parameter.requires_grad = True
    return YOLOv10WithDinoV3(
        backbone_model,
        embed_dim=backbone_model.embed_dim,
        num_classes=num_classes,
    )


class CocoYOLODataset(Dataset):
    """COCO-style dataset for the custom model.

    Args:
        annotation_file: COCO JSON 的路径，包含图像和注释信息。
        image_dir: 图像文件所在的目录。
        image_size: 输入图像的目标大小，格式为 (height, width)。
        drop_empty: 如果为 True，将丢弃没有标注的图像。
        augment: 如果为 True，将应用随机水平翻转数据增强。
    """

    def __init__(self, annotation_file, image_dir, image_size, drop_empty=False, augment=False):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.image_height, self.image_width = image_size
        self.augment = augment
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
        image = image.resize((self.image_width, self.image_height))
        image_tensor = pil_to_tensor(image).float() / 255.0

        scale_x = self.image_width / original_width
        scale_y = self.image_height / original_height
        boxes, labels = [], []
        for annotation in self.annotations.get(image_id, []):
            x, y, width, height = annotation["bbox"]
            boxes.append([
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ])
            labels.append(annotation["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.augment and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            if len(boxes):
                left = boxes[:, 0].clone()
                boxes[:, 0] = self.image_width - boxes[:, 2]
                boxes[:, 2] = self.image_width - left

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_width": original_width,
            "original_height": original_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
        return image_tensor, target


def detection_collate(batch):
    """
    Stack fixed-size images while preserving the per-image target dictionaries.
    """
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def build_dataloaders():
    """Build train/validation loaders from the generated All_Diseases COCO files."""
    project_root = Path(__file__).resolve().parent.parent
    image_dir = (project_root / Config.IMAGE_DIR).resolve()
    train_json = project_root / Config.TRAIN_JSON
    val_json = project_root / Config.VAL_JSON
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Configured IMAGE_DIR does not exist: {image_dir}")

    train_dataset = CocoYOLODataset(
        train_json,
        image_dir,
        (Config.IMG_SIZE, Config.IMG_SIZE),
        drop_empty=Config.DROP_EMPTY,
        augment=True,
    )
    val_dataset = CocoYOLODataset(
        val_json,
        image_dir,
        (Config.IMG_SIZE, Config.IMG_SIZE),
        drop_empty=Config.DROP_EMPTY,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, use_class_thresholds=True):
    """Evaluate predictions with COCO bbox metrics and print diagnostic counts."""
    model.eval()
    coco_results = []
    total_predictions = 0
    score_values = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)

            # Use class-specific thresholds if defined
            if use_class_thresholds:
                predictions = model(images, conf_threshold=0.001)  # 先用低阈值获取所有预测
                # 后处理：应用类别自适应阈值
                filtered_predictions = []
                for pred in predictions:
                    if len(pred) == 0:
                        filtered_predictions.append(pred)
                        continue
                    mask = torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
                    for i, p in enumerate(pred):
                        cls_id = int(p[5].item())
                        cls_thresh = Config.VAL_CLASS_THRESHOLDS.get(cls_id, Config.VAL_CONF_THRESHOLD_DEFAULT)
                        if p[4].item() >= cls_thresh:
                            mask[i] = True
                    filtered_predictions.append(pred[mask])
                predictions = filtered_predictions
            else:
                predictions = model(images, conf_threshold=Config.VAL_CONF_THRESHOLD_DEFAULT)
            for prediction, target in zip(predictions, targets):
                total_predictions += len(prediction)
                if len(prediction):
                    score_values.extend(prediction[:, 4].detach().cpu().tolist())
                scale_x = target["scale_x"]
                scale_y = target["scale_y"]
                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    x1, y1, x2, y2 = x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y
                    coco_results.append({
                        "image_id": target["image_id"],
                        "category_id": int(label) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })

    average_score = sum(score_values) / len(score_values) if score_values else 0.0
    max_score = max(score_values) if score_values else 0.0
    print(
        f"Validation predictions: total={total_predictions}, "
        f"average/image={total_predictions / max(1, len(val_loader.dataset)):.2f}, "
        f"max_score={max_score:.6f}, mean_score={average_score:.6f}"
    )

    if not coco_results:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0}

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
    }


def train():
    """Run the conventional train/validation loop for the custom YOLOv10 model."""
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    output_dir = Path(__file__).resolve().parent.parent / Config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders()
    project_root = Path(__file__).resolve().parent.parent
    num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    print(f"Device: {device}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset)}")
    print(f"Classes inferred from COCO: {num_classes}")
    print(f"Batch size: {Config.BATCH_SIZE}, image size: {Config.IMG_SIZE}")

    model = build_model(num_classes=num_classes).to(device)
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.backbone"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": Config.BACKBONE_LR},
        {"params": head_params, "lr": Config.LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6,
    )
    best_map = -1.0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        loss_sum = torch.zeros(3, device=device)
        grad_norm_sum = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}"):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images, targets)
            loss = output["loss"]
            loss.backward()

            squared_grad_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared_grad_norm += parameter.grad.detach().float().pow(2).sum().item()
            grad_norm_sum += squared_grad_norm ** 0.5
            optimizer.step()

            total_loss += loss.item()
            loss_sum += output["loss_items"].to(device)

        if model.criterion is not None:
            model.criterion.update()
        scheduler.step()

        steps = max(1, len(train_loader))
        print(
            f"Epoch {epoch}: loss={total_loss / steps:.6f}, "
            f"box/cls/dfl={(loss_sum / steps).detach().cpu().tolist()}, "
            f"grad_norm={grad_norm_sum / steps:.6e}, "
            f"lr={[group['lr'] for group in optimizer.param_groups]}"
        )

        metrics = evaluate_model(model, val_loader, device)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            torch.save(checkpoint, output_dir / "best_map.pth")
            print(f"New best mAP@[.5:.95]: {best_map:.6f}")


if __name__ == "__main__":
    train()
