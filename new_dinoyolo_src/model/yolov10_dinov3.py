import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from model.dinov3_backbone import Dinov3Backbone, ConvGNAct
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple
from ultralytics.nn.modules.head import v10Detect
from ultralytics.utils.loss import E2ELoss
from config.config import Config

class DinoV3Adapter(nn.Module):
    """
    #### 现有的DinoV3Backbone输出的特征图是兼容Faster R-CNN的，但YOLOv10需要一个特定的输出格式。
    #### 这个类将DinoV3Backbone的输出转换为YOLOv10所需的格式。

    **输入输出：**
    - 输入： 由DinoV3Backbone输出的特征图字典。
    - 输出： 一个列表，包含三个特征图，分别对应YOLOv10所需的不同尺度(stride 8, 16, 32)。其中丢弃了最大的stride 64。
    """

    def __init__(self, backbone_model, embed_dim = 768, out_indices=None):
        super().__init__()
        self.backbone = Dinov3Backbone(
            backbone_model,
            embed_dim=embed_dim,
            out_channels=256,
            out_indices=out_indices,
        )

    def forward(self, x):
        # 获取DinoV3Backbone的特征图
        features = self.backbone(x)
        # 返回一个列表，包含三个特征图，分别对应YOLOv10所需的不同尺度
        return [
            features['0'],   # Stride 8 特征图
            features['1'],  # Stride 16 特征图
            features['2']   # Stride 32 特征图
            , features['3'],
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


class CascadeDetectHead(nn.Module):
    """Two-phase detection head using a single v10Detect.

    Phase 1 (epochs 1..stage1_epochs):
        Backbone + neck + detect head are trained normally.
        stage2_refine convs are frozen (zero-initialized residuals → no-op).

    Phase 2 (epochs stage1_epochs+1..end):
        stage2_refine is unfrozen and jointly trained at a lower LR.
        The refinement is a residual add on top of neck features, so at the
        moment of unfreezing the forward pass is identical to phase 1 and no
        loss spike occurs — there is no second v10Detect head, no reg_max
        mismatch, and no stale BN statistics to worry about.
    """

    def __init__(self, nc, ch, stage1_epochs=20):
        super().__init__()
        # Single shared detection head — same reg_max throughout training.
        try:
            self.detect = v10Detect(nc=nc, ch=ch)
        except TypeError:
            self.detect = v10Detect(nc=nc, ch=ch)
        # Lightweight per-scale residual refinement convs, frozen in phase 1.
        self.stage2_refine = nn.ModuleList([ConvBNAct(c, c, 3) for c in ch])
        # Zero-init → identity residual at start of phase 2.
        for refine in self.stage2_refine:
            nn.init.zeros_(refine.block[0].weight)
            if refine.block[0].bias is not None:
                nn.init.zeros_(refine.block[0].bias)
        # Proxy attributes consumed by Ultralytics E2ELoss / DetectionLoss.
        for attr in ("nc", "nl", "no", "reg_max", "type"):
            if hasattr(self.detect, attr):
                setattr(self, attr, getattr(self.detect, attr))
        self.stage1_epochs = stage1_epochs
        self.current_epoch = 1
        self.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
        # Start with refinement frozen.
        self._set_refine_grad(False)

    def _set_refine_grad(self, enabled: bool):
        for p in self.stage2_refine.parameters():
            p.requires_grad = enabled

    @property
    def stage2_enabled(self):
        return self.current_epoch > self.stage1_epochs

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self._set_refine_grad(self.stage2_enabled)

    def forward(self, features):
        if self.stage2_enabled:
            # Residual refinement: output is features + small learned delta.
            # At the first epoch of phase 2 the convs are still zero → no-op.
            features = [f + r(f) for f, r in zip(features, self.stage2_refine)]
        return self.detect(features)

    def bias_init(self):
        stride = self.stride.clone()
        if torch.any(stride <= 0):
            raise ValueError(f"Cascade head strides must be positive: {stride.tolist()}")
        self.detect.stride = stride.to(next(self.detect.parameters()).device)
        self.detect.bias_init()


class DinoPANNeck(nn.Module):
    """Three-scale PAN/FPN neck for P3/P4/P5 from DinoV3Adapter."""

    def __init__(self, in_channels=(256, 256, 256)):
        super().__init__()
        self.p2_lateral = ConvBNAct(128, 64, kernel_size=1)
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
        self.p2_out = ConvBNAct(64, 64)

    def forward(self, features):
        if len(features) == 4:
            p2, p3, p4, p5 = features
            p2 = self.p2_out(self.p2_lateral(p2))
        else:
            p3, p4, p5 = features
            p2 = self.p2_out(self.p2_lateral(F.interpolate(p3, scale_factor=2, mode="nearest")))
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
            p2,
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

    def __init__(self, backbone_model, embed_dim=768, num_classes=None, out_indices=None, config=None):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided or inferred before model construction")
        if config is None:
            raise ValueError(
                "config must be provided (pass the training script's Config which holds the actual values)"
            )
        self.config = config
        self.backbone = DinoV3Adapter(backbone_model, embed_dim=embed_dim, out_indices=out_indices)
        self.neck = DinoPANNeck(in_channels=(256, 256, 256))
        self.detect_head = CascadeDetectHead(
            nc=num_classes,
            ch=(64, 256, 512, 1024),
            stage1_epochs=getattr(config, "CASCADE_STAGE1_EPOCHS", 20),

        )

        # E2ELoss 计算， 需要在 forward 中传入 targets，返回 loss 和 loss_items
        self.model = nn.ModuleList([self.backbone, self.neck, self.detect_head])
        self.args = SimpleNamespace(box=7.5, cls=1.5, dfl=1.5, epochs=self.config.EPOCHS)
        self.focal_loss_gamma = 2.0
        if self.config.CLASS_WEIGHTS is None:
            self.class_weights = None
        else:
            if len(self.config.CLASS_WEIGHTS) != num_classes:
                raise ValueError(
                    "CLASS_WEIGHTS must have exactly NUM_CLASSES entries"
                )
            self.register_buffer(
                "class_weights",
                torch.tensor(self.config.CLASS_WEIGHTS, dtype=torch.float32),
            )
        self.register_buffer(
            "dino_mean",
            torch.tensor(self.config.DINO_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "dino_std",
            torch.tensor(self.config.DINO_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.detect_head.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
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

    def forward(self, images, targets=None, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.config.CONF_THRESHOLD
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


def build_model(num_classes=None, config=None):
    """
    加载 DinoV3 backbone 并构建 YOLOv10 模型。

    Args:
        num_classes: 前景类别数量。如果为 None，将从 config.TRAIN_JSON 推断。
        config: 实际配置来源（训练/评估脚本里继承并填好值的 Config 类）。
                必填——本文件里的 Config 只做“声明”，不提供取值。

    Returns:
        YOLOv10WithDinoV3: 构建的 YOLOv10模型，使用 DinoV3 作为 backbone。
    """
    if config is None:
        raise ValueError(
            "config must be provided (pass the training script's Config which holds the actual values)"
        )
    if num_classes is None:
        # 模型文件不能反向 import 训练脚本（会循环导入），故这里内联推断逻辑。
        # 本文件位于 new_dinoyolo_src/model/ 下，向上三级才是仓库根目录。
        project_root = Path(__file__).resolve().parents[2]
        with (project_root / config.TRAIN_JSON).open("r", encoding="utf-8") as file:
            coco_data = json.load(file)
        used_ids = sorted({int(item["category_id"]) for item in coco_data.get("annotations", [])})
        if not used_ids:
            raise ValueError(f"No annotated category IDs found in {config.TRAIN_JSON}")
        if used_ids != list(range(1, len(used_ids) + 1)):
            raise ValueError(f"COCO category IDs must be contiguous 1..N, got {used_ids}")
        num_classes = len(used_ids)
    backbone_model = torch.hub.load(
        config.REPO_DIR,
        "dinov3_vitb16",
        source="local",
        weights=config.WEIGHTS,
    )
    for parameter in backbone_model.parameters():
        parameter.requires_grad = False
    for parameter in backbone_model.blocks[-config.UNFREEZE_BLOCKS:].parameters():
        parameter.requires_grad = True
    return YOLOv10WithDinoV3(
        backbone_model,
        embed_dim=backbone_model.embed_dim,
        num_classes=num_classes,
        out_indices=config.BACKBONE_OUT_INDICES,
        config=config,
    )
