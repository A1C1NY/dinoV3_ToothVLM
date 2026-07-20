import os
import json
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path
from types import SimpleNamespace
from ultralytics.nn.modules.head import v10Detect
from ultralytics.utils.loss import E2ELoss

from dinov3_backbone import Dinov3Backbone

class Config:
    # 路径配置
    REPO_DIR = "."
    
    # # --- 选项 B：所有疾病混合训练 (All Diseases) ---
    IMAGE_DIR = "../405/image_filtered"
    TRAIN_JSON = "coco/All_Diseases/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    VAL_JSON = "coco/All_Diseases/val.json"
    SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR = "res_checkpoints/multi_disease_405_expt"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


    # 数据集配置
    DROP_EMPTY = True     # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 50  # <--- 增加总轮次到35，给微调留足空间
    LR = 0.001
    BACKBONE_LR = 0.00001
    WARMUP_EPOCHS = 5
    UNFREEZE_BLOCKS = 4
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
    NUM_CLASSES = 5
    CONF_THRESHOLD = 0.25
    # Set to a sequence with NUM_CLASSES entries when class reweighting is needed.
    CLASS_WEIGHTS = None
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)


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

    def __init__(self, backbone_model, embed_dim=768, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.backbone = DinoV3Adapter(backbone_model, embed_dim=embed_dim)
        self.neck = DinoPANNeck(in_channels=(256, 256, 256))
        self.detect_head = v10Detect(nc=num_classes, ch=(256, 512, 1024))

        # E2ELoss accesses model.args, model.model[-1], and model.class_weights.
        self.model = nn.ModuleList([self.backbone, self.neck, self.detect_head])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=Config.EPOCHS)
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
        self.criterion = None

    @staticmethod
    def targets_to_yolo_batch(images, targets):
        """Convert torchvision COCO targets (xyxy, labels 1..N) to Ultralytics loss targets."""
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

            batch = self.targets_to_yolo_batch(images, targets)
            loss_items, loss_detached = self.criterion(predictions, batch)
            self.criterion.update()

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


def build_model():
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
    return YOLOv10WithDinoV3(backbone_model, embed_dim=backbone_model.embed_dim)


