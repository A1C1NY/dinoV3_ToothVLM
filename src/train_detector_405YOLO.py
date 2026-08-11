import os
import json
import math
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
    OUTPUT_DIR = "res_checkpoints/multi_disease_562_expt_v2_adaptive_low_threshold"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # 数据集配置
    DROP_EMPTY = True     # 是否丢弃没有标注的图片

    # 数据增强：只做几何变换。
    # 明确不做曝光/亮度/对比度/饱和度/色相扰动——颜色本身是牙科病灶的判别特征
    # （caries 偏暗、calculus 偏黄白、tooth_discoloration 由颜色定义），
    # 扰动颜色会破坏类间可分性。
    AUG_HFLIP = 0.5           # 水平翻转概率
    AUG_AFFINE = 0.7          # 随机仿射概率（缩放/平移/小角度旋转）
    AUG_SCALE = 0.25          # 缩放抖动幅度：ratio ∈ [1-0.25, 1+0.25]
    AUG_TRANSLATE = 0.10      # 平移幅度，占边长比例
    AUG_ROTATE = 7.0          # 旋转角度上限（度）。轴对齐框会随旋转膨胀，故取小值
    AUG_MIN_BOX_SIZE = 4.0    # 变换后小于该边长（像素）的框丢弃
    AUG_MIN_BOX_KEEP = 0.25   # 变换后保留面积低于原面积该比例的框丢弃
    PAD_VALUE = 114           # letterbox 填充灰度值（YOLO 惯例）

    # 梯度裁剪：拦住尖峰，但不要把每一步都裁。None 表示不裁剪。
    # 本项目实测（前 50 步，不裁剪）：中位数 154、p90 303、但出现过 4793 的尖峰。
    # 取 10 会裁掉 100% 的步（等于把学习率砍掉一个数量级）；取 200 只裁约 22%，
    # 拦住尾部与尖峰而放过中位数。换数据集/改 batch 后建议重新看日志里的 grad_norm。
    CLIP_GRAD_NORM = 200.0

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 50  # <--- 增加总轮次到35，给微调留足空间
    LR = 0.001
    BACKBONE_LR = 0.0001
    WARMUP_EPOCHS = 5
    UNFREEZE_BLOCKS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 从 ViT 的哪三个 block 取特征构造 P3/P4/P5（升序 = shallow→deep）。
    # ViT-B/16 共 12 个 block；(5, 8, 11) 在 UNFREEZE_BLOCKS=6 时有两个落在可训练区。
    # 设为 None 则退回旧行为：只用最后一层，三尺度由它重采样派生。
    BACKBONE_OUT_INDICES = (5, 8, 11)

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
        0: 0.28,  # Caries
        1: 0.14,  # Calculus
        2: 0.20,  # Mouth_Ulcer
        3: 0.28,  # Tooth_Discoloration
    }

    # VAL_CLASS_THRESHOLDS = {
    #     0: 0.30,  # Caries
    #     1: 0.50,  # Calculus
    #     2: 0.20,  # Mouth_Ulcer
    #     3: 0.30,  # Tooth_Discoloration
    # }


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

    def __init__(self, backbone_model, embed_dim=768, num_classes=None, out_indices=None):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided or inferred before model construction")
        self.backbone = DinoV3Adapter(backbone_model, embed_dim=embed_dim, out_indices=out_indices)
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
        out_indices=Config.BACKBONE_OUT_INDICES,
    )


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

        image, ratio, pad_x, pad_y = letterbox_image(
            image, self.image_width, self.image_height, pad_value=Config.PAD_VALUE
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
            if random.random() < Config.AUG_HFLIP:
                image_tensor = image_tensor.flip(-1)
                if len(boxes):
                    left = boxes[:, 0].clone()
                    boxes[:, 0] = self.image_width - boxes[:, 2]
                    boxes[:, 2] = self.image_width - left
            if random.random() < Config.AUG_AFFINE:
                warped, new_boxes, new_labels = random_affine(
                    image_tensor,
                    boxes,
                    labels,
                    degrees=Config.AUG_ROTATE,
                    translate=Config.AUG_TRANSLATE,
                    scale=Config.AUG_SCALE,
                    pad_value=Config.PAD_VALUE,
                    min_box_size=Config.AUG_MIN_BOX_SIZE,
                    min_box_keep=Config.AUG_MIN_BOX_KEEP,
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
                ratio = target["letterbox_ratio"]
                pad_x = target["pad_x"]
                pad_y = target["pad_y"]
                for x1, y1, x2, y2, score, label in prediction.detach().cpu().tolist():
                    # letterbox 反变换：先去填充偏移，再按比例还原到原图坐标。
                    x1, x2 = (x1 - pad_x) / ratio, (x2 - pad_x) / ratio
                    y1, y2 = (y1 - pad_y) / ratio, (y2 - pad_y) / ratio
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
    # 只有 ViT 本体（backbone.backbone.backbone.*）用低学习率。
    # 注意不能用 "backbone.backbone" 前缀：那会把 Dinov3Backbone 里随机初始化的
    # 金字塔投影层（p3_proj/p4_proj/... 或 conv_c4/deconv_c3）也归进 backbone 组，
    # 让最需要从头学的层以 1/10 的学习率训练。
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.backbone.backbone"):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    print(f"Param groups: backbone(ViT)={len(backbone_params)}, head/neck/pyramid={len(head_params)}")

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

            # clip_grad_norm_ 返回裁剪前的总范数，正好替代原来的手写逐参数求和。
            # max_norm=inf 时只统计不裁剪，便于对照实验。
            max_norm = Config.CLIP_GRAD_NORM if Config.CLIP_GRAD_NORM else float("inf")
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            grad_norm_sum += float(total_norm)
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
