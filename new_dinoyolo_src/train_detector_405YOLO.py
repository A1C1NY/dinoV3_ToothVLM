import json
import math
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pycocotools.coco import COCO
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datetime import datetime

from utils.log import TeeLogger, TeeLoggerStderr
from model.yolov10_dinov3 import YOLOv10WithDinoV3, Config as BaseConfig
from model.yolov10_dinov3 import build_model
from data.model_data import build_dataloaders, infer_num_classes


class Config(BaseConfig):
    """实际取值在此填入；结构（字段声明）继承自模型文件的 Config 基类。"""

    # 路径配置
    REPO_DIR = "."
    
    # # --- 选项 B：所有疾病混合训练 (All Diseases) ---
    IMAGE_DIR = "../767/image"
    TRAIN_JSON = "coco/All_Diseases_767/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    VAL_JSON = "coco/All_Diseases_767/val.json"
    SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR = "res_checkpoints/multi_disease_767_expt_v3_1_highsize"
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

    # Small-lesion augmentation, used by the training dataset only.
    MOSAIC_PROB = 0.30
    MOSAIC_CENTER_RANGE = (0.45, 0.55)
    COPY_PASTE_PROB = 0.00
    COPY_PASTE_MAX_BOX_AREA_RATIO = 0.02
    COPY_PASTE_MAX_OBJECTS = 2
    COPY_PASTE_CONTEXT_RATIO = 0.20
    COPY_PASTE_MAX_IOU = 0.10
    OVERSAMPLE_CATEGORY_ID = 3  # COCO Mouth_Ulcer category id
    OVERSAMPLE_FACTOR = 1.5

    # 梯度裁剪：拦住尖峰，但不要把每一步都裁。None 表示不裁剪。
    # 本项目实测（前 50 步，不裁剪）：中位数 154、p90 303、但出现过 4793 的尖峰。
    # 取 10 会裁掉 100% 的步（等于把学习率砍掉一个数量级）；取 200 只裁约 22%，
    # 拦住尾部与尖峰而放过中位数。换数据集/改 batch 后建议重新看日志里的 grad_norm。
    CLIP_GRAD_NORM = 200.0

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 70  # <--- 增加总轮次到70，给微调留足空间
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

    # # 对目标的自适应阈值（类别ID从0开始）
    # VAL_CLASS_THRESHOLDS = {
    #     0: 0.28,  # Caries
    #     1: 0.14,  # Calculus
    #     2: 0.20,  # Mouth_Ulcer
    #     3: 0.28,  # Tooth_Discoloration
    # }

    VAL_CLASS_THRESHOLDS = {
        0: 0.30,  # Caries
        1: 0.30,  # Calculus
        2: 0.30,  # Mouth_Ulcer
        3: 0.30,  # Tooth_Discoloration
    }


    VAL_CONF_THRESHOLD_DEFAULT = 0.3 # 不在 VAL_CLASS_THRESHOLDS 中的类别使用此默认阈值

    # Set to a sequence with NUM_CLASSES entries when class reweighting is needed.
    CLASS_WEIGHTS = [1.2, 1.3, 2.5, 1.1]
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)
    IMG_SIZE = 768
    NUM_WORKERS = 0
    SEED = 42


def evaluate_model(model, val_loader, device, use_class_thresholds=True, tqdm_file=None):
    """Evaluate predictions with COCO bbox metrics and print diagnostic counts."""
    model.eval()
    coco_results = []
    total_predictions = 0
    score_values = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", file=tqdm_file):
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

    # 设置日志文件，基于OUTPUT_DIR的名称
    log_filename = f"{Path(Config.OUTPUT_DIR).name}.log"
    log_path = output_dir / log_filename

    # 保存原始的stdout/stderr，供tqdm使用
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 创建日志记录器并重定向stdout和stderr
    stdout_logger = TeeLogger(log_path, mode='a')
    stderr_logger = TeeLoggerStderr(log_path, mode='a')
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    # 记录训练开始时间和配置信息
    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print("=" * 80)

    train_loader, val_loader = build_dataloaders(config=Config)
    project_root = Path(__file__).resolve().parent.parent
    num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    print(f"Device: {device}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset)}")
    print(f"Classes inferred from COCO: {num_classes}")
    print(f"Batch size: {Config.BATCH_SIZE}, image size: {Config.IMG_SIZE}")

    model = build_model(num_classes=num_classes, config=Config).to(device)
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

        # tqdm使用原始stdout，不写入日志文件
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", file=original_stdout):
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

        metrics = evaluate_model(model, val_loader, device, tqdm_file=original_stdout)
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

    # 训练结束，记录结束时间并恢复标准输出
    print("=" * 80)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best mAP: {best_map:.6f}")
    print("=" * 80)

    # 恢复原始的stdout和stderr
    sys.stdout = stdout_logger.terminal
    sys.stderr = stderr_logger.terminal
    stdout_logger.close()
    stderr_logger.close()


if __name__ == "__main__":
    train()
