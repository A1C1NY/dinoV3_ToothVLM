import os
import json
import re
import argparse
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as roi_heads
import random
import torchvision.transforms.functional as TF

# --- Monkey Patch: 修改 Faster R-CNN 分类头使用 Softmax Focal Loss ---
orig_fastrcnn_loss = roi_heads.fastrcnn_loss

def fastrcnn_focal_loss(class_logits, box_regression, labels, regression_targets):
    # 原版获取包围盒回归损失 (回归损失保持原样)
    _, box_loss = orig_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    
    # Torchvision 传给 fastrcnn_loss 的 labels 是个 List[Tensor]，必须先拼接再算交叉熵
    labels_cat = torch.cat(labels, dim=0)
    
    # 针对分类头计算 Softmax Focal Loss
    ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none")
    pt = torch.exp(-ce_loss)
    gamma = 2.0
    alpha = 0.25
    # Focal loss 计算：降低易分类样本的权重
    focal_loss = (alpha * ((1 - pt) ** gamma) * ce_loss).mean()
    
    return focal_loss, box_loss

# 替换原本的 loss 计算函数
roi_heads.fastrcnn_loss = fastrcnn_focal_loss
# ----------------------------------------------------------------------

from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval  # <--- 新增 mAP 评估库
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# 导入自定义 backbone
from dinov3_backbone import Dinov3Backbone

# ========== 配置区（直接在此修改）==========
class Config:
        # 路径配置
    REPO_DIR = "."

    # 路径配置（可以通过注释快速切换单疾病/多疾病）
    
    # --- 选项 A：单疾病训练 (例如 Caries) ---
    # IMAGE_DIR = "../Dataset/Caries/image"
    # TRAIN_JSON = "coco/Caries/train.json"
    # VAL_JSON = "coco/Caries/val.json"
    # SINGLE_CAT_ID = 1      # 指定只保留哪个原始 category_id
    # OUTPUT_DIR = "res_checkpoints/caries_expt" 
    
    # # --- 选项 B：所有疾病混合训练 (All Diseases) ---
    IMAGE_DIR = "../Dataset"
    TRAIN_JSON = "coco/All_Diseases/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    VAL_JSON = "coco/All_Diseases/val.json"
    SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR = "res_checkpoints/multi_disease_expt"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


    # 数据集配置
    DROP_EMPTY = True      # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 40  # <--- 增加微调轮次
    LR = 2e-3    # <--- 略微调低初始学习率，或者更激进一点 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 继续训练 (可选)
    RESUME_CHECKPOINT = None  # 填写 .pth 文件路径以继续训练，例如 r"..."
    START_EPOCH = 1           # 继续训练时的起始 epoch

    # 验证与评估参数
    IOU_THRESHOLD = 0.5       # 用于评估时判断正样本的 IoU 阈值
    SCORE_THRESHOLD = 0.3     # <--- 降低置信度阈值，通常 0.5 太高了，0.3 更适合评估

    # 模型参数
    MIN_SIZE = 1200
    MAX_SIZE = 1200

def build_category_map(train_json, single_cat_id=None):
    coco = COCO(train_json)
    # 优先使用实际出现在 annotations 中的 category_id（防止 categories 字段包含全部类别但未使用）
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids) if ann_ids else []
    used_cat_ids = sorted({a['category_id'] for a in anns})
    cat_ids = used_cat_ids if used_cat_ids else coco.getCatIds()
    if single_cat_id is not None:
        if single_cat_id not in cat_ids:
            print(f"Warning: single_cat_id {single_cat_id} not found in {train_json}. Available cat ids: {cat_ids}")
        # 只保留指定类别并映射为 1
        return {single_cat_id: 1}
    # 多类别：建立从原始 id 到连续 id 的映射（1..K）
    cat_ids_sorted = sorted(cat_ids)
    category_map = {old: i + 1 for i, old in enumerate(cat_ids_sorted)}
    return category_map


class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, category_map=None, drop_empty=False):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.category_map = category_map
        self.drop_empty = drop_empty
        if self.drop_empty:
            # 过滤掉无标注图像
            filtered = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                if self.category_map is not None:
                    anns = [a for a in anns if a['category_id'] in self.category_map]
                if len(anns) > 0:
                    filtered.append(img_id)
            self.ids = filtered

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 若提供 category_map，则过滤并映射 category_id
        if self.category_map is not None:
            anns = [a for a in anns if a['category_id'] in self.category_map]

        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_folder, file_name)

        # 兼容多疾病文件夹结构：如果在根目录找不到图片，则遍历子目录寻找 (如 Dataset/Caries/image/...)
        if not os.path.exists(img_path):
            for root, dirs, files in os.walk(self.img_folder):
                if file_name in files:
                    img_path = os.path.join(root, file_name)
                    break

        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            orig_cat = ann['category_id']
            if self.category_map is not None:
                new_cat = self.category_map[orig_cat]
            else:
                new_cat = orig_cat
            labels.append(new_cat)
            areas.append(ann.get('area', w * h))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target



class ComposeDetection:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class RandomHorizontalFlipDetection:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            width, height = img.size
            img = TF.hflip(img)
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                # 水平翻转：xmin 和 xmax 互换并用 width 减
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target

class RandomVerticalFlipDetection:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            width, height = img.size
            img = TF.vflip(img)
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                # 垂直翻转：ymin 和 ymax 互换并用 height 减
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return img, target


class ColorJitterDetection:
    def __init__(self, *args, **kwargs):
        self.transform = T.ColorJitter(*args, **kwargs)

    def __call__(self, img, target):
        img = self.transform(img)
        return img, target

class ToTensorDetection:
    def __call__(self, img, target):
        img = TF.to_tensor(img)
        return img, target

def get_transform(train):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlipDetection(0.5))
        transforms.append(RandomVerticalFlipDetection(0.5))
        transforms.append(ColorJitterDetection(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    transforms.append(ToTensorDetection())
    return ComposeDetection(transforms)


def main():
    # 增加命令行参数：允许通过附加 --continue 自动恢复训练
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_train', '--resume', dest='resume', action='store_true', help="自动在此文件夹(OUTPUT_DIR)中寻找最新的 .pth 权重继续训练")
    args, unknown = parser.parse_known_args()

    # 如果指定了继续训练，在 OUTPUT_DIR 自动寻找最新的 checkpoint
    if args.resume or any(arg.startswith('--continue') for arg in unknown):
        if os.path.exists(Config.OUTPUT_DIR):
            target_ckpt = None
            start_epoch = 0
            
            # 优先级1：新版的 best_f1.pth 或 latest.pth
            best_f1_path = os.path.join(Config.OUTPUT_DIR, "best_f1.pth")
            latest_path = os.path.join(Config.OUTPUT_DIR, "latest.pth")
            
            if os.path.exists(best_f1_path):
                target_ckpt = best_f1_path
            elif os.path.exists(latest_path):
                target_ckpt = latest_path
            else:
                # 优先级2：老版的 fasterrcnn_epoch{X}.pth (注意过滤掉 .json)
                ckpts = [f for f in os.listdir(Config.OUTPUT_DIR) if re.match(r'fasterrcnn_epoch(\d+)\.pth$', f)]
                if ckpts:
                    # 找到数字最大的那个文件
                    ckpts.sort(key=lambda x: int(re.search(r'fasterrcnn_epoch(\d+)\.pth', x).group(1)))
                    latest_old_ckpt = ckpts[-1]
                    target_ckpt = os.path.join(Config.OUTPUT_DIR, latest_old_ckpt)
                    start_epoch = int(re.search(r'fasterrcnn_epoch(\d+)\.pth', latest_old_ckpt).group(1))

            if target_ckpt:
                try:
                    checkpoint = torch.load(target_ckpt, map_location='cpu', weights_only=False)
                    Config.RESUME_CHECKPOINT = target_ckpt
                    
                    # 如果是新版封装了字典的保存格式，提取里面记录的 epoch
                    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                        Config.START_EPOCH = checkpoint['epoch'] + 1
                    else:
                        # 否则使用老版从文件名提取出的 epoch
                        Config.START_EPOCH = start_epoch + 1
                        
                    print(f"==================================================")
                    print(f"Auto-resuming enabled!")
                    print(f"Found checkpoint: {Config.RESUME_CHECKPOINT}")
                    print(f"Will resume from epoch: {Config.START_EPOCH}")
                    print(f"==================================================")
                except Exception as e:
                    print(f"Error reading checkpoint: {e}")
            else:
                print(f"Warning: --continue passed, but no valid checkpoint found in {Config.OUTPUT_DIR}. Starting from scratch.")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 构建 category_map
    category_map = build_category_map(Config.TRAIN_JSON, single_cat_id=Config.SINGLE_CAT_ID)
    max_mapped = max(category_map.values()) if category_map else 0
    num_classes = max_mapped + 1 if max_mapped >= 1 else 2
    print(f"Category map: {category_map}")
    print(f"num_classes set to {num_classes} (包括背景)")

    # 加载 DINOv3 骨干 (Base 版本，参数量小很多)
    backbone_model = torch.hub.load(Config.REPO_DIR, 'dinov3_vitb16', source='local', weights=Config.WEIGHTS)
    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False

    # 对于 ViT Backbone，解冻最后 4 层 Transformer Block 以适应检测任务 (Base 模型解冻的负担很小，提效极大)
    for param in backbone_model.blocks[-4:].parameters(): 
        param.requires_grad = True

    # 自动获取当前模型的 embed dim 
    # vits: 384, vitb: 768, vitl: 1024, vitg: 1536
    embed_dim = getattr(backbone_model, 'embed_dim', 768)
    print(f"Detected backbone embed_dim: {embed_dim}")

    dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=embed_dim, out_channels=256)
    print(f"Backbone out_channels: {dinov3_backbone.out_channels}")

    # 更精细的 Anchors: 每个特征层加上 3 种尺度 (2^0, 2^(1/3), 2^(2/3))，以及 5种长宽比
    anchor_generator = AnchorGenerator(
        sizes=(
            (32, 40, 50), 
            (64, 80, 101), 
            (128, 161, 203), 
            (256, 322, 406)
        ), 
        aspect_ratios=((0.5, 0.75, 1.0, 1.33, 2.0),) * 4
    )
    # Note: dinov3_backbone returns dict keys '0', '1', '2', '3'
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=dinov3_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE
    )
    model.to(device)

    # 数据集与 DataLoader
    train_dataset = CocoDetectionDataset(Config.IMAGE_DIR, Config.TRAIN_JSON, get_transform(train=True), category_map=category_map, drop_empty=Config.DROP_EMPTY)
    val_dataset = CocoDetectionDataset(Config.IMAGE_DIR, Config.VAL_JSON, get_transform(train=False), category_map=category_map, drop_empty=Config.DROP_EMPTY)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # 优化器：对预训练的主干网络和新初始化的头部使用不同的学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': Config.LR * 0.1},  # 主干网络用较小的学习率防止破坏预训练权重
        {'params': head_params, 'lr': Config.LR}             # 新初始化的 FPN 和检测头用正常学习率
    ], momentum=0.9, weight_decay=0.0001) # 略微降低 weight decay
    
    # 使用余弦退火学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    
    # 记录历史最佳指标
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    # 可选 resume
    start_epoch = Config.START_EPOCH
    if Config.RESUME_CHECKPOINT:
        if not os.path.exists(Config.RESUME_CHECKPOINT):
            raise FileNotFoundError(f"Checkpoint not found: {Config.RESUME_CHECKPOINT}")
        print(f"Loading checkpoint {Config.RESUME_CHECKPOINT}")
        checkpoint = torch.load(Config.RESUME_CHECKPOINT, map_location=device, weights_only=False)

        # 1) 恢复模型（兼容新/旧格式）
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict_to_load, strict=False)

        # 2) 若 checkpoint 里有 epoch，则以它为准
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1

        # 3) 若 checkpoint 里有 optimizer/scheduler 状态，则恢复（无则跳过）
        if isinstance(checkpoint, dict) and checkpoint.get('optimizer_state_dict') is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Warning: failed to load optimizer_state_dict, will continue with fresh optimizer state. Reason: {e}")

        if isinstance(checkpoint, dict) and checkpoint.get('lr_scheduler_state_dict') is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                print("LR scheduler state restored.")
            except Exception as e:
                print(f"Warning: failed to load lr_scheduler_state_dict, will continue with fresh scheduler state. Reason: {e}")

        # 强制将优化器中的学习率重置为 Config 中新设置的 LR
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                param_group['lr'] = Config.LR * 0.1  # Backbone
            else:
                param_group['lr'] = Config.LR        # Head
        
        # 使用余弦退火学习率
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
        print(f"Forced learning rate update to: Backbone={Config.LR * 0.1}, Head={Config.LR}")

        # 4) 恢复 best_* 初值（否则会从 0 开始导致“续训第一轮就覆盖 best”）
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get('metrics'), dict):
            m = checkpoint['metrics']
            best_f1 = float(m.get('f1', 0.0))
            best_precision = float(m.get('precision', 0.0))
            best_recall = float(m.get('recall', 0.0))
        else:
            best_f1 = best_precision = best_recall = 0.0

        print(f"Checkpoint loaded. Resume from epoch={start_epoch}, best_f1={best_f1:.4f}, best_precision={best_precision:.4f}, best_recall={best_recall:.4f}")


    # 训练循环
    num_epochs = Config.EPOCHS
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1}") as pbar:
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                pbar.set_postfix(loss=losses.item())

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # 验证（使用 IoU 阈值评估 Precision, Recall, F1）
        model.eval()
        with torch.no_grad():
            iou_threshold = Config.IOU_THRESHOLD
            score_threshold = Config.SCORE_THRESHOLD
            
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            # 用于收集 COCO 格式的预测结果
            coco_results = []
            # 反向映射 category_id (模型输出的连续 id -> 原始标注 id)
            inv_category_map = {v: k for k, v in category_map.items()} if category_map else {}

            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                for output, target in zip(outputs, targets):
                    img_id = target['image_id'].item()
                    pred_boxes = output['boxes'].cpu()
                    pred_scores = output['scores'].cpu()
                    pred_labels = output['labels'].cpu()
                    
                    gt_boxes = target['boxes'].cpu()
                    gt_labels = target['labels'].cpu()
                    
                    # --- 收集用于计算标准 mAP 的数据 (不走自定义置信度阈值过滤，由 coco_eval 自行处理) ---
                    for p_box, p_score, p_label in zip(pred_boxes, pred_scores, pred_labels):
                        x1, y1, x2, y2 = p_box.tolist()
                        w, h = x2 - x1, y2 - y1
                        orig_cat_id = inv_category_map.get(p_label.item(), p_label.item())
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": orig_cat_id,
                            "bbox": [x1, y1, w, h],
                            "score": p_score.item()
                        })
                    
                    # --- 以下是你原有的 F1 计算逻辑 ---
                    # 过滤低置信度预测
                    keep = pred_scores >= score_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_labels = pred_labels[keep]
                    
                    if len(gt_boxes) == 0:
                        false_positives += len(pred_boxes)
                        continue
                        
                    if len(pred_boxes) == 0:
                        false_negatives += len(gt_boxes)
                        continue
                        
                    # 计算 IoU 矩阵 [N_pred, M_gt]
                    ious = box_iou(pred_boxes, gt_boxes)
                    
                    # 贪婪匹配机制
                    matched_gt = set()
                    for p_idx in range(len(pred_boxes)):
                        max_iou, gt_idx = ious[p_idx].max(dim=0)
                        if max_iou >= iou_threshold and pred_labels[p_idx] == gt_labels[gt_idx]:
                            if gt_idx.item() not in matched_gt:
                                true_positives += 1
                                matched_gt.add(gt_idx.item())
                            else:
                                false_positives += 1 # 已经被其他更高置信度的预测框匹配
                        else:
                            false_positives += 1
                            
                    false_negatives += len(gt_boxes) - len(matched_gt)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            print(f"Validation Metrics (IoU@{iou_threshold}, Score@{score_threshold}) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # --- 新增的 标准 COCO mAP 评估 ---
            map_05095 = 0.0
            map_05 = 0.0
            if len(coco_results) > 0:
                coco_dt = val_dataset.coco.loadRes(coco_results)
                coco_eval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                map_05095 = coco_eval.stats[0] # mAP @ IoU=0.50:0.95
                map_05 = coco_eval.stats[1]    # mAP @ IoU=0.50

        # --- 完整的模型保存逻辑 ---
        save_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'metrics': {'f1': f1, 'precision': precision, 'recall': recall, 'mAP_0.5': map_05, 'mAP_0.5:0.95': map_05095}
        }
        
        # 2. 始终保存一个 latest 版本，方便无缝继续
        torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "latest.pth"))

        # 3. 判断并覆盖三大最佳权重
        if f1 > best_f1:
            best_f1 = f1
            torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "best_f1.pth"))
            print(f"*** New Best F1: {best_f1:.4f} ! Saved. ***")
            
        if precision > best_precision:
            best_precision = precision
            torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "best_precision.pth"))
            
        if recall > best_recall:
            best_recall = recall
            torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "best_recall.pth"))

        # 保存元信息，便于流水线管理和后续兼容
        meta = {
            "train_json": Config.TRAIN_JSON,
            "val_json": Config.VAL_JSON,
            "category_map": category_map
        }
        try:
            with open(os.path.join(Config.OUTPUT_DIR, "latest.meta.json"), 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to write meta file: {e}")

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate now: {current_lr:.2e}")

    print("Training complete")


if __name__ == '__main__':
    main()