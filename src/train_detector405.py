import os
import json
import re
import argparse
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
import random
import torchvision.transforms.functional as TF

# --- Monkey Patch: 修改 Faster R-CNN 分类头使用 Softmax Focal Loss ---

def fastrcnn_focal_loss(class_logits, box_regression, labels, regression_targets):
    # 原版获取包围盒回归损失 (回归损失保持原样)
    _, box_loss = orig_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    
    # ⚠️ 经过检查修复：Torchvision 传给 fastrcnn_loss 的 labels 是个 List[Tensor]，必须先拼接再算交叉熵
    labels_cat = torch.cat(labels, dim=0)
    
    # 针对分类头计算 Softmax Focal Loss
    ce_loss = F.cross_entropy(class_logits, labels_cat, reduction="none")
    pt = torch.exp(-ce_loss)
    gamma = 1.5
    alpha = 0.25
    # Focal loss 计算：降低易分类样本的权重
    focal_loss = (alpha * ((1 - pt) ** gamma) * ce_loss).mean()
    
    return focal_loss, box_loss

# 替换原本的 loss 计算函数
 # Native torchvision Faster R-CNN loss is intentionally used.
# ----------------------------------------------------------------------

from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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
    IMAGE_DIR = "../405/image_filtered"
    TRAIN_JSON = "coco/All_Diseases/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    VAL_JSON = "coco/All_Diseases/val.json"
    SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR = "res_checkpoints/multi_disease_405_expt"
    WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


    # 数据集配置
    DROP_EMPTY = False     # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 8
    EPOCHS = 50  # <--- 增加总轮次到35，给微调留足空间
    LR = 0.001
    BACKBONE_LR = 0.0001
    WARMUP_EPOCHS = 5
    UNFREEZE_LAST2_EPOCH = 6
    UNFREEZE_LAST4_EPOCH = 16
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
        transforms.append(ColorJitterDetection(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
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
    # Backbone blocks are unfrozen progressively inside the epoch loop.

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
        if "backbone.backbone" in name:
            backbone_params.append(param)
        elif param.requires_grad:
            head_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': Config.BACKBONE_LR},
        {'params': head_params, 'lr': Config.LR}             # 新初始化的 FPN 和检测头用正常学习率
    ], momentum=0.9, weight_decay=0.0005)
    def lr_lambda(epoch_index):
        if epoch_index < Config.WARMUP_EPOCHS:
            return float(epoch_index + 1) / Config.WARMUP_EPOCHS
        progress = (epoch_index - Config.WARMUP_EPOCHS) / max(1, Config.EPOCHS - Config.WARMUP_EPOCHS)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    
    # 记录历史最佳指标
    best_f1 = 0.0
    best_map = 0.0
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
                param_group['lr'] = Config.BACKBONE_LR
            else:
                param_group['lr'] = Config.LR        # Head
        
        # 也可以选择重置 scheduler，让衰减重新开始计算（可选）
        print(f"Resumed learning rates: Backbone={Config.BACKBONE_LR}, Head={Config.LR}")

        # 4) 恢复 best_* 初值（否则会从 0 开始导致“续训第一轮就覆盖 best”）
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get('metrics'), dict):
            m = checkpoint['metrics']
            best_f1 = float(m.get('f1', 0.0))
            best_map = float(m.get('coco', {}).get('map', 0.0))
            best_precision = float(m.get('precision', 0.0))
            best_recall = float(m.get('recall', 0.0))
        else:
            best_f1 = best_precision = best_recall = 0.0

        print(f"Checkpoint loaded. Resume from epoch={start_epoch}, best_f1={best_f1:.4f}, best_precision={best_precision:.4f}, best_recall={best_recall:.4f}")


    # 训练循环
    num_epochs = Config.EPOCHS
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if epoch == Config.UNFREEZE_LAST2_EPOCH:
            for param in backbone_model.blocks[-2:].parameters():
                param.requires_grad = True
            print("Stage 2: unfroze the last 2 DINOv3 blocks.")
        elif epoch == Config.UNFREEZE_LAST4_EPOCH:
            for param in backbone_model.blocks[-4:].parameters():
                param.requires_grad = True
            print("Stage 3: unfroze the last 4 DINOv3 blocks.")
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
            
            # 收集用于多阈值计算的所有结果
            all_val_results = []
            
            # 收集用于官方 COCO 评价标准的预测结果列表
            coco_results = []
            # 建立从本地从1开始的连续ID 回推至 真实Json文件的category_id 映射
            inv_category_map = {v: k for k, v in category_map.items()} if category_map else {}

            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                for output, target in zip(outputs, targets):
                    pred_boxes = output['boxes'].cpu()
                    pred_scores = output['scores'].cpu()
                    pred_labels = output['labels'].cpu()
                    
                    gt_boxes = target['boxes'].cpu()
                    gt_labels = target['labels'].cpu()
                    img_id = target['image_id'].item()
                    
                    # 保存用于多阈值计算的数据
                    all_val_results.append({
                        'pred_boxes': pred_boxes,
                        'pred_scores': pred_scores,
                        'pred_labels': pred_labels,
                        'gt_boxes': gt_boxes,
                        'gt_labels': gt_labels
                    })

                    # === 注入 COCOeval 所需结果 (在分数过滤前保存，以便COCO自己的多阈值算AR/mAP) ===
                    for p_box, p_score, p_label in zip(pred_boxes, pred_scores, pred_labels):
                        x1, y1, x2, y2 = p_box.tolist()
                        orig_cat_id = inv_category_map.get(p_label.item(), p_label.item())
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": orig_cat_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO格式为[x,y,w,h]
                            "score": float(p_score.item())
                        })

            # 计算多个阈值下的指标 (0.1 - 0.9)
            print(f"\n{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 50)
            
            multi_metrics = {}
            for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                tp, fp, fn = 0, 0, 0
                for img_results in all_val_results:
                    p_boxes = img_results['pred_boxes']
                    p_labels = img_results['pred_labels']
                    p_scores = img_results['pred_scores']
                    g_boxes = img_results['gt_boxes']
                    g_labels = img_results['gt_labels']
                    
                    keep = p_scores >= thr
                    cur_p_boxes = p_boxes[keep]
                    cur_p_labels = p_labels[keep]
                    
                    if len(g_boxes) == 0:
                        fp += len(cur_p_boxes)
                        continue
                    if len(cur_p_boxes) == 0:
                        fn += len(g_boxes)
                        continue
                        
                    ious = box_iou(cur_p_boxes, g_boxes)
                    matched_gt = set()
                    for p_idx in range(len(cur_p_boxes)):
                        same_class = g_labels == cur_p_labels[p_idx]
                        if not same_class.any():
                            fp += 1
                            continue
                        candidate_indices = torch.nonzero(same_class, as_tuple=False).flatten()
                        candidate_ious = ious[p_idx, candidate_indices]
                        max_iou, candidate_idx = candidate_ious.max(dim=0)
                        gt_idx = candidate_indices[candidate_idx]
                        if max_iou >= iou_threshold:
                            if gt_idx.item() not in matched_gt:
                                tp += 1
                                matched_gt.add(gt_idx.item())
                            else:
                                fp += 1
                        else:
                            fp += 1
                    fn += len(g_boxes) - len(matched_gt)
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                multi_metrics[thr] = (p, r, f)
                print(f"{thr:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f}")

            # 保持原有的 0.5 阈值用于保存逻辑
            precision, recall, f1 = multi_metrics[0.5]
            # print(f"\n[Custom Metric] IoU@{iou_threshold}, Score@{score_threshold} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # --- COCO 官方 API 评估 ---
            coco_metrics = {'map': 0.0, 'map50': 0.0, 'map75': 0.0, 'per_category_ap': {}}
            if len(coco_results) > 0:
                print("\n[COCOeval] 官方评测指标体系 (重点关注 AR指标):")
                try:
                    # 使用 loadRes 加载预测结果并在验证集实例上进行评测
                    coco_dt = val_dataset.coco.loadRes(coco_results)
                    coco_eval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
                    # 运行评测核心管线
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    coco_metrics['map'] = float(coco_eval.stats[0])
                    coco_metrics['map50'] = float(coco_eval.stats[1])
                    coco_metrics['map75'] = float(coco_eval.stats[2])
                    coco_precision = coco_eval.eval['precision']
                    for category_index, category_id in enumerate(coco_eval.params.catIds):
                        values = coco_precision[:, :, category_index, 0, -1]
                        values = values[values > -1]
                        coco_metrics['per_category_ap'][str(category_id)] = float(values.mean()) if values.size else 0.0
                except Exception as e:
                    print(f"Warning: COCOeval failed -> {e}")
                print("-" * 60)

        # --- 完整的模型保存逻辑 ---
        # 既然换成了参数量极小的 ViT-Base 版本，这里强烈建议保存完整的 state_dict
        # 避免仅保存部分参数导致在验证集推理或其他测试脚本中因为缺失 Buffer (如 BatchNorm) 和冻结的主干参数而引发血案
        save_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 关键：保存 optimizer/scheduler，才能无缝续训
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'metrics': {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'coco': coco_metrics,
            }
        }
        
        # 2. 始终保存一个 latest 版本，方便无缝继续
        torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "latest.pth"))

        # 3. 判断并覆盖三大最佳权重
        if coco_metrics['map'] > best_map:
            best_map = coco_metrics['map']
            torch.save(save_data, os.path.join(Config.OUTPUT_DIR, "best_map.pth"))
            print(f"*** New Best COCO mAP@[.5:.95]: {best_f1:.4f} ! Saved. ***")

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
