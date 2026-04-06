import os
import json
import re
import argparse
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
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
    IMAGE_DIR = "../Dataset/Caries/image"
    TRAIN_JSON = "coco/Caries/train.json"
    VAL_JSON = "coco/Caries/val.json"
    SINGLE_CAT_ID = 1      # 指定只保留哪个原始 category_id
    OUTPUT_DIR = "res_checkpoints/caries_expt" 
    
    # # --- 选项 B：所有疾病混合训练 (All Diseases) ---
    # IMAGE_DIR = "../Dataset"
    # TRAIN_JSON = "coco/All_Diseases/train.json"  # 注意：目前 prepare_data 混在了一起，用于此示例
    # VAL_JSON = "coco/All_Diseases/val.json"
    # SINGLE_CAT_ID = None   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    # OUTPUT_DIR = "res_checkpoints/multi_disease_expt"
    WEIGHTS = "pretrained_checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"


    # 数据集配置
    DROP_EMPTY = True      # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 0.005
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 继续训练 (可选)
    RESUME_CHECKPOINT = None  # 填写 .pth 文件路径以继续训练，例如 r"..."
    START_EPOCH = 1           # 继续训练时的起始 epoch

    # 验证与评估参数
    IOU_THRESHOLD = 0.5       # 用于评估时判断正样本的 IoU 阈值
    SCORE_THRESHOLD = 0.5     # 用于过滤低置信度预测的阈值

    # 模型参数
    MIN_SIZE = 512
    MAX_SIZE = 512

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


class DetectionTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        img = self.transforms(img)
        return img, target


def get_transform(train):
    transforms = []
    if train:
        # transforms.append(T.RandomHorizontalFlip(0.5)) 不知是哪个神经AI写的，但牙齿图像水平翻转可能会导致标签和病灶位置不匹配。
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    transforms.append(T.ToTensor())
    # transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return DetectionTransform(T.Compose(transforms))


def main():
    # 增加命令行参数：允许通过附加 --continue 自动恢复训练
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_train', '--resume', dest='resume', action='store_true', help="自动在此文件夹(OUTPUT_DIR)中寻找最新的 .pth 权重继续训练")
    args, unknown = parser.parse_known_args()

    # 如果指定了继续训练，在 OUTPUT_DIR 自动寻找最新的 checkpoint
    # (也兼容传入 --continue_train=True 等未知参数)
    if args.resume or any(arg.startswith('--continue') for arg in unknown):
        if os.path.exists(Config.OUTPUT_DIR):
            checkpoints = [f for f in os.listdir(Config.OUTPUT_DIR) if f.endswith('.pth')]
            if checkpoints:
                def get_epoch(filename):
                    match = re.search(r'epoch(\d+)', filename)
                    return int(match.group(1)) if match else -1
                
                latest_ckpt = max(checkpoints, key=get_epoch)
                Config.RESUME_CHECKPOINT = os.path.join(Config.OUTPUT_DIR, latest_ckpt)
                Config.START_EPOCH = get_epoch(latest_ckpt) + 1
                print(f"==================================================")
                print(f"Auto-resuming enabled!")
                print(f"Found latest checkpoint: {Config.RESUME_CHECKPOINT}")
                print(f"Will resume from epoch: {Config.START_EPOCH}")
                print(f"==================================================")
            else:
                print(f"Warning: --continue passed, but no .pth files found in {Config.OUTPUT_DIR}. Starting from scratch.")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 构建 category_map
    category_map = build_category_map(Config.TRAIN_JSON, single_cat_id=Config.SINGLE_CAT_ID)
    max_mapped = max(category_map.values()) if category_map else 0
    num_classes = max_mapped + 1 if max_mapped >= 1 else 2
    print(f"Category map: {category_map}")
    print(f"num_classes set to {num_classes} (包括背景)")

    # 加载 DINOv3 骨干
    backbone_model = torch.hub.load(Config.REPO_DIR, 'dinov3_vit7b16', source='local', weights=Config.WEIGHTS)
    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False

    # 对于 ViT Backbone，通常最后几层包含更多语义信息，解冻最后两层 Transformer Block 以适应检测任务
    for param in backbone_model.blocks[-4:].parameters(): # 解冻最后4层 Transformer Block
        param.requires_grad = True

    # 自动获取当前模型的 embed dim 
    # vits: 384, vitb: 768, vitl: 1024, vitg: 1536
    embed_dim = getattr(backbone_model, 'embed_dim', 768)
    print(f"Detected backbone embed_dim: {embed_dim}")

    dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=embed_dim, out_channels=256)
    print(f"Backbone out_channels: {dinov3_backbone.out_channels}")

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

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

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=Config.LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.9)

    # 可选 resume
    start_epoch = Config.START_EPOCH
    if Config.RESUME_CHECKPOINT:
        if not os.path.exists(Config.RESUME_CHECKPOINT):
            raise FileNotFoundError(f"Checkpoint not found: {Config.RESUME_CHECKPOINT}")
        print(f"Loading checkpoint {Config.RESUME_CHECKPOINT}")
        state_dict = torch.load(Config.RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded")

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

            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                for output, target in zip(outputs, targets):
                    pred_boxes = output['boxes'].cpu()
                    pred_scores = output['scores'].cpu()
                    pred_labels = output['labels'].cpu()
                    
                    gt_boxes = target['boxes'].cpu()
                    gt_labels = target['labels'].cpu()
                    
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

        # 保存模型
        save_path = os.path.join(Config.OUTPUT_DIR, f"fasterrcnn_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        # 保存元信息，便于流水线管理和后续兼容（记录 category_map 与运行参数）
        meta = {
            "train_json": Config.TRAIN_JSON,
            "val_json": Config.VAL_JSON,
            "category_map": category_map
        }
        try:
            with open(save_path + ".meta.json", 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to write meta file: {e}")

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate now: {current_lr:.2e}")

    print("Training complete")


if __name__ == '__main__':
    main()