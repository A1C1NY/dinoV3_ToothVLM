import os
import json
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
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
    IMAGE_DIR = r"..\Dataset\\Caries\\image"
    TRAIN_JSON = r"coco\\Caries\\train.json"
    VAL_JSON = r"coco\\Caries\\val.json"
    WEIGHTS = r"pretrained_checkpoints\\dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    OUTPUT_DIR = r"res_checkpoints\\caries_expt"

    # 数据集配置
    SINGLE_CAT_ID = 1      # 单疾病训练：指定只保留哪个原始 category_id，通常填 1 即可。若为多类请填 None
    DROP_EMPTY = True      # 是否丢弃没有标注的图片

    # 训练超参数
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 0.005
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 继续训练 (可选)
    RESUME_CHECKPOINT = None  # 填写 .pth 文件路径以继续训练，例如 r"..."
    START_EPOCH = 1           # 继续训练时的起始 epoch

    # 模型参数
    MIN_SIZE = 256
    MAX_SIZE = 256

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
        img_path = os.path.join(self.img_folder, img_info['file_name'])
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
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return DetectionTransform(T.Compose(transforms))


def main():
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
    backbone_model = torch.hub.load(Config.REPO_DIR, 'dinov3_vits16plus', source='local', weights=Config.WEIGHTS)
    backbone_model.eval()
    for param in backbone_model.parameters():
        param.requires_grad = False

    dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=384, out_channels=256)
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
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

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

        # 验证（计算验证 loss）
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_count = 0
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # FasterRCNN 在 eval 模式下返回的是预测结果 (list of dict) 而不是 loss
                # 如果我们需要评估 loss，可以临时切回 train 模式
                # 注意：这会导致 Dropout / BatchNorm 的行为改变（如果 backbone 中有），
                # 但一般用于仅需快速查看 loss 趋势时使用。更标准的做法是使用 mAP 等指标。
                
                model.train()  # 临时切回 train 以获取 loss
                loss_dict = model(images, targets)
                model.eval()   # 切回 eval 保证下一个 step 不用梯度
                
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_count += 1
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            print(f"Validation loss: {avg_val_loss:.4f}")

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