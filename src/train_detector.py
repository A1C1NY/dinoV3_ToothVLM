import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 导入自定义 backbone
from dinov3_backbone import Dinov3Backbone

# 配置
REPO_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = "/Users/mises/Python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
IMAGE_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/image"
ANNOTATION_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/coco_annotations"
TRAIN_JSON = os.path.join(ANNOTATION_DIR, "train.json")
VAL_JSON = os.path.join(ANNOTATION_DIR, "val.json")
OUTPUT_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/detector_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 DINOv3 骨干
backbone_model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH)
backbone_model.eval()
for param in backbone_model.parameters():
    param.requires_grad = False

# 构建自定义骨干包装器
dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=384, out_channels=256)
print(f"Backbone out_channels: {dinov3_backbone.out_channels}")

# 创建 RPN 和 Faster R-CNN 头部
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone=dinov3_backbone,
    num_classes=2,  # 背景 + 龋齿
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    min_size=256,
    max_size=256
)
model.to(device)

# 定义数据增强（针对目标检测）
class DetectionTransform:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, target):
        # 应用图像变换
        img = self.transforms(img)
        return img, target

def get_transform(train):
    transforms = []
    if train:
        # 训练时的数据增强
        transforms.append(T.RandomHorizontalFlip(0.5))
        # 随机调整亮度、对比度等
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # 转换为 Tensor 并归一化
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return DetectionTransform(T.Compose(transforms))

# 自定义数据集类
class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # 提取边界框和标签
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # 转换为 [xmin, ymin, xmax, ymax]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
        
        if len(boxes) == 0:
            # 如果没有标注，返回空列表
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

# 加载数据集
print("Loading training data...")
train_dataset = CocoDetectionDataset(IMAGE_DIR, TRAIN_JSON, get_transform(train=True))
print("Loading validation data...")
val_dataset = CocoDetectionDataset(IMAGE_DIR, VAL_JSON, get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 优化器（只训练检测头参数）
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_count = 0
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            val_count += 1
        avg_val_loss = val_loss / val_count if val_count > 0 else 0
        print(f"Validation loss: {avg_val_loss:.4f}")

    lr_scheduler.step()
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"fasterrcnn_epoch{epoch+1}.pth"))
    print(f"Model saved to {OUTPUT_DIR}/fasterrcnn_epoch{epoch+1}.pth")

print("Training complete!")