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
from PIL import Image
import numpy as np

# 导入自定义 backbone
from dinov3_backbone import Dinov3Backbone

# ========== 配置（与原始训练完全一致）==========
REPO_DIR = "/Users/mises/Python/dinov3"
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

# ========== 定义数据增强和数据集类（与原始训练相同）==========
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

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])

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

# ========== 加载数据 ==========
print("Loading training data...")
train_dataset = CocoDetectionDataset(IMAGE_DIR, TRAIN_JSON, get_transform(train=True))
print("Loading validation data...")
val_dataset = CocoDetectionDataset(IMAGE_DIR, VAL_JSON, get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ========== 构建模型（与原始训练完全一致）==========
print("Loading backbone...")
backbone_model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH)
backbone_model.eval()
for param in backbone_model.parameters():
    param.requires_grad = False

dinov3_backbone = Dinov3Backbone(backbone_model, embed_dim=384, out_channels=256)
print(f"Backbone out_channels: {dinov3_backbone.out_channels}")

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone=dinov3_backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    min_size=256,
    max_size=256
)

# ========== 加载已训练 20 轮的权重 ==========
checkpoint_path = os.path.join(OUTPUT_DIR, "fasterrcnn_epoch20.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
print(f"Loaded checkpoint from {checkpoint_path}")

model.to(device)

# ========== 设置优化器和学习率调度（重新初始化，使用较小学习率）==========
params = [p for p in model.parameters() if p.requires_grad]
# 使用较小的初始学习率继续微调
optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=0.0005)
# 学习率调度：每 3 个 epoch 衰减为原来的 0.1，这样 10 个 epoch 内逐步降低
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# ========== 继续训练 10 轮 ==========
num_epochs = 10
start_epoch = 21  # 从第 21 轮开始（原已训练 20 轮）
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    total_loss = 0
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

            # 更新进度条
            pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    # 验证：统计平均预测框数（可选，仅用于监控）
    model.eval()
    with torch.no_grad():
        total_boxes = 0
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred in predictions:
                total_boxes += len(pred['boxes'])
        avg_boxes = total_boxes / len(val_dataset) if len(val_dataset) > 0 else 0
        print(f"Average predicted boxes per image (val): {avg_boxes:.2f}")

    # 保存模型
    save_path = os.path.join(OUTPUT_DIR, f"fasterrcnn_epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 更新学习率
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate now: {current_lr:.2e}")

print("Continued training complete!")