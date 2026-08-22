"""
训练 DINOv3 (ViT-B/16) + CNN 分类头：牙周病图像分类（Periodontal_Disease）。

数据集: Sonata/Periodontal_Disease/<class>/ 下已分好类的图像，默认训练 3 类
        (gingival_diseases / non_periodontal_disease / periodontitis)，
        unknown 类默认剔除（可通过 --include_unknown 重新加入，变为 4 类）。
划分  : 按类别分层划分 train / val / test（默认 70% / 15% / 15%），固定随机种子，
        划分清单保存为 <output_dir>/splits.json，保证训练与推理使用同一划分。
模型  : DINOv3 vitb16 预训练权重 -> patch tokens 重排为特征图
        [B, 768, H/16, W/16] -> CNN 头(Conv+BN+ReLU -> GAP -> Dropout -> FC)。
微调  : 默认冻结 backbone，仅解冻最后 UNFREEZE_BLOCKS=6 个 Transformer Block
        （与检测任务 train_detector_sonata.py 的策略一致），backbone 用较小学习率。
设备  : 自动选择 cuda -> mps -> cpu，也可用 --device 手动指定；数据加载、损失、
        指标计算均与设备无关，兼容 CUDA / MPS / CPU。
权重  : 输出到 <repo>/weights/periodontal_classifier/（latest.pth / best_val_acc.pth）。
评估  : 测试集 accuracy、macro-F1、每类 precision/recall/F1，混淆矩阵 PNG。

用法示例:
    # 本地 (Mac MPS 或 CPU)
    python src/train_classifier_periodontal.py --epochs 30 --batch_size 16

    # CUDA 服务器 (Slurm 提交脚本见 submit_sonata_classifier.sh)
    python src/train_classifier_periodontal.py --data_dir /path/Sonata/Periodontal_Disease \
        --weights /path/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --dinov3_repo /path/dinov3-main --device cuda
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = "."  # 兼容检测任务写法（服务器上以 src 为工作目录时使用）

# ---------------------------------------------------------------------------
# 配置（所有项均可被命令行参数覆盖，与 train_detector_sonata.py 风格一致）
# ---------------------------------------------------------------------------
class Config:
    # --- 路径 ---
    DATA_DIR = PROJECT_ROOT / "Sonata" / "Periodontal_Disease"   # 分类数据集根目录
    WEIGHTS = PROJECT_ROOT / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    OUTPUT_DIR = PROJECT_ROOT / "weights" / "periodontal_classifier"  # 训练权重输出目录（用户要求放 weights 下）
    # DINOv3 官方代码目录（包含 dinov3/ 包的仓库根），脚本会自动探测常见位置
    DINOV3_REPO = "/Users/mises/Agent/dinov3-main"

    # --- 数据 ---
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15          # 剩余为 test
    INCLUDE_UNKNOWN = False   # 默认剔除 unknown 类（只训 3 类）
    IMG_SIZE = 224            # DINOv3 原生训练尺寸
    NUM_WORKERS = 0           # 本地建议 0；服务器可设 4

    # --- 训练超参数 ---
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.001                # 分类头学习率
    BACKBONE_LR = 0.0001      # DINOv3 解冻部分学习率（与检测任务一致）
    UNFREEZE_BLOCKS = 6       # 解冻最后 N 个 Transformer Block
    WEIGHT_DECAY = 1e-4
    CLASS_WEIGHTS = "none"    # "none"=等权, "auto"=按样本数逆频率, 或 "w1,w2,w3"
    DROPOUT = 0.2             # CNN 头分类层 dropout

    # --- 其它 ---
    DEVICE = "auto"           # auto: cuda -> mps -> cpu
    SEED = 42
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 + CNN head classifier (Periodontal_Disease)")
    parser.add_argument("--data_dir", default=str(Config.DATA_DIR), help="分类数据集根目录，含子目录每类一个文件夹")
    parser.add_argument("--weights", default=str(Config.WEIGHTS), help="DINOv3 预训练权重路径")
    parser.add_argument("--output_dir", default=str(Config.OUTPUT_DIR), help="训练权重/日志/混淆矩阵输出目录")
    parser.add_argument("--dinov3_repo", default=None,
                        help="DINOv3 官方代码仓库路径（含 dinov3/ 包），默认自动探测")
    parser.add_argument("--train_ratio", type=float, default=Config.TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=Config.VAL_RATIO)
    parser.add_argument("--include_unknown", action="store_true", default=Config.INCLUDE_UNKNOWN,
                        help="把 unknown/ 作为第 4 类一起训练（默认剔除）")
    parser.add_argument("--img_size", type=int, default=Config.IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR, help="分类头学习率")
    parser.add_argument("--backbone_lr", type=float, default=Config.BACKBONE_LR, help="DINOv3 解冻部分学习率")
    parser.add_argument("--unfreeze_blocks", type=int, default=Config.UNFREEZE_BLOCKS,
                        help="解冻的 DINOv3 Transformer Block 数量（0=完全冻结 backbone）")
    parser.add_argument("--class_weights", default=Config.CLASS_WEIGHTS,
                        help="'none' 等权 / 'auto' 逆频率 / 逗号分隔权重列表")
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--device", default=Config.DEVICE, help="auto / cuda / mps / cpu")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    return parser.parse_args()


def apply_args(args):
    Config.DATA_DIR = Path(args.data_dir)
    Config.WEIGHTS = Path(args.weights)
    Config.OUTPUT_DIR = Path(args.output_dir)
    if args.dinov3_repo:
        Config.DINOV3_REPO = args.dinov3_repo
    Config.TRAIN_RATIO = args.train_ratio
    Config.VAL_RATIO = args.val_ratio
    Config.INCLUDE_UNKNOWN = args.include_unknown
    Config.IMG_SIZE = args.img_size
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.BACKBONE_LR = args.backbone_lr
    Config.UNFREEZE_BLOCKS = args.unfreeze_blocks
    Config.CLASS_WEIGHTS = args.class_weights
    Config.NUM_WORKERS = args.num_workers
    Config.DEVICE = args.device
    Config.SEED = args.seed


# ---------------------------------------------------------------------------
# 设备选择：auto = cuda -> mps -> cpu
# ---------------------------------------------------------------------------
def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DINOv3 加载：优先从官方仓库 dinov3/hub/backbones 导入，失败则探测本地仓库
# ---------------------------------------------------------------------------
def _locate_dinov3_repo():
    candidates = [
        Config.DINOV3_REPO,
        "/Users/mises/Agent/dinov3-main",
        PROJECT_ROOT.parent / "dinov3-main",
        PROJECT_ROOT / "dinov3-main",
        Path.cwd().parent / "dinov3-main",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_dir() and (p / "dinov3").is_dir():
            return p
    return None


def load_dinov3_backbone(weights_path):
    """加载 DINOv3 ViT-B/16 backbone。

    Args:
        weights_path: 预训练权重路径；None 表示随机初始化（推理时使用，权重随后被 checkpoint 覆盖）。
    """
    try:
        import dinov3  # noqa: F401
    except ImportError:
        repo = _locate_dinov3_repo()
        if repo is None:
            raise RuntimeError(
                "无法导入 dinov3 包，请通过 --dinov3_repo 指定 DINOv3 官方代码仓库路径"
                "（包含 dinov3/ 包与 hubconf.py 的目录），或先 pip install -e <dinov3-main>"
            )
        sys.path.insert(0, str(repo))
        import dinov3  # noqa: F401

    from dinov3.hub.backbones import dinov3_vitb16

    if weights_path is None:
        # 随机初始化（不加载任何权重）
        return dinov3_vitb16(pretrained=False)

    # check_hash=False：使用本地权重文件，不联网校验
    backbone = dinov3_vitb16(pretrained=True, weights=str(weights_path), check_hash=False)
    return backbone


# ---------------------------------------------------------------------------
# CNN 分类头：patch tokens 特征图 [B, D, H/16, W/16] -> Conv -> GAP -> FC
# ---------------------------------------------------------------------------
class CNNClassificationHead(nn.Module):
    """基于 CNN 的分类头：1x1 降维 -> 3x3 卷积 -> 全局平均池化 -> FC。

    输入: [B, D, H/16, W/16]（DINOv3 patch tokens 重排后的特征图）
    输出: [B, num_classes] logits
    """

    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat_map):
        x = F.relu(self.bn1(self.conv1(feat_map)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, hidden_dim]
        x = self.dropout(x)
        return self.fc(x)


class DinoV3Classifier(nn.Module):
    """DINOv3 backbone + CNN 分类头。"""

    def __init__(self, backbone, embed_dim=768, num_classes=3, dropout=0.2, patch_size=16):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.head = CNNClassificationHead(embed_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        out = self.backbone.forward_features(x)
        if "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"]  # [B, N, D]
        else:
            patch_tokens = out["x_patchtokens"]
        B, N, D = patch_tokens.shape
        expected_N = h_feat * w_feat
        if N != expected_N:
            raise ValueError(
                f"Patch token count mismatch: expected {expected_N} (from {H}x{W}), got {N}. "
                f"Image size must be divisible by patch size {self.patch_size}."
            )
        # [B, N, D] -> [B, D, h_feat, w_feat]（先 contiguous，兼容 MPS）
        feat_map = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, D, h_feat, w_feat)
        return self.head(feat_map)


# ---------------------------------------------------------------------------
# 分层划分 train / val / test，保存 splits.json 供推理复用
# ---------------------------------------------------------------------------
def resolve_data_dir():
    """解析数据集目录：优先使用 Config.DATA_DIR，不存在时自动探测常见候选位置。

    服务器上数据可能放在项目外（../Sonata，与检测任务一致）或项目内（Sonata），
    两种布局都兼容；都找不到时给出清晰报错。
    """
    candidates = [Path(Config.DATA_DIR)]
    project_root = Path(__file__).resolve().parent.parent
    if not candidates[0].is_absolute():
        candidates.insert(0, project_root / Config.DATA_DIR)
    candidates += [
        project_root / "Sonata" / "Periodontal_Disease",        # 本地/项目内布局
        project_root.parent / "Sonata" / "Periodontal_Disease",  # 服务器布局（与检测任务 ../Sonata 一致）
    ]
    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_dir():
            if c != Path(Config.DATA_DIR).resolve():
                logging.info("数据集目录自动探测为: %s", c)
            return c
    raise FileNotFoundError(
        "找不到分类数据集 Periodontal_Disease。已尝试以下位置:\n  "
        + "\n  ".join(str(c) for c in seen)
        + "\n请通过 --data_dir 指定正确路径（或把数据放到 Sonata/Periodontal_Disease）"
    )


def build_splits(data_dir: Path, train_ratio: float, val_ratio: float, seed: int,
                 include_unknown: bool, output_dir: Path):
    classes = sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and (include_unknown or p.name != "unknown")
    )
    if not classes:
        raise ValueError(f"{data_dir} 下没有类别子目录")

    rng = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}
    class_stats = {}
    for cls in classes:
        files = sorted(
            f.name for f in (data_dir / cls).iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        # 每类至少 1 张训练、1 张测试（防极端小类）
        if n_train == 0 and n > 0:
            n_train, n_val = 1, max(0, n_val - 1)
        if n_test == 0 and n > 0:
            n_train, n_test = max(0, n_train - 1), 1
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        for split, part in (("train", train_files), ("val", val_files), ("test", test_files)):
            for f in part:
                splits[split][f] = cls
        class_stats[cls] = {"total": n, "train": len(train_files), "val": len(val_files), "test": len(test_files)}

    for split in splits:
        if not splits[split]:
            raise ValueError(f"划分后 {split} 集为空，请检查数据量")

    manifest = {
        "data_dir": str(data_dir.resolve()),
        "classes": classes,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio},
        "include_unknown": include_unknown,
        "class_stats": class_stats,
        "splits": splits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info("数据划分: %s", json.dumps(class_stats, ensure_ascii=False))
    return manifest


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def _default_loader(path: Path):
    with Image.open(path) as img:
        return img.convert("RGB")


class PeriodontalDataset(Dataset):
    """按 splits.json 中的划分读取 Periodontal_Disease 图像。"""

    def __init__(self, manifest, split, img_size=224, train=False):
        self.data_dir = Path(manifest["data_dir"])
        self.classes = manifest["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(f, self.class_to_idx[cls]) for f, cls in manifest["splits"][split].items()]
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(Config.DINO_MEAN, Config.DINO_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(Config.DINO_MEAN, Config.DINO_STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        try:
            img = _default_loader(self.data_dir / self.classes[label] / fname)
        except Exception as e:  # 损坏图片：返回同目录下一张（尽力而为）
            logging.warning("加载失败 %s: %s", fname, e)
            img = _default_loader(self.data_dir / self.classes[label] / self.samples[0][0])
        return self.transform(img), label


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------
def build_model(num_classes, pretrained_weights=True):
    """构建 DINOv3 + CNN 分类头。

    Args:
        num_classes: 类别数。
        pretrained_weights: True 时从 Config.WEIGHTS 加载 DINOv3 预训练权重（训练用）；
                            False 时随机初始化（推理时会被 checkpoint 覆盖，避免重复加载大权重）。
    """
    backbone = load_dinov3_backbone(Config.WEIGHTS) if pretrained_weights \
        else load_dinov3_backbone(None)
    # 冻结全部参数，再解冻最后 UNFREEZE_BLOCKS 个 Transformer Block
    for p in backbone.parameters():
        p.requires_grad = False
    if Config.UNFREEZE_BLOCKS > 0:
        for p in backbone.blocks[-Config.UNFREEZE_BLOCKS:].parameters():
            p.requires_grad = True
        logging.info("解冻 DINOv3 最后 %d 个 Transformer Block", Config.UNFREEZE_BLOCKS)
    else:
        logging.info("DINOv3 backbone 完全冻结，只训练 CNN 分类头")
    model = DinoV3Classifier(
        backbone,
        embed_dim=backbone.embed_dim,
        num_classes=num_classes,
        dropout=Config.DROPOUT,
    )
    return model


def compute_class_weights(manifest, num_classes):
    """按 --class_weights 计算 CrossEntropyLoss 的类别权重。"""
    if Config.CLASS_WEIGHTS == "none":
        return None
    counts = np.zeros(num_classes, dtype=np.float64)
    for f, cls in manifest["splits"]["train"].items():
        counts[manifest["classes"].index(cls)] += 1
    if Config.CLASS_WEIGHTS == "auto":
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (num_classes * counts)
        logging.info("自动类别权重(auto): %s", np.round(weights, 3).tolist())
        return torch.tensor(weights, dtype=torch.float32)
    try:
        weights = [float(x.strip()) for x in Config.CLASS_WEIGHTS.split(",")]
        assert len(weights) == num_classes, "类别权重数量必须等于类别数"
        return torch.tensor(weights, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"无法解析 --class_weights '{Config.CLASS_WEIGHTS}': {e}")


# ---------------------------------------------------------------------------
# 训练 / 验证 / 测试
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, class_names, split_name="val"):
    """在验证/测试集上评估，返回指标 dict 和混淆矩阵。"""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"{split_name} 评估", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    per_class = {
        name: {"precision": float(pi), "recall": float(ri), "f1": float(fi)}
        for name, pi, ri, fi in zip(class_names, p, r, f1)
    }
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "n_samples": len(all_labels),
    }
    logging.info(
        "%s: accuracy=%.4f macro_f1=%.4f per_class=%s",
        split_name, acc, macro_f1, json.dumps(per_class, ensure_ascii=False),
    )
    return metrics, cm, all_labels, all_preds


def save_confusion_matrix(cm, class_names, path, title="Confusion Matrix", normalized=False):
    """保存（归一化）混淆矩阵 PNG。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    if normalized:
        cm_disp = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_disp, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        fmt = lambda v: f"{v:.2f}"
        thresh = 0.5
    else:
        cm_disp = cm
        im = ax.imshow(cm_disp, interpolation="nearest", cmap=plt.cm.Blues)
        fmt = lambda v: format(v, "d")
        thresh = cm_disp.max() / 2.0 if cm_disp.max() > 0 else 0.5
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, fmt(cm_disp[i, j]),
                    ha="center", va="center",
                    color="white" if cm_disp[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(histories, output_dir):
    """保存训练曲线：train_loss / train_acc / val_acc / lr 随 epoch 变化。"""
    epochs = list(range(1, len(histories["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes
    ax1.plot(epochs, histories["train_loss"], "b-o", label="train_loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax1.grid(True)
    ax2.plot(epochs, histories["train_acc"], "g-o", label="train_acc")
    ax2.plot(epochs, histories["val_acc"], "r-s", label="val_acc")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()
    ax2.grid(True)
    ax3.plot(epochs, histories["lr"], "m-o")
    ax3.set(xlabel="Epoch", ylabel="LR", title="Learning Rate")
    ax3.grid(True)
    ax4.axis("off")
    fig.suptitle("Training Curves (Periodontal_Disease classifier)")
    fig.tight_layout()
    path = Path(output_dir) / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    with (Path(output_dir) / "training_curves.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    logging.info("训练曲线已保存: %s", path)


def _denormalize(tensor):
    """反归一化图像张量（用于可视化）。"""
    mean = torch.tensor(Config.DINO_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.DINO_STD).view(3, 1, 1)
    return tensor * std + mean


@torch.no_grad()
def visualize_test_samples(model, test_ds, device, class_names, output_path, per_class=3):
    """测试集可视化：每类选若干正确/错误样本拼接成网格图（绿=正确 红=错误）。"""
    model.eval()
    by_class = {c: {"correct": [], "wrong": []} for c in range(len(class_names))}
    for idx in range(len(test_ds)):
        img, label = test_ds[idx]
        logits = model(img.unsqueeze(0).to(device, non_blocking=True))
        probs = F.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        prob = float(probs[pred])
        bucket = "correct" if pred == label else "wrong"
        if len(by_class[label][bucket]) < per_class:
            fname = test_ds.samples[idx][0]
            by_class[label][bucket].append((img, label, pred, prob, fname))

    n_classes = len(class_names)
    n_cols = 2 * per_class
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(n_cols * 2.4, n_classes * 2.4))
    if n_classes == 1:
        axes = axes[None, :]
    for i in range(n_classes):
        for j, bucket in enumerate(("correct", "wrong")):
            samples = by_class[i][bucket]
            for k, (img, label, pred, prob, fname) in enumerate(samples):
                ax = axes[i, j * per_class + k]
                disp = _denormalize(img).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                ax.imshow(disp)
                color = "green" if bucket == "correct" else "red"
                ax.set_title(f"GT={class_names[label]}\nPred={class_names[pred]} ({prob:.2f})",
                             fontsize=8, color=color)
                ax.axis("off")
            for k in range(len(samples), per_class):  # 空位补白
                axes[i, j * per_class + k].axis("off")
    fig.suptitle("Test Samples: green=correct, red=wrong", fontsize=12)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("测试样本可视化已保存: %s", output_path)


def train():
    args = parse_args()
    apply_args(args)

    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    device = resolve_device(Config.DEVICE)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 日志：同时输出到终端与 <output_dir>/train.log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Config.OUTPUT_DIR / "train.log", encoding="utf-8"),
        ],
    )
    logging.info("Device: %s | torch %s", device, torch.__version__)
    logging.info("Data dir: %s", Config.DATA_DIR)

    # 1) 解析数据集目录（自动探测）并分层划分，保存 splits.json
    data_dir = resolve_data_dir()
    Config.DATA_DIR = data_dir
    manifest = build_splits(
        data_dir, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.SEED,
        Config.INCLUDE_UNKNOWN, Config.OUTPUT_DIR,
    )
    class_names = manifest["classes"]
    num_classes = len(class_names)
    logging.info("类别: %s (%d 类)", class_names, num_classes)

    # 2) 数据加载
    train_ds = PeriodontalDataset(manifest, "train", Config.IMG_SIZE, train=True)
    val_ds = PeriodontalDataset(manifest, "val", Config.IMG_SIZE, train=False)
    test_ds = PeriodontalDataset(manifest, "test", Config.IMG_SIZE, train=False)
    loader_kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    logging.info(
        "train=%d val=%d test=%d batch_size=%d",
        len(train_ds), len(val_ds), len(test_ds), Config.BATCH_SIZE,
    )

    # 3) 模型
    model = build_model(num_classes).to(device)
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.")]
    logging.info(
        "可训练参数: backbone=%d head=%d 总计=%d",
        sum(p.numel() for p in backbone_params),
        sum(p.numel() for p in head_params),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": Config.BACKBONE_LR},
        {"params": head_params, "lr": Config.LR},
    ], weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    class_weights = compute_class_weights(manifest, num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    # 4) 训练循环
    histories = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        train_acc = total_correct / max(1, total)
        epoch_loss = total_loss / max(1, total)
        logging.info(
            "Epoch %d: loss=%.4f train_acc=%.4f lr=%s",
            epoch, epoch_loss, train_acc,
            [g["lr"] for g in optimizer.param_groups],
        )

        # 验证（每 epoch）
        val_metrics, _, _, _ = evaluate(model, val_loader, device, class_names, "val")
        histories["train_loss"].append(epoch_loss)
        histories["train_acc"].append(train_acc)
        histories["val_acc"].append(val_metrics["accuracy"])
        histories["lr"].append(optimizer.param_groups[0]["lr"])
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "class_names": class_names,
            "img_size": Config.IMG_SIZE,
            "config": vars(args),
        }
        torch.save(checkpoint, Config.OUTPUT_DIR / "latest.pth")
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(checkpoint, Config.OUTPUT_DIR / "best_val_acc.pth")
            logging.info("新的最佳验证准确率: %.4f", best_val_acc)

    # 5) 训练曲线
    save_training_curves(histories, Config.OUTPUT_DIR)

    # 6) 测试集最终评估（加载 best 权重）
    logging.info("== 测试集评估（best_val_acc.pth） ==")
    best_ckpt = torch.load(Config.OUTPUT_DIR / "best_val_acc.pth", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics, cm, _, _ = evaluate(model, test_loader, device, class_names, "test")
    save_confusion_matrix(
        cm, class_names, Config.OUTPUT_DIR / "test_confusion_matrix.png",
        title="Test Confusion Matrix",
    )
    save_confusion_matrix(
        cm, class_names, Config.OUTPUT_DIR / "test_confusion_matrix_normalized.png",
        title="Test Confusion Matrix (normalized)", normalized=True,
    )
    visualize_test_samples(
        model, test_ds, device, class_names,
        Config.OUTPUT_DIR / "test_samples_visualization.png",
    )

    summary = {
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "test": test_metrics,
    }
    with (Config.OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("训练完成，测试集指标: %s", json.dumps(test_metrics, ensure_ascii=False))
    logging.info("输出目录: %s", Config.OUTPUT_DIR)


if __name__ == "__main__":
    train()
