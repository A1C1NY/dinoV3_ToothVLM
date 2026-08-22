"""
DINOv3 + CNN 分类头推理脚本（牙周病图像分类）。

支持两种模式:
1) 图片预测:   --image <文件或目录>  -> 对每张图输出类别与概率，保存 CSV
2) 测试集评估: --eval_test            -> 使用训练时保存的 splits.json 在测试集上评估，
                                        输出 accuracy / macro-F1 / 每类 P/R/F1 与混淆矩阵 PNG

用法示例:
    # 预测单张图片
    python src/infer_classifier_periodontal.py --checkpoint weights/periodontal_classifier/best_val_acc.pth \
        --image Sonata/Periodontal_Disease/periodontitis/Dental_Dieases_Calculus_100.jpg

    # 预测整个文件夹
    python src/infer_classifier_periodontal.py --checkpoint weights/periodontal_classifier/best_val_acc.pth \
        --image /path/to/folder

    # 测试集评估（复用训练时的划分）
    python src/infer_classifier_periodontal.py --checkpoint weights/periodontal_classifier/best_val_acc.pth \
        --eval_test --output_dir weights/periodontal_classifier/eval_test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier_periodontal import (  # noqa: E402
    PROJECT_ROOT,
    Config,
    PeriodontalDataset,
    build_model,
    evaluate,
    load_dinov3_backbone,
    resolve_device,
    save_confusion_matrix,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "periodontal_classifier" / "best_val_acc.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with DINOv3 + CNN head classifier")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="训练权重 .pth 路径")
    parser.add_argument("--dinov3_repo", default=None,
                        help="DINOv3 官方代码仓库路径（默认自动探测，同训练脚本）")
    parser.add_argument("--image", default=None, help="单张图片或图片目录（预测模式）")
    parser.add_argument("--eval_test", action="store_true",
                        help="测试集评估模式（需训练输出目录中的 splits.json）")
    parser.add_argument("--data_dir", default=str(Config.DATA_DIR),
                        help="分类数据集根目录（--eval_test 且 splits.json 缺失时用于重建划分）")
    parser.add_argument("--splits_json", default=None,
                        help="训练时保存的 splits.json 路径（默认取 checkpoint 所在目录）")
    parser.add_argument("--output_dir", default=None, help="CSV / 混淆矩阵输出目录（默认 checkpoint 所在目录）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    parser.add_argument("--topk", type=int, default=3, help="预测模式每张图输出前 k 个类别")
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    """加载 checkpoint，返回 (model, class_names, img_size)。"""
    # This checkpoint is a trusted local training artifact. PyTorch 2.6+
    # defaults to weights_only=True, which cannot deserialize this full dict.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", Config.IMG_SIZE)
    Config.IMG_SIZE = img_size
    # 推理时随机初始化 backbone（马上被 checkpoint 覆盖），避免重复加载 342MB 预训练权重
    model = build_model(num_classes=len(class_names), pretrained_weights=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, class_names, img_size


def predict_images(model, class_names, image_path, device, topk, output_csv):
    """对单张图片/目录做预测，输出 top-k 类别与概率，保存 CSV。"""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(Config.IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(Config.DINO_MEAN, Config.DINO_STD),
    ])
    path = Path(image_path)
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    else:
        files = [path]
    if not files:
        print(f"未找到图片: {image_path}")
        return

    rows = []
    with torch.no_grad():
        for f in tqdm(files, desc="Predicting"):
            img = Image.open(f).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu()
            top_probs, top_idx = probs.topk(min(topk, len(class_names)))
            row = {"image": str(f)}
            for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_idx.tolist())):
                row[f"top{i + 1}_class"] = class_names[idx]
                row[f"top{i + 1}_prob"] = round(prob, 4)
            rows.append(row)
            print(f"{f.name}: " + ", ".join(
                f"{class_names[idx]}={prob:.4f}" for prob, idx in zip(top_probs.tolist(), top_idx.tolist())
            ))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"预测结果已保存: {output_csv}")


def eval_test(model, class_names, device, splits_json, data_dir, output_dir, batch_size, num_workers):
    """在测试集上评估并输出混淆矩阵。"""
    manifest = json.loads(Path(splits_json).read_text(encoding="utf-8"))
    if Path(data_dir).resolve() != Path(manifest["data_dir"]).resolve() and not Path(manifest["data_dir"]).exists():
        manifest["data_dir"] = str(data_dir)
    test_ds = PeriodontalDataset(manifest, "test", Config.IMG_SIZE, train=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"测试集样本数: {len(test_ds)}")
    metrics, cm, _, _ = evaluate(model, test_loader, device, class_names, "test")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(cm, class_names, output_dir / "test_confusion_matrix.png",
                          title="Test Confusion Matrix")
    save_confusion_matrix(cm, class_names, output_dir / "test_confusion_matrix_normalized.png",
                          title="Test Confusion Matrix (normalized)", normalized=True)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"评估结果已保存: {output_dir}")
    return metrics


def main():
    args = parse_args()
    if args.dinov3_repo:
        Config.DINOV3_REPO = args.dinov3_repo
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
    if not args.image and not args.eval_test:
        raise SystemExit("必须指定 --image（预测模式）或 --eval_test（测试集评估模式）")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    model, class_names, img_size = load_checkpoint(args.checkpoint)
    model = model.to(device)
    print(f"类别: {class_names} | img_size: {img_size}")

    ckpt_dir = Path(args.checkpoint).resolve().parent
    if args.image:
        output_csv = ckpt_dir / "predictions.csv"
        predict_images(model, class_names, args.image, device, args.topk, output_csv)
    if args.eval_test:
        splits_json = args.splits_json or ckpt_dir / "splits.json"
        if not Path(splits_json).exists():
            raise FileNotFoundError(f"splits.json 不存在: {splits_json}（训练时会自动生成）")
        output_dir = args.output_dir or ckpt_dir
        eval_test(model, class_names, device, splits_json, args.data_dir,
                  output_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
