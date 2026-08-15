"""
测试Cascade检测头的stage1->stage2转换，无需等待21轮训练。
这个脚本加载epoch 20的checkpoint，模拟epoch 21-24的转换过程。
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from model.yolov10_dinov3 import build_model
from data.model_data import build_dataloaders, infer_num_classes
from train_detector_405YOLO import Config, evaluate_model


def test_cascade_transition():
    """模拟cascade stage转换，验证修复是否有效"""

    print("="*80)
    print("Cascade Stage Transition Test")
    print("="*80)

    device = torch.device(Config.DEVICE)
    project_root = Path(__file__).resolve().parent.parent

    # 加载checkpoint
    checkpoint_path = project_root / Config.OUTPUT_DIR / "latest.pth"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("请先训练至少1个epoch")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint epoch: {loaded_epoch}")

    # 构建模型和数据
    num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    model = build_model(num_classes=num_classes, config=Config).to(device)

    # checkpoint是用旧架构(detect_head.stage1/stage2)保存的，
    # 新架构改为detect_head.detect。把stage1的权重映射过去，丢弃stage2。
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()
    remapped = {}
    for k, v in old_state.items():
        if k.startswith("detect_head.stage1."):
            new_k = k.replace("detect_head.stage1.", "detect_head.detect.", 1)
            if new_k in new_state and new_state[new_k].shape == v.shape:
                remapped[new_k] = v
        elif k.startswith("detect_head.stage2."):
            pass  # 丢弃旧stage2权重
        elif k.startswith("detect_head.stage2_refine."):
            if k in new_state and new_state[k].shape == v.shape:
                remapped[k] = v
        else:
            if k in new_state:
                remapped[k] = v
    missing = [k for k in new_state if k not in remapped]
    print(f"Remapped {len(remapped)} keys, missing {len(missing)} (will use random init)")
    new_state.update(remapped)
    model.load_state_dict(new_state)

    train_loader, val_loader = build_dataloaders(config=Config)
    print(f"Val images: {len(val_loader.dataset)}")

    # 测试不同epoch下的表现
    test_epochs = [20, 21, 22, 24, 26, 30]  # stage1末期, stage2开始几轮
    results = []

    for test_epoch in test_epochs:
        print(f"\n{'='*80}")
        print(f"Testing as if epoch = {test_epoch}")
        print(f"{'='*80}")

        # 设置epoch
        model.detect_head.set_epoch(test_epoch)

        # 打印状态
        stage2_enabled = model.detect_head.stage2_enabled
        stage_label = "Stage2" if stage2_enabled else "Stage1"
        print(f"Stage2 enabled: {stage2_enabled}  [{stage_label}]")

        # 测试前向传播
        model.eval()
        test_losses = []

        with torch.no_grad():
            # 只测试前5个batch以节省时间
            for idx, (images, targets) in enumerate(train_loader):
                if idx >= 5:
                    break

                images = images.to(device, non_blocking=True)

                # 训练模式forward (带loss计算)
                model.train()
                output = model(images, targets)
                loss = output["loss"].item()
                test_losses.append(loss)

                if idx == 0:
                    # 打印第一个batch的loss详情
                    loss_items = output["loss_items"].detach().cpu().tolist()
                    print(f"First batch loss: {loss:.4f}, box/cls/dfl={loss_items}")

        avg_loss = sum(test_losses) / len(test_losses)
        print(f"Average loss (5 batches): {avg_loss:.4f}")

        # 验证集快速评估（可选，较慢）
        if test_epoch in [20, 21, 24, 26, 30]:  # 只在关键epoch做完整验证
            print("Running validation...")
            metrics = evaluate_model(model, val_loader, device, use_class_thresholds=False)
            print(f"mAP@[.5:.95]: {metrics['map']:.4f}")
            print(f"mAP@0.50: {metrics['map50']:.4f}")
            print(f"mAP@0.75: {metrics['map75']:.4f}")

            results.append({
                "epoch": test_epoch,
                "loss": avg_loss,
                "map": metrics["map"],
                "map50": metrics["map50"],
                "map75": metrics["map75"],
            })
        else:
            results.append({
                "epoch": test_epoch,
                "loss": avg_loss,
            })

    # 总结
    print("\n" + "="*80)
    print("SUMMARY - Cascade Transition Test Results")
    print("="*80)
    print(f"{'Epoch':<8} {'Stage':<20} {'Loss':<12} {'mAP@[.5:.95]':<15} {'Status'}")
    print("-"*80)

    for r in results:
        epoch = r["epoch"]
        loss = r["loss"]
        map_val = r.get("map", None)

        if epoch <= Config.CASCADE_STAGE1_EPOCHS:
            stage = "Stage1"
        else:
            stage = "Stage2"

        map_str = f"{map_val:.4f}" if map_val else "N/A"

        # 判断状态
        if epoch == 20:
            status = "✓ Baseline"
        elif epoch == 21:
            if loss < 80:  # 期望损失不超过80
                status = "✓ Good"
            else:
                status = "✗ LOSS SPIKE!"
        elif epoch in [22, 23]:
            if loss < 90:
                status = "✓ Stabilizing"
            else:
                status = "⚠ High loss"
        else:
            status = "✓ Stage2 active"

        print(f"{epoch:<8} {stage:<20} {loss:<12.4f} {map_str:<15} {status}")

    print("\n" + "="*80)
    print("Expected behavior:")
    print("  • Epoch 20: Baseline (loss ~50-60)")
    print("  • Epoch 21: Warmup start, loss should stay <80 (NOT 140+)")
    print("  • Epoch 22-23: Warmup continue, loss <90")
    print("  • Epoch 24+: Stage2 active, loss converging")
    print("="*80)

    # 判断修复是否成功
    epoch21_loss = next(r["loss"] for r in results if r["epoch"] == 21)
    if epoch21_loss < 80:
        print("\n✓✓✓ TEST PASSED: Epoch 21 loss is under control!")
        print(f"    Loss={epoch21_loss:.2f} < 80 (Previous bug: 143.8)")
    else:
        print("\n✗✗✗ TEST FAILED: Epoch 21 still has loss spike!")
        print(f"    Loss={epoch21_loss:.2f} >= 80")
        print("    The fix may need adjustment.")


if __name__ == "__main__":
    test_cascade_transition()
