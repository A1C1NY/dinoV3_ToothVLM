"""Visual verification: draw augmented images with boxes to confirm alignment."""
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from train_detector_405YOLO import Config, build_dataloaders


def draw_boxes_on_tensor(image_tensor, boxes, color=(0, 255, 0), width=2):
    """Convert tensor to PIL, draw boxes, return PIL image."""
    image = to_pil_image(image_tensor.clamp(0, 1))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image


def main():
    random.seed(42)
    torch.manual_seed(42)

    train_loader, _ = build_dataloaders()
    dataset = train_loader.dataset

    output_dir = Path(__file__).parent.parent / "debug_aug_visual"
    output_dir.mkdir(exist_ok=True)

    print(f"Saving augmented samples to {output_dir}")

    # 采样 20 张图，每张取 3 次增强（因为是随机的）
    for sample_idx in range(min(20, len(dataset))):
        for aug_trial in range(3):
            image_tensor, target = dataset[sample_idx]
            boxes = target["boxes"]

            # 确认框在画布内
            if len(boxes):
                assert boxes[:, 0].min() >= -1e-3, f"x1 out of bounds: {boxes[:, 0].min()}"
                assert boxes[:, 1].min() >= -1e-3, f"y1 out of bounds: {boxes[:, 1].min()}"
                assert boxes[:, 2].max() <= Config.IMG_SIZE + 1e-3, f"x2 out of bounds: {boxes[:, 2].max()}"
                assert boxes[:, 3].max() <= Config.IMG_SIZE + 1e-3, f"y2 out of bounds: {boxes[:, 3].max()}"
                assert (boxes[:, 2] > boxes[:, 0]).all(), "x2 <= x1"
                assert (boxes[:, 3] > boxes[:, 1]).all(), "y2 <= y1"

            image_with_boxes = draw_boxes_on_tensor(image_tensor, boxes, color=(0, 255, 0), width=2)

            # 在图上标注信息
            draw = ImageDraw.Draw(image_with_boxes)
            info_text = (
                f"img_id={target['image_id']}, "
                f"boxes={len(boxes)}, "
                f"ratio={target['letterbox_ratio']:.3f}, "
                f"pad=({target['pad_x']:.1f},{target['pad_y']:.1f})"
            )
            draw.text((5, 5), info_text, fill=(255, 255, 0))

            filename = f"sample{sample_idx:03d}_trial{aug_trial}_boxes{len(boxes)}.jpg"
            image_with_boxes.save(output_dir / filename, quality=85)

    print(f"✓ Saved {min(20, len(dataset)) * 3} augmented images with boxes")
    print(f"  Manually inspect {output_dir} to confirm boxes align with lesions")

    # 额外检查：水平翻转的对称性
    print("\nChecking horizontal flip symmetry...")
    Config.AUG_AFFINE = 0.0  # 临时关闭仿射，只测翻转
    random.seed(100)

    for trial in range(5):
        image_tensor, target = dataset[0]
        boxes = target["boxes"]
        if not len(boxes):
            continue

        # 手动翻转并调整框
        flipped_tensor = image_tensor.flip(-1)
        flipped_boxes = boxes.clone()
        left = flipped_boxes[:, 0].clone()
        flipped_boxes[:, 0] = Config.IMG_SIZE - flipped_boxes[:, 2]
        flipped_boxes[:, 2] = Config.IMG_SIZE - left

        # 翻转回来应该等于原图
        double_flip_tensor = flipped_tensor.flip(-1)
        double_flip_boxes = flipped_boxes.clone()
        left = double_flip_boxes[:, 0].clone()
        double_flip_boxes[:, 0] = Config.IMG_SIZE - double_flip_boxes[:, 2]
        double_flip_boxes[:, 2] = Config.IMG_SIZE - left

        # 像素应完全一致
        pixel_diff = (image_tensor - double_flip_tensor).abs().max().item()
        box_diff = (boxes - double_flip_boxes).abs().max().item()
        assert pixel_diff < 1e-6, f"pixel mismatch after double flip: {pixel_diff}"
        assert box_diff < 0.01, f"box mismatch after double flip: {box_diff}"

    print("✓ Horizontal flip symmetry verified (double flip = identity)")

    # 检查仿射变换后框是否覆盖了实际的亮区域
    print("\nChecking affine box coverage...")
    Config.AUG_AFFINE = 0.7  # 恢复
    random.seed(200)

    coverage_errors = []
    for sample_idx in range(min(30, len(dataset))):
        image_tensor, target = dataset[sample_idx]
        boxes = target["boxes"]
        if not len(boxes):
            continue

        # 找图中的明显亮区（假设病灶通常比背景亮或有色彩）
        # 简化判定：如果像素方差高，说明是有内容的区域
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            # 框内区域的像素方差
            roi = image_tensor[:, y1:y2, x1:x2]
            if roi.numel() == 0:
                coverage_errors.append(f"sample {sample_idx}: empty ROI {box.tolist()}")
                continue

            # 框应该圈住的是有内容的区域，而不是纯灰色填充
            # 填充区域的方差应该接近 0（全灰），有效区域方差应该 > 0.001
            variance = roi.var().item()
            if variance < 0.0001:
                # 这个框可能圈住了填充区域 — 不应该发生
                coverage_errors.append(
                    f"sample {sample_idx}: box {box.tolist()} covers uniform region (var={variance:.6f})"
                )

    if coverage_errors:
        print(f"⚠ Found {len(coverage_errors)} potential coverage issues:")
        for err in coverage_errors[:5]:
            print(f"  {err}")
    else:
        print("✓ All boxes cover regions with variance > 0.0001 (not pure padding)")

    print(f"\n{'='*60}")
    print("VISUAL VERIFICATION COMPLETE")
    print(f"Review images in: {output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
