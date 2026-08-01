"""
随机可视化 LabelMe 标注在图片上的实际位置，用于验证标注质量。
"""
import os
import json
import random
import argparse
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = ROOT_DIR / "405" / "image_filtered"
LABEL_DIR = ROOT_DIR / "405" / "label_filtered"

# 每个类别的展示颜色 (B, G, R)
CATEGORY_COLORS = {
    "Caries": (0, 255, 0),                # 绿色
    "Calculus": (255, 0, 0),              # 蓝色
    "Mouth_Ulcer": (0, 165, 255),         # 橙色
    "Periodontal_Disease": (0, 0, 255),   # 红色
    "Tooth_Discoloration": (255, 255, 0), # 青色
}

CATEGORIES = {
    "Caries": {"id": 1, "name": "caries"},
    "Calculus": {"id": 2, "name": "calculus"},
    "Mouth_Ulcer": {"id": 3, "name": "mouth_ulcer"},
    "Periodontal_Disease": {"id": 4, "name": "periodontal_disease"},
    "Tooth_Discoloration": {"id": 5, "name": "tooth_discoloration"},
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def extract_image_filename(image_path_value: str, fallback_filename: str) -> str:
    raw = str(image_path_value or "").strip()
    if not raw:
        return fallback_filename
    win_name = PureWindowsPath(raw).name
    posix_name = Path(raw).name
    if win_name and win_name not in (".", ".."):
        return win_name
    if posix_name and posix_name not in (".", ".."):
        return posix_name
    return os.path.basename(raw.replace("\\", "/")) or fallback_filename


def find_image_path(json_file: str, json_path: Path, image_path_value: str) -> Path | None:
    """根据 JSON 中的 imagePath 找到实际图片文件。"""
    default_name = os.path.splitext(json_file)[0] + ".jpg"
    img_filename = extract_image_filename(image_path_value, default_name)
    stem = Path(img_filename).stem

    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = IMAGE_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_labelme_annotations(json_path: Path):
    """加载单个 LabelMe JSON，返回 (image_path, shapes_list) 或 None。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_file = json_path.name
    img_path = find_image_path(json_file, json_path, data.get("imagePath", ""))
    if img_path is None:
        return None

    shapes = data.get("shapes", [])
    if not shapes:
        return None

    return img_path, shapes


def draw_annotations(img: np.ndarray, shapes: list):
    """在图像上绘制所有标注的多边形轮廓和最小外接矩形。"""
    overlay = img.copy()

    for shape in shapes:
        label_raw = shape.get("label", "unknown")
        normalized = normalize_label(label_raw)

        # 找到匹配的类别
        matched_category = None
        for cat_name, cat_info in CATEGORIES.items():
            if normalize_label(cat_info["name"]) == normalized:
                matched_category = cat_name
                break

        color = CATEGORY_COLORS.get(matched_category, (128, 128, 128))
        display_name = matched_category or label_raw

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        pts = np.array(points, dtype=np.int32)

        # 绘制多边形轮廓（半透明）
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

        # 计算并绘制最小外接矩形
        xs, ys = pts[:, 0], pts[:, 1]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2)

        # 在矩形左上角标注类别名
        label_y = max(y1 - 5, 20)
        cv2.putText(overlay, display_name, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)

    # 图像融合，使标注半透明
    alpha = 0.5
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description="随机可视化 LabelMe 标注")
    parser.add_argument("-n", "--num", type=int, default=5,
                        help="随机展示的图片数量（默认 5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("-d", "--disease", type=str, default=None,
                        choices=list(CATEGORIES.keys()),
                        help="只展示指定疾病的标注")
    parser.add_argument("--only-annotated", action="store_true",
                        help="只展示有标注的图片（跳过无标注或标注不匹配的）")
    args = parser.parse_args()

    random.seed(args.seed)

    # 收集所有 JSON 文件
    json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {LABEL_DIR}")
        return

    # 按疾病筛选
    if args.disease:
        prefix = "Dental_Dieases_"
        json_files = [
            f for f in json_files
            if Path(f).stem.startswith(prefix)
            and Path(f).stem[len(prefix):].rpartition("_")[0] == args.disease
        ]
        print(f"Filtered to disease '{args.disease}': {len(json_files)} JSON files")

    random.shuffle(json_files)

    shown = 0
    idx = 0
    while shown < args.num and idx < len(json_files):
        json_file = json_files[idx]
        idx += 1

        json_path = LABEL_DIR / json_file
        result = load_labelme_annotations(json_path)

        if result is None:
            continue

        img_path, shapes = result

        # 根据 --disease 过滤不匹配的标注
        if args.disease:
            expected = normalize_label(CATEGORIES[args.disease]["name"])
            shapes = [s for s in shapes if normalize_label(s.get("label", "")) == expected]
            if not shapes and args.only_annotated:
                continue

        if not shapes and args.only_annotated:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue

        annotated = draw_annotations(img, shapes)

        # 缩放到合适大小显示
        h, w = annotated.shape[:2]
        max_display = 1200
        scale = min(max_display / max(h, w), 1.0)
        if scale < 1.0:
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        title = f"[{shown + 1}/{args.num}] {img_path.name} | {len(shapes)} annotations"
        cv2.imshow(title, annotated)

        print(f"[{shown + 1}/{args.num}] {img_path.name}: {len(shapes)} annotation(s)")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)

        if key == 27:  # ESC 退出
            print("ESC pressed, exiting.")
            break
        elif key == ord("s"):
            # 按 s 跳过，不减 shown 计数
            shown -= 1

        shown += 1

    cv2.destroyAllWindows()
    print(f"\nFinished. Displayed {shown} images.")


if __name__ == "__main__":
    main()
