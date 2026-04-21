import os
import json
import random
from tqdm import tqdm
import cv2
from pathlib import Path, PureWindowsPath


ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # 仓库上级目录
DISEASES = [
    "Caries",
    "Calculus",
    "Mouth_Ulcer",
    "Periodontal_Disease",
    "Tooth_Discoloration",
]
IMAGE_DIRS = [ROOT_DIR / "Dataset" / disease / "image" for disease in DISEASES]
LABEL_DIRS = [ROOT_DIR / "Dataset" / disease / "label" for disease in DISEASES]
OUTPUT_DIRS = [Path(__file__).resolve().parent.parent / "coco" / disease for disease in DISEASES]
for output_dir in OUTPUT_DIRS:
    output_dir.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "Caries": {"id": 1, "name": "caries"},
    "Calculus": {"id": 2, "name": "calculus"},
    "Mouth_Ulcer": {"id": 3, "name": "mouth_ulcer"},
    "Periodontal_Disease": {"id": 4, "name": "periodontal_disease"},
    "Tooth_Discoloration": {"id": 5, "name": "tooth_discoloration"},
}


def normalize_label(label):
    # 统一大小写与分隔符，兼容 Mouth_Ulcer / mouth ulcer / mouth-ulcer 等写法
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def extract_image_filename(image_path_value, fallback_filename):
    """
    从 LabelMe 的 imagePath 中稳健提取文件名。
    兼容 Windows 路径（反斜杠）和 POSIX 路径。
    """
    raw = str(image_path_value or "").strip()
    if not raw:
        return fallback_filename

    # PureWindowsPath 能正确处理诸如 "..\\image\\a.jpg" 的场景。
    win_name = PureWindowsPath(raw).name
    posix_name = Path(raw).name

    if win_name and win_name not in (".", ".."):
        return win_name
    if posix_name and posix_name not in (".", ".."):
        return posix_name

    # 最后兜底：手动替换反斜杠后再取 basename
    return os.path.basename(raw.replace("\\", "/")) or fallback_filename


def convert_labelme_to_coco(image_dir, label_dir, output_dir, set_name, category_info):
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    print(f"{set_name}: Found {len(json_files)} JSON files in {label_dir}")
    random.seed(42)  # 固定随机种子确保可复现
    random.shuffle(json_files)

    split_idx = int(0.8 * len(json_files))
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    def convert_file_list(file_list, output_json, subset_name, img_id_offset, ann_id_offset):
        images = []
        annotations = []
        ann_id = ann_id_offset
        expected_label = normalize_label(category_info['name'])
        for current_idx, json_file in enumerate(tqdm(file_list, desc=subset_name), start=1):
            img_id = img_id_offset + current_idx
            json_path = label_dir / json_file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            default_name = os.path.splitext(json_file)[0] + '.jpg'
            img_filename = extract_image_filename(data.get('imagePath', ''), default_name)

            stem = Path(img_filename).stem
            candidate_names = [
                img_filename,
                f"{stem}.jpg",
                f"{stem}.jpeg",
                f"{stem}.png",
                f"{stem}.JPG",
                f"{stem}.JPEG",
                f"{stem}.PNG",
            ]

            img_path = None
            for candidate_name in dict.fromkeys(candidate_names):
                candidate_path = image_dir / candidate_name
                if candidate_path.exists():
                    img_path = candidate_path
                    break

            if img_path is None:
                print(f"Warning: image not found for {json_path.name}, imagePath={data.get('imagePath', '')}, parsed={img_filename}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: cannot read {img_path}, skip")
                continue
            height, width = img.shape[:2]

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })

            shapes = data.get('shapes', [])
            for shape in shapes:
                label = normalize_label(shape.get('label', ''))
                # 兼容不同疾病的标签匹配
                if label != expected_label:
                    continue

                points = shape.get('points', [])
                if len(points) < 2:
                    continue

                # 自动从点集计算最小外接矩形 [x, y, w, h]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                area = bbox[2] * bbox[3]
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_info['id'],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": list(CATEGORIES.values())  # 写入所有类别定义
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        print(f"{subset_name}: Saved {len(images)} images, {len(annotations)} annotations")

    # 为不同疾病分配不冲突的 ID 空间
    # 每个疾病分配 100,000 个 ID，训练集从 0 开始，验证集从 50,000 开始
    # 如果以后数据更大，考虑将 ID 空间扩大到 1,000,000 或使用 UUID 等更灵活的 ID 生成方式，以避免 ID 冲突
    category_offset = category_info['id'] * 100000
    convert_file_list(train_files, output_dir / "train.json", f"{set_name} Train", category_offset, category_offset)
    convert_file_list(val_files, output_dir / "val.json", f"{set_name} Val", category_offset + 50000, category_offset + 50000)


def main():
    for disease, image_dir, label_dir, output_dir in zip(DISEASES, IMAGE_DIRS, LABEL_DIRS, OUTPUT_DIRS):
        category_info = CATEGORIES[disease]
        print(f"Processing {disease}: image_dir={image_dir}, label_dir={label_dir}, output_dir={output_dir}, category={category_info}")
        convert_labelme_to_coco(image_dir, label_dir, output_dir, disease, category_info)


    # 再生成一份所有疾病合并的 COCO 数据集，供整体训练使用
    # 关键修复：train/val 必须分别累积，不能共用同一个列表，否则会互相污染。
    all_images_by_subset = {"train": [], "val": []}
    all_annotations_by_subset = {"train": [], "val": []}

    for disease, output_dir in zip(DISEASES, OUTPUT_DIRS):
        for subset in ["train", "val"]:
            json_path = output_dir / f"{subset}.json"
            if not json_path.exists():
                print(f"Warning: {json_path} not found, skip")
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_images_by_subset[subset].extend(data.get("images", []))
            all_annotations_by_subset[subset].extend(data.get("annotations", []))

    # 写出并做基本一致性检查（ID 冲突、annotation.image_id/category_id 有效性）
    valid_category_ids = {c["id"] for c in CATEGORIES.values()}
    for subset in ["train", "val"]:
        images = all_images_by_subset[subset]
        annotations = all_annotations_by_subset[subset]

        image_ids = [img["id"] for img in images]
        ann_ids = [ann["id"] for ann in annotations]
        image_id_set = set(image_ids)

        dup_image_ids = len(image_ids) - len(image_id_set)
        dup_ann_ids = len(ann_ids) - len(set(ann_ids))

        bad_image_ref = 0
        bad_category_ref = 0
        for ann in annotations:
            if ann.get("image_id") not in image_id_set:
                bad_image_ref += 1
            if ann.get("category_id") not in valid_category_ids:
                bad_category_ref += 1

        if dup_image_ids or dup_ann_ids or bad_image_ref or bad_category_ref:
            print(
                f"Warning [{subset}] integrity issue: "
                f"dup_image_ids={dup_image_ids}, dup_ann_ids={dup_ann_ids}, "
                f"bad_image_ref={bad_image_ref}, bad_category_ref={bad_category_ref}"
            )

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": list(CATEGORIES.values())
        }

        all_diseases_path = Path(__file__).resolve().parent.parent / "coco" / "All_Diseases" / f"{subset}.json"
        all_diseases_path.parent.mkdir(parents=True, exist_ok=True)

        with open(all_diseases_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        print(f"All diseases [{subset}]: Saved {len(images)} images, {len(annotations)} annotations to {all_diseases_path}")

if __name__ == "__main__":
    main()


