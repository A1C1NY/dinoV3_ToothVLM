import os
import json
import random
from tqdm import tqdm
import cv2
from pathlib import Path


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
        for current_idx, json_file in enumerate(tqdm(file_list, desc=subset_name), start=1):
            img_id = img_id_offset + current_idx
            json_path = label_dir / json_file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_filename = data.get('imagePath', os.path.splitext(json_file)[0] + '.jpg')
            if os.path.dirname(img_filename):
                img_filename = os.path.basename(img_filename)
            img_path = image_dir / img_filename

            if not img_path.exists():
                for ext in ['.jpg', '.png', '.jpeg']:
                    alt_path = image_dir / (os.path.splitext(img_filename)[0] + ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    print(f"Warning: {img_path} not found, skip")
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
                label = shape.get('label', '').strip().lower()
                # 兼容不同疾病的标签匹配
                if label != category_info['name'].lower():
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


if __name__ == "__main__":
    main()


