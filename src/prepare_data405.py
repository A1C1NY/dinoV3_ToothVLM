import os
import json 
import random
from tqdm import tqdm
import cv2
from pathlib import Path, PureWindowsPath

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # 仓库上级目录

IMAGE_DIR = ROOT_DIR / "405" / "image_filtered"
LABEL_DIR = ROOT_DIR / "405" / "label_filtered"
COCO_DIR = Path(__file__).resolve().parent.parent / "coco"

# Set either flag to False when only one output layout is needed.
EXPORT_SEPARATE_DATASETS = False
EXPORT_MERGED_DATASET = True

CATEGORIES = {
    "Caries": {"id": 1, "name": "caries"},
    "Calculus": {"id": 2, "name": "calculus"},
    "Mouth_Ulcer": {"id": 3, "name": "mouth_ulcer"},
    "Periodontal_Disease": {"id": 4, "name": "periodontal_disease"},
    "Tooth_Discoloration": {"id": 5, "name": "tooth_discoloration"},
}

DISEASES = [
    "Caries",
    "Calculus",
    "Mouth_Ulcer",
    "Periodontal_Disease",
    "Tooth_Discoloration",
]

def normalize_label(label):
    """
    统一大小写与分隔符，兼容 Mouth_Ulcer / mouth ulcer / mouth-ulcer 等写法
    """
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def extract_image_filename(image_path_value, fallback_filename):
    """
    从 LabelMe 的 imagePath 中稳健提取文件名。
    兼容 Windows 路径（反斜杠）和 POSIX 路径。

    :param image_path_value: LabelMe JSON 中的 imagePath 字段值
    :param fallback_filename: 如果 imagePath 无效，则使用的默认文件名
    
    :return: 提取出的文件名
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

def convert_labelme_to_coco(image_dir, label_dir, output_dir, set_name, category_info, json_files):
    """
    将 LabelMe 格式的标注文件转换为 COCO 格式。

    :param image_dir: 存放图像的目录
    :param label_dir: 存放 LabelMe JSON 文件的目录
    :param output_dir: 输出 COCO JSON 文件的目录
    :param set_name: 数据集名称（如 'train' 或 'val'）
    :param category_info: 类别信息字典
    """
    print(f"Found {len(json_files)} JSON files in {label_dir} for set '{set_name}'.")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42) 
    random.shuffle(json_files)  

    split_idx = int(len(json_files) * 0.8)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    def convert_file_list(file_list, output_json, subset_name, img_id_offset, ann_id_offset):
        """
        将单个文件列表转换为 COCO 格式。

        :param file_list: JSON 文件列表
        :param output_json: 输出的 COCO JSON 文件路径
        :param subset_name: 数据集子集名称（如 'train' 或 'val'）
        :param img_id_offset: 图像 ID 偏移量
        :param ann_id_offset: 标注 ID 偏移量
        """
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
            "categories": list(CATEGORIES.values())
        }
        if output_json is not None:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)
        print(f"{subset_name}: Saved {len(images)} images, {len(annotations)} annotations")
        return coco_data

    # 为不同疾病分配不冲突的 ID 空间
    # 每个疾病分配 100,000 个 ID，训练集从 0 开始，验证集从 50,000 开始
    # 如果以后数据更大，考虑将 ID 空间扩大到 1,000,000 或使用 UUID 等更灵活的 ID 生成方式，以避免 ID 冲突
    category_offset = category_info['id'] * 100000
    train_path = output_dir / "train.json" if output_dir is not None else None
    val_path = output_dir / "val.json" if output_dir is not None else None
    return {
        "train": convert_file_list(train_files, train_path, f"{set_name} Train", category_offset, category_offset),
        "val": convert_file_list(val_files, val_path, f"{set_name} Val", category_offset + 50000, category_offset + 50000),
    }

def source_disease(json_file):
    """Map the flat 405 filename back to its original disease directory."""
    prefix = "Dental_Dieases_"
    stem = Path(json_file).stem
    if not stem.startswith(prefix):
        return None

    disease, separator, _ = stem[len(prefix):].rpartition("_")
    return disease if separator and disease in CATEGORIES else None


def main():
    json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    json_files_by_disease = {
        disease: [f for f in json_files if source_disease(f) == disease]
        for disease in DISEASES
    }

    output_dirs = {
        disease: COCO_DIR / disease
        for disease in DISEASES
    }

    converted_by_disease = {}
    if EXPORT_SEPARATE_DATASETS or EXPORT_MERGED_DATASET:
        for disease in DISEASES:
            category_info = CATEGORIES[disease]
            output_dir = output_dirs[disease] if EXPORT_SEPARATE_DATASETS else None
            print(
                f"Processing {disease}: image_dir={IMAGE_DIR}, label_dir={LABEL_DIR}, "
                f"output_dir={output_dir}, category={category_info}"
            )
            converted_by_disease[disease] = convert_labelme_to_coco(
                IMAGE_DIR,
                LABEL_DIR,
                output_dir,
                disease,
                category_info,
                json_files_by_disease[disease],
            )

    if not EXPORT_MERGED_DATASET:
        return

    all_images_by_subset = {"train": [], "val": []}
    all_annotations_by_subset = {"train": [], "val": []}

    for disease in DISEASES:
        for subset in ["train", "val"]:
            data = converted_by_disease[disease][subset]
            all_images_by_subset[subset].extend(data.get("images", []))
            all_annotations_by_subset[subset].extend(data.get("annotations", []))

    valid_category_ids = {category["id"] for category in CATEGORIES.values()}
    for subset in ["train", "val"]:
        images = all_images_by_subset[subset]
        annotations = all_annotations_by_subset[subset]
        image_ids = [image["id"] for image in images]
        annotation_ids = [annotation["id"] for annotation in annotations]
        image_id_set = set(image_ids)

        duplicate_images = len(image_ids) - len(image_id_set)
        duplicate_annotations = len(annotation_ids) - len(set(annotation_ids))
        bad_image_refs = sum(
            annotation.get("image_id") not in image_id_set
            for annotation in annotations
        )
        bad_category_refs = sum(
            annotation.get("category_id") not in valid_category_ids
            for annotation in annotations
        )
        if duplicate_images or duplicate_annotations or bad_image_refs or bad_category_refs:
            print(
                f"Warning [{subset}] integrity issue: "
                f"dup_image_ids={duplicate_images}, dup_ann_ids={duplicate_annotations}, "
                f"bad_image_ref={bad_image_refs}, bad_category_ref={bad_category_refs}"
            )

        output_path = COCO_DIR / "All_Diseases" / f"{subset}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": list(CATEGORIES.values()),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"All diseases [{subset}]: Saved {len(images)} images, "
            f"{len(annotations)} annotations to {output_path}"
        )


if __name__ == "__main__":
    main()
