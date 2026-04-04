import os
import json
import random
import cv2
from pathlib import Path


ROOT_DIR = 
IMAGE_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/image"
LABEL_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/label"
OUTPUT_DIR = "/Users/mises/Desktop/Sitp/Dataset/Caries/coco_annotations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

categories = [{"id": 1, "name": "caries"}]

# 收集所有 JSON 文件
json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
print(f"Found {len(json_files)} JSON files")
random.shuffle(json_files)

split_idx = int(0.8 * len(json_files))
train_files = json_files[:split_idx]
val_files = json_files[split_idx:]

def convert_labelme_to_coco(json_files, output_json, set_name):
    images = []
    annotations = []
    ann_id = 1
    for img_id, json_file in enumerate(json_files, start=1):
        # 读取 JSON
        json_path = os.path.join(LABEL_DIR, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 图像文件名（从 JSON 的 imagePath 获取）
        img_filename = data.get('imagePath', os.path.splitext(json_file)[0] + '.jpg')
        # 如果 imagePath 包含路径，只取文件名
        if os.path.dirname(img_filename):
            img_filename = os.path.basename(img_filename)
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        # 如果图片不存在，尝试其他扩展名
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = os.path.join(IMAGE_DIR, os.path.splitext(img_filename)[0] + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
            else:
                print(f"Warning: {img_path} not found, skip")
                continue

        # 读取图像尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}, skip")
            continue
        height, width = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": width,
            "height": height
        })

        # 解析标注
        shapes = data.get('shapes', [])
        if not shapes:
            print(f"Warning: {json_file} has no shapes")
        
        for shape in shapes:
            # 匹配标签（不区分大小写）
            label = shape.get('label', '').lower()
            if label != 'caries':
                continue
            
            points = shape.get('points', [])
            if len(points) < 2:
                continue
            
            # 取前两个点作为矩形（左上和右下）
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 确保坐标是整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 确保 x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            # 边界框 [x, y, w, h]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            # 确保宽高为正
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            
            area = bbox[2] * bbox[3]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    print(f"{set_name}: Saved {len(images)} images, {len(annotations)} annotations to {output_json}")

convert_labelme_to_coco(train_files, os.path.join(OUTPUT_DIR, "train.json"), "Train")
convert_labelme_to_coco(val_files, os.path.join(OUTPUT_DIR, "val.json"), "Val")