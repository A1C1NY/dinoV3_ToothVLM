"""
对比 957 和 957n 数据集的详细差异分析

用法：
    cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM" && \
    C:/ProgramData/anaconda3/envs/dino_VLM/python.exe src/utils/compare_datasets.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def analyze_dataset(name, train_json, val_json):
    """分析单个数据集的统计特征"""
    train = json.load(open(train_json))
    val = json.load(open(val_json))

    # 1. 类别分布
    train_cats = Counter([a['category_id'] for a in train['annotations']])
    val_cats = Counter([a['category_id'] for a in val['annotations']])

    # 2. 框面积分布（相对于图像面积的比例）
    train_areas = []
    for a in train['annotations']:
        img = next(i for i in train['images'] if i['id'] == a['image_id'])
        img_area = img['width'] * img['height']
        bbox_area = a['bbox'][2] * a['bbox'][3]
        train_areas.append(bbox_area / img_area)

    val_areas = []
    for a in val['annotations']:
        img = next(i for i in val['images'] if i['id'] == a['image_id'])
        img_area = img['width'] * img['height']
        bbox_area = a['bbox'][2] * a['bbox'][3]
        val_areas.append(bbox_area / img_area)

    # 3. 每图标注数分布
    img_to_annots_train = defaultdict(int)
    for a in train['annotations']:
        img_to_annots_train[a['image_id']] += 1

    img_to_annots_val = defaultdict(int)
    for a in val['annotations']:
        img_to_annots_val[a['image_id']] += 1

    # 4. 宽高比分布
    train_aspect_ratios = [a['bbox'][2] / max(a['bbox'][3], 1e-6) for a in train['annotations']]
    val_aspect_ratios = [a['bbox'][2] / max(a['bbox'][3], 1e-6) for a in val['annotations']]

    # 5. 类别级的框面积统计
    cat_areas_train = defaultdict(list)
    for a in train['annotations']:
        img = next(i for i in train['images'] if i['id'] == a['image_id'])
        img_area = img['width'] * img['height']
        bbox_area = a['bbox'][2] * a['bbox'][3]
        cat_areas_train[a['category_id']].append(bbox_area / img_area)

    cat_areas_val = defaultdict(list)
    for a in val['annotations']:
        img = next(i for i in val['images'] if i['id'] == a['image_id'])
        img_area = img['width'] * img['height']
        bbox_area = a['bbox'][2] * a['bbox'][3]
        cat_areas_val[a['category_id']].append(bbox_area / img_area)

    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"{'='*70}")

    print(f"\n【基本统计】")
    print(f"  Train: {len(train['images'])} images, {len(train['annotations'])} annotations")
    print(f"  Val:   {len(val['images'])} images, {len(val['annotations'])} annotations")

    print(f"\n【类别分布】")
    category_names = {1: "Caries", 2: "Calculus", 3: "Mouth_Ulcer", 4: "Tooth_Discoloration"}
    print(f"  Train:")
    for cat_id in sorted(train_cats.keys()):
        print(f"    {category_names.get(cat_id, f'Class{cat_id}')}: {train_cats[cat_id]} "
              f"({100*train_cats[cat_id]/sum(train_cats.values()):.1f}%)")
    print(f"  Val:")
    for cat_id in sorted(val_cats.keys()):
        print(f"    {category_names.get(cat_id, f'Class{cat_id}')}: {val_cats[cat_id]} "
              f"({100*val_cats[cat_id]/sum(val_cats.values()):.1f}%)")

    print(f"\n【框面积分布】(相对于图像面积)")
    print(f"  Train: mean={np.mean(train_areas):.4f}, std={np.std(train_areas):.4f}, "
          f"median={np.median(train_areas):.4f}")
    print(f"  Val:   mean={np.mean(val_areas):.4f}, std={np.std(val_areas):.4f}, "
          f"median={np.median(val_areas):.4f}")

    print(f"\n【每图标注数】")
    train_annots_per_img = list(img_to_annots_train.values())
    val_annots_per_img = list(img_to_annots_val.values())
    print(f"  Train: mean={np.mean(train_annots_per_img):.2f}, std={np.std(train_annots_per_img):.2f}, "
          f"median={np.median(train_annots_per_img):.1f}, max={max(train_annots_per_img)}")
    print(f"  Val:   mean={np.mean(val_annots_per_img):.2f}, std={np.std(val_annots_per_img):.2f}, "
          f"median={np.median(val_annots_per_img):.1f}, max={max(val_annots_per_img)}")

    print(f"\n【宽高比分布】")
    print(f"  Train: mean={np.mean(train_aspect_ratios):.2f}, std={np.std(train_aspect_ratios):.2f}")
    print(f"  Val:   mean={np.mean(val_aspect_ratios):.2f}, std={np.std(val_aspect_ratios):.2f}")

    print(f"\n【类别不平衡比率】")
    if train_cats:
        imbalance_ratio_train = max(train_cats.values()) / min(train_cats.values())
        print(f"  Train: {imbalance_ratio_train:.2f}:1 "
              f"(最多类={max(train_cats.values())}, 最少类={min(train_cats.values())})")
    if val_cats:
        imbalance_ratio_val = max(val_cats.values()) / min(val_cats.values())
        print(f"  Val:   {imbalance_ratio_val:.2f}:1 "
              f"(最多类={max(val_cats.values())}, 最少类={min(val_cats.values())})")

    print(f"\n【各类别框面积统计】(相对面积)")
    for split_name, cat_areas in [("Train", cat_areas_train), ("Val", cat_areas_val)]:
        print(f"  {split_name}:")
        for cat_id in sorted(cat_areas.keys()):
            areas = cat_areas[cat_id]
            print(f"    {category_names.get(cat_id, f'Class{cat_id}')}: "
                  f"mean={np.mean(areas):.4f}, std={np.std(areas):.4f}, "
                  f"median={np.median(areas):.4f}")

    return {
        'train_cats': train_cats,
        'val_cats': val_cats,
        'train_areas': train_areas,
        'val_areas': val_areas,
        'train_annots_per_img': train_annots_per_img,
        'val_annots_per_img': val_annots_per_img,
        'cat_areas_train': cat_areas_train,
        'cat_areas_val': cat_areas_val,
    }


def compare_two_datasets(stats1, stats2, name1, name2):
    """对比两个数据集的关键差异"""
    print(f"\n{'='*70}")
    print(f"对比分析: {name1} vs {name2}")
    print(f"{'='*70}")

    print(f"\n【类别分布差异】")
    category_names = {1: "Caries", 2: "Calculus", 3: "Mouth_Ulcer", 4: "Tooth_Discoloration"}

    print(f"  Train集差异:")
    for cat_id in sorted(set(stats1['train_cats'].keys()) | set(stats2['train_cats'].keys())):
        count1 = stats1['train_cats'].get(cat_id, 0)
        count2 = stats2['train_cats'].get(cat_id, 0)
        diff = count2 - count1
        pct_diff = 100 * diff / count1 if count1 > 0 else float('inf')
        print(f"    {category_names.get(cat_id, f'Class{cat_id}')}: "
              f"{name1}={count1}, {name2}={count2}, "
              f"差异={diff:+d} ({pct_diff:+.1f}%)")

    print(f"\n  Val集差异:")
    for cat_id in sorted(set(stats1['val_cats'].keys()) | set(stats2['val_cats'].keys())):
        count1 = stats1['val_cats'].get(cat_id, 0)
        count2 = stats2['val_cats'].get(cat_id, 0)
        diff = count2 - count1
        pct_diff = 100 * diff / count1 if count1 > 0 else float('inf')
        print(f"    {category_names.get(cat_id, f'Class{cat_id}')}: "
              f"{name1}={count1}, {name2}={count2}, "
              f"差异={diff:+d} ({pct_diff:+.1f}%)")

    print(f"\n【框面积差异】")
    mean1_train = np.mean(stats1['train_areas'])
    mean2_train = np.mean(stats2['train_areas'])
    print(f"  Train: {name1}={mean1_train:.4f}, {name2}={mean2_train:.4f}, "
          f"差异={mean2_train - mean1_train:+.4f} ({100*(mean2_train-mean1_train)/mean1_train:+.1f}%)")

    mean1_val = np.mean(stats1['val_areas'])
    mean2_val = np.mean(stats2['val_areas'])
    print(f"  Val:   {name1}={mean1_val:.4f}, {name2}={mean2_val:.4f}, "
          f"差异={mean2_val - mean1_val:+.4f} ({100*(mean2_val-mean1_val)/mean1_val:+.1f}%)")

    print(f"\n【每图标注数差异】")
    mean1_train_annots = np.mean(stats1['train_annots_per_img'])
    mean2_train_annots = np.mean(stats2['train_annots_per_img'])
    print(f"  Train: {name1}={mean1_train_annots:.2f}, {name2}={mean2_train_annots:.2f}, "
          f"差异={mean2_train_annots - mean1_train_annots:+.2f} "
          f"({100*(mean2_train_annots-mean1_train_annots)/mean1_train_annots:+.1f}%)")

    mean1_val_annots = np.mean(stats1['val_annots_per_img'])
    mean2_val_annots = np.mean(stats2['val_annots_per_img'])
    print(f"  Val:   {name1}={mean1_val_annots:.2f}, {name2}={mean2_val_annots:.2f}, "
          f"差异={mean2_val_annots - mean1_val_annots:+.2f} "
          f"({100*(mean2_val_annots-mean1_val_annots)/mean1_val_annots:+.1f}%)")

    print(f"\n【关键发现】")
    # 找出最大的类别数量差异
    max_cat_diff = 0
    max_cat_diff_name = None
    for cat_id in set(stats1['val_cats'].keys()) | set(stats2['val_cats'].keys()):
        count1 = stats1['val_cats'].get(cat_id, 0)
        count2 = stats2['val_cats'].get(cat_id, 0)
        if count1 > 0:
            pct_diff = abs(100 * (count2 - count1) / count1)
            if pct_diff > max_cat_diff:
                max_cat_diff = pct_diff
                max_cat_diff_name = category_names.get(cat_id, f'Class{cat_id}')

    if max_cat_diff_name:
        print(f"  * 验证集中 {max_cat_diff_name} 类别差异最大: {max_cat_diff:.1f}%")

    val_annots_diff_pct = abs(100 * (mean2_val_annots - mean1_val_annots) / mean1_val_annots)
    if val_annots_diff_pct > 10:
        print(f"  * 验证集每图标注数差异显著: {val_annots_diff_pct:.1f}%")
        print(f"    -> 这可能导致评估指标的显著差异")

    # 检查小目标比例差异
    small_threshold = 0.01  # 小于图像面积1%的算小目标
    small1_train = sum(1 for a in stats1['train_areas'] if a < small_threshold)
    small2_train = sum(1 for a in stats2['train_areas'] if a < small_threshold)
    small_pct1_train = 100 * small1_train / len(stats1['train_areas'])
    small_pct2_train = 100 * small2_train / len(stats2['train_areas'])

    print(f"  * 小目标比例(<1%图像面积):")
    print(f"    Train: {name1}={small_pct1_train:.1f}%, {name2}={small_pct2_train:.1f}%")

    if abs(small_pct2_train - small_pct1_train) > 5:
        print(f"    -> 小目标比例差异>{5}%，可能影响检测性能")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    # 分析两个数据集
    stats_957 = analyze_dataset(
        "957",
        project_root / "coco/All_Diseases_957/train.json",
        project_root / "coco/All_Diseases_957/val.json"
    )

    stats_957n = analyze_dataset(
        "957n",
        project_root / "coco/All_Diseases_957n/train.json",
        project_root / "coco/All_Diseases_957n/val.json"
    )

    # 对比分析
    compare_two_datasets(stats_957, stats_957n, "957", "957n")

    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}\n")
