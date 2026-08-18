"""
扫描置信度阈值，寻找 F1/mAP 的最佳权衡（new_dinoyolo_src 架构专用）。

用法：所有配置都已写在文件首部「配置区」，无需命令行参数。
    cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM" && \
    C:/ProgramData/anaconda3/envs/dino_VLM/python.exe new_dinoyolo_src/utils/threshold_sweep.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from tqdm import tqdm

from model.yolov10_dinov3 import build_model
from data.model_data import build_dataloaders, infer_num_classes
from train_detector_405YOLO import Config


# ============================================================
#  配置区：请直接在此修改，无需命令行参数
# ============================================================

# 待评估的模型权重（.pth 文件，相对仓库根目录 dinoV3_ToothVLM）
CHECKPOINT = "res_checkpoints/multi_disease_957_expt_v3_1_adaptive_low_threshold/best_map.pth"

# 结果输出 JSON 路径（相对仓库根目录 dinoV3_ToothVLM）
OUTPUT_JSON = "new_dinoyolo_src/utils/threshold_sweep_results——.json"

# 预测与 GT 匹配时的 IoU 阈值
IOU_MATCH = 0.5

# True 时使用 0.01 步长的精细扫描（更慢但更精确）；False 使用下方 UNIFORM_SWEEP 粗扫描
FINE_GRAIN = False

# 推理时用于收集「所有」候选预测的最低置信度（尽量低，阈值扫描在此基础上进行）
INFERENCE_CONF_THRESHOLD = 0.001

# 类别名（顺序与类别 ID 0..N-1 对应）
CLASS_NAMES = ["caries", "calculus", "mouth_ulcer", "tooth_discoloration"]

# 当前基线阈值（来自 Config.VAL_CLASS_THRESHOLDS；缺失的类别用 VAL_CONF_THRESHOLD_DEFAULT）
BASE_THRESHOLDS = {
    0: Config.VAL_CLASS_THRESHOLDS.get(0, Config.VAL_CONF_THRESHOLD_DEFAULT),  # Caries
    1: Config.VAL_CLASS_THRESHOLDS.get(1, Config.VAL_CONF_THRESHOLD_DEFAULT),  # Calculus
    2: Config.VAL_CLASS_THRESHOLDS.get(2, Config.VAL_CONF_THRESHOLD_DEFAULT),  # Mouth_Ulcer
    3: Config.VAL_CLASS_THRESHOLDS.get(3, Config.VAL_CONF_THRESHOLD_DEFAULT),  # Tooth_Discoloration
}

# 粗粒度统一扫描阈值（FINE_GRAIN=False 时使用）
UNIFORM_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# 各类别定向扫描（依据混淆矩阵的漏检率分析，尝试更低阈值）
CARIES_SWEEP = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]   # class 0 漏检率 32.8%
CALCULUS_SWEEP = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]  # class 1 漏检率 41.3%
ULCER_SWEEP = [0.20, 0.22, 0.25, 0.27, 0.30, 0.32, 0.35]    # class 2 漏检率 30%
DISCOLOR_SWEEP = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]  # class 3 漏检率 32.3%

# 网格搜索范围（围绕有希望的区间）
GRID_RANGES = {
    0: [0.25, 0.28, 0.30],          # Caries
    1: [0.20, 0.22, 0.25, 0.28],    # Calculus（更低区间）
    2: [0.25, 0.27, 0.30],          # Mouth_Ulcer
    3: [0.24, 0.26, 0.28],          # Tooth_Discoloration
}

# 每种目标各打印前 N 名配置
TOP_K = 10


def calculate_metrics_at_thresholds(predictions, targets, class_thresholds, iou_threshold=0.5):
    """Calculate per-class TP/FP/FN at given confidence thresholds.

    Args:
        predictions: List of tensors, each [N, 6] (x1, y1, x2, y2, conf, class_id)
        targets: List of dicts with 'boxes' [M, 4] and 'labels' [M]
        class_thresholds: Dict mapping class_id (0-based) to confidence threshold
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        per_class_stats: Dict[class_id] -> {'tp': int, 'fp': int, 'fn': int}
    """
    num_classes = len(class_thresholds)
    stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in range(num_classes)}

    for pred, target in zip(predictions, targets):
        gt_boxes = target['boxes'].cpu()
        gt_labels = target['labels'].cpu()

        # Filter predictions by class-specific thresholds
        filtered_pred = []
        for detection in pred:
            x1, y1, x2, y2, conf, cls_id = detection.tolist()
            cls_id = int(cls_id)
            threshold = class_thresholds.get(cls_id, 0.3)
            if conf >= threshold:
                filtered_pred.append([x1, y1, x2, y2, conf, cls_id])

        if not filtered_pred:
            filtered_pred = torch.empty(0, 6)
        else:
            filtered_pred = torch.tensor(filtered_pred)

        # Match predictions to ground truth (one-to-one, greedy by confidence)
        matched_gt = set()

        for detection in filtered_pred:
            x1, y1, x2, y2, conf, cls_id = detection.tolist()
            cls_id = int(cls_id)
            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in matched_gt:
                    continue
                # Only match same class
                if int(gt_label) != cls_id + 1:  # GT labels are 1-indexed
                    continue

                gx1, gy1, gx2, gy2 = gt_box.tolist()
                inter_w = max(0, min(x2, gx2) - max(x1, gx1))
                inter_h = max(0, min(y2, gy2) - max(y1, gy1))
                inter_area = inter_w * inter_h
                pred_area = max(0, x2 - x1) * max(0, y2 - y1)
                gt_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                union_area = pred_area + gt_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                stats[cls_id]['tp'] += 1
            else:
                stats[cls_id]['fp'] += 1

        # Count false negatives (unmatched ground truth)
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                cls_id = int(gt_label) - 1  # Convert 1-indexed to 0-indexed
                if cls_id in stats:
                    stats[cls_id]['fn'] += 1

    return stats


def compute_f1_and_map(stats):
    """Compute per-class F1, precision, recall, and overall metrics."""
    results = {}
    aps = []
    all_tp, all_fp, all_fn = 0, 0, 0

    for cls_id, counts in stats.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']

        all_tp += tp
        all_fp += fp
        all_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AP approximation: precision at this threshold (simplified, not true AP curve)
        ap = precision
        aps.append(ap)

        results[cls_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    # Overall metrics
    results['mAP'] = sum(aps) / len(aps) if aps else 0.0
    results['mean_f1'] = sum(r['f1'] for r in results.values() if isinstance(r, dict)) / len(stats)

    # Global precision/recall/F1 (across all classes)
    results['global_precision'] = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    results['global_recall'] = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    results['global_f1'] = (
        2 * results['global_precision'] * results['global_recall'] /
        (results['global_precision'] + results['global_recall'])
        if (results['global_precision'] + results['global_recall']) > 0 else 0.0
    )

    return results


def run_inference_once(model, val_loader, device):
    """Run inference once and cache all predictions with minimal confidence threshold."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Running inference"):
            images = images.to(device, non_blocking=True)
            # Use minimal threshold to get all predictions
            outputs = model(images, conf_threshold=INFERENCE_CONF_THRESHOLD)

            # outputs is a list of tensors, each [N, 6]
            for pred in outputs:
                all_predictions.append(pred.cpu())
            all_targets.extend(targets)

    return all_predictions, all_targets


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model from {CHECKPOINT}")

    # Load model
    checkpoint_path = Path(CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Infer number of classes from training data
    project_root = Path(__file__).resolve().parent.parent.parent
    num_classes = infer_num_classes(project_root / Config.TRAIN_JSON)
    print(f"Number of classes: {num_classes}")

    model = build_model(num_classes=num_classes, config=Config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle potential missing keys (e.g., class_weights)
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"⚠️  Load state dict warnings:")
        if load_result.missing_keys:
            print(f"   Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"   Unexpected keys: {load_result.unexpected_keys}")

    model.to(device)
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load validation data
    _, val_loader = build_dataloaders(config=Config)
    print(f"Validation set: {len(val_loader.dataset)} images")

    # Run inference once and cache predictions
    print("\n=== Step 1: Running inference (one-time) ===")
    all_predictions, all_targets = run_inference_once(model, val_loader, device)
    print(f"✓ Cached {len(all_predictions)} predictions")

    # Define threshold sweep strategies
    print("\n=== Step 2: Sweeping thresholds ===")

    if FINE_GRAIN:
        print("Using fine-grained sweep (0.01 steps)")
        uniform_sweep = [round(t, 2) for t in torch.arange(0.10, 0.51, 0.01).tolist()]
    else:
        uniform_sweep = UNIFORM_SWEEP

    # Current baseline from top-of-file config
    current_baseline = dict(BASE_THRESHOLDS)
    print(f"Current baseline thresholds: {current_baseline}")

    # Strategy 1: Uniform thresholds
    uniform_configs = [(f'uniform_{t:.2f}', {0: t, 1: t, 2: t, 3: t}) for t in uniform_sweep]

    # Strategy 2: Per-class focused sweeps (based on current confusion matrix analysis)
    # Calculus has 41.3% miss rate → try lower thresholds
    calculus_focused = [
        (f'calculus_{t:.2f}', {**BASE_THRESHOLDS, 1: t})
        for t in CALCULUS_SWEEP
    ]

    # Caries has 32.8% miss rate → try lower thresholds
    caries_focused = [
        (f'caries_{t:.2f}', {**BASE_THRESHOLDS, 0: t})
        for t in CARIES_SWEEP
    ]

    # Mouth_Ulcer has 30% miss rate (already improved from 50%) → fine-tune
    ulcer_focused = [
        (f'ulcer_{t:.2f}', {**BASE_THRESHOLDS, 2: t})
        for t in ULCER_SWEEP
    ]

    # Tooth_Discoloration has 32.3% miss rate → try lower thresholds
    discolor_focused = [
        (f'discolor_{t:.2f}', {**BASE_THRESHOLDS, 3: t})
        for t in DISCOLOR_SWEEP
    ]

    # Strategy 3: Grid search around promising regions
    # Based on analysis: need to lower thresholds for most classes
    grid_search = []
    for c0 in GRID_RANGES[0]:  # Caries
        for c1 in GRID_RANGES[1]:  # Calculus (lower range)
            for c2 in GRID_RANGES[2]:  # Mouth_Ulcer
                for c3 in GRID_RANGES[3]:  # Tooth_Discoloration
                    grid_search.append({0: c0, 1: c1, 2: c2, 3: c3})

    all_strategies = {
        'current_baseline': [('current', current_baseline)],
        'uniform': uniform_configs,
        'calculus_focused': calculus_focused,
        'caries_focused': caries_focused,
        'ulcer_focused': ulcer_focused,
        'discolor_focused': discolor_focused,
        'grid_search': [(f'grid_{i:03d}', c) for i, c in enumerate(grid_search)],
    }

    print(f"Total configurations to test: {sum(len(configs) for configs in all_strategies.values())}")

    results = {}

    for strategy_name, configs in all_strategies.items():
        print(f"\n--- Strategy: {strategy_name} ({len(configs)} configs) ---")
        strategy_results = []

        for config_name, thresholds in tqdm(configs, desc=strategy_name):
            stats = calculate_metrics_at_thresholds(
                all_predictions, all_targets, thresholds, iou_threshold=IOU_MATCH
            )
            metrics = compute_f1_and_map(stats)

            strategy_results.append({
                'name': config_name,
                'thresholds': thresholds,
                'metrics': metrics,
            })

        results[strategy_name] = strategy_results

    # Find best configurations
    print("\n=== Step 3: Finding optimal thresholds ===")

    all_configs = []
    for strategy_results in results.values():
        all_configs.extend(strategy_results)

    # Sort by different objectives
    by_global_f1 = sorted(all_configs, key=lambda x: x['metrics']['global_f1'], reverse=True)[:10]
    by_mean_f1 = sorted(all_configs, key=lambda x: x['metrics']['mean_f1'], reverse=True)[:10]
    by_map = sorted(all_configs, key=lambda x: x['metrics']['mAP'], reverse=True)[:10]

    # Balanced: global_F1 × mAP
    by_balanced = sorted(
        all_configs,
        key=lambda x: x['metrics']['global_f1'] * x['metrics']['mAP'],
        reverse=True
    )[:10]

    # Recall-focused: maximize global recall
    by_recall = sorted(
        all_configs,
        key=lambda x: x['metrics']['global_recall'],
        reverse=True
    )[:10]

    # Precision-focused: maximize global precision
    by_precision = sorted(
        all_configs,
        key=lambda x: x['metrics']['global_precision'],
        reverse=True
    )[:10]

    print("\n🏆 Top 10 by Global F1 (overall performance):")
    for rank, cfg in enumerate(by_global_f1, 1):
        m = cfg['metrics']
        print(f"{rank}. {cfg['name']}: "
              f"F1={m['global_f1']:.4f} (P={m['global_precision']:.3f}, R={m['global_recall']:.3f}), "
              f"mAP={m['mAP']:.3f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 10 by Mean F1 (per-class average):")
    for rank, cfg in enumerate(by_mean_f1, 1):
        m = cfg['metrics']
        print(f"{rank}. {cfg['name']}: "
              f"Mean F1={m['mean_f1']:.4f}, Global F1={m['global_f1']:.4f}, "
              f"mAP={m['mAP']:.3f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 10 by Balanced (Global F1 × mAP):")
    for rank, cfg in enumerate(by_balanced, 1):
        m = cfg['metrics']
        score = m['global_f1'] * m['mAP']
        print(f"{rank}. {cfg['name']}: "
              f"Score={score:.4f} (F1={m['global_f1']:.4f}, mAP={m['mAP']:.3f})")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 10 by Recall (minimize false negatives):")
    for rank, cfg in enumerate(by_recall, 1):
        m = cfg['metrics']
        print(f"{rank}. {cfg['name']}: "
              f"Recall={m['global_recall']:.4f}, Precision={m['global_precision']:.3f}, "
              f"F1={m['global_f1']:.4f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 10 by Precision (minimize false positives):")
    for rank, cfg in enumerate(by_precision, 1):
        m = cfg['metrics']
        print(f"{rank}. {cfg['name']}: "
              f"Precision={m['global_precision']:.4f}, Recall={m['global_recall']:.3f}, "
              f"F1={m['global_f1']:.4f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    # Save detailed results
    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'checkpoint': str(checkpoint_path.resolve()),
        'iou_threshold': IOU_MATCH,
        'current_baseline': current_baseline,
        'all_results': results,
        'top_by_global_f1': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_global_f1
        ],
        'top_by_mean_f1': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_mean_f1
        ],
        'top_by_map': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_map
        ],
        'top_by_balanced': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_balanced
        ],
        'top_by_recall': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_recall
        ],
        'top_by_precision': [
            {'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']}
            for c in by_precision
        ],
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved detailed results to {output_path.resolve()}")

    # Print per-class breakdown for best balanced config
    best_config = by_balanced[0]
    print(f"\n=== Per-class breakdown for best balanced config: {best_config['name']} ===")
    print(f"Thresholds: {best_config['thresholds']}")
    print(f"Overall: F1={best_config['metrics']['global_f1']:.4f}, "
          f"P={best_config['metrics']['global_precision']:.3f}, "
          f"R={best_config['metrics']['global_recall']:.3f}, "
          f"mAP={best_config['metrics']['mAP']:.3f}")
    print()

    class_names = CLASS_NAMES
    for cls_id in range(len(CLASS_NAMES)):
        m = best_config['metrics'][cls_id]
        thresh = best_config['thresholds'][cls_id]
        print(f"{class_names[cls_id]:20s} (thresh={thresh:.2f}): "
              f"P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f} | "
              f"TP={m['tp']:3d}, FP={m['fp']:3d}, FN={m['fn']:3d}")

    print("\n✓ Threshold sweep complete!")


if __name__ == '__main__':
    main()
