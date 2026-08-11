"""
Sweep confidence thresholds to find optimal F1/mAP trade-off.

example usage:
cd "d:/File/Programming/Tooth_VLM/dinoV3_ToothVLM" && C:/ProgramData/anaconda3/envs/dino_VLM/python.exe src/threshold_sweep.py --checkpoint "res_checkpoints/multi_disease_562_expt_v2_adaptive/best_map.pth" --output "threshold_sweep_results.json"



"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from train_detector_405YOLO import Config, build_dataloaders, build_model


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
    """Compute per-class F1, precision, recall, and mAP."""
    results = {}
    aps = []

    for cls_id, counts in stats.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']

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

    results['mAP'] = sum(aps) / len(aps) if aps else 0.0
    results['mean_f1'] = sum(r['f1'] for r in results.values() if isinstance(r, dict)) / len(stats)

    return results


def run_inference_once(model, val_loader, device):
    """Run inference once and cache all predictions."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Running inference"):
            images = images.to(device)
            outputs = model(images)

            # outputs is a list of tensors, each [N, 6]
            for pred in outputs:
                all_predictions.append(pred.cpu())
            all_targets.extend(targets)

    return all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--output', type=str, default='threshold_sweep_results.json', help='Output JSON file')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint}")

    # Load model
    model = build_model(num_classes=4)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load validation data
    _, val_loader = build_dataloaders()
    print(f"Validation set: {len(val_loader.dataset)} images")

    # Run inference once and cache predictions
    print("\n=== Step 1: Running inference (one-time) ===")
    all_predictions, all_targets = run_inference_once(model, val_loader, device)
    print(f"Cached {len(all_predictions)} predictions")

    # Define threshold sweep strategies
    print("\n=== Step 2: Sweeping thresholds ===")

    # Strategy 1: Uniform thresholds (baseline)
    uniform_sweep = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # Strategy 2: Per-class based on confusion matrix analysis
    # Current issues: calculus (50% recall), caries (61% recall) need lower thresholds
    # mouth_ulcer (88% recall), tooth_discoloration (81% recall) are good

    # Start from current adaptive thresholds
    current_adaptive = {0: 0.28, 1: 0.25, 2: 0.18, 3: 0.28}

    # Strategy 2a: Lower calculus threshold more aggressively
    calculus_focused = [
        {0: 0.28, 1: t, 2: 0.18, 3: 0.28} for t in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    ]

    # Strategy 2b: Lower caries threshold
    caries_focused = [
        {0: t, 1: 0.25, 2: 0.18, 3: 0.28} for t in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]
    ]

    # Strategy 2c: Raise mouth_ulcer threshold (reduce FP, already high recall)
    ulcer_focused = [
        {0: 0.28, 1: 0.25, 2: t, 3: 0.28} for t in [0.18, 0.22, 0.25, 0.28, 0.30]
    ]

    # Strategy 2d: Grid search around current adaptive
    grid_search = []
    for c0 in [0.22, 0.25, 0.28, 0.30]:
        for c1 in [0.18, 0.22, 0.25, 0.28]:
            for c2 in [0.15, 0.18, 0.20]:
                for c3 in [0.25, 0.28, 0.30]:
                    grid_search.append({0: c0, 1: c1, 2: c2, 3: c3})

    all_strategies = {
        'uniform': [(f'all_{t:.2f}', {0: t, 1: t, 2: t, 3: t}) for t in uniform_sweep],
        'calculus_focused': [(f'calc_{c[1]:.2f}', c) for c in calculus_focused],
        'caries_focused': [(f'caries_{c[0]:.2f}', c) for c in caries_focused],
        'ulcer_focused': [(f'ulcer_{c[2]:.2f}', c) for c in ulcer_focused],
        'grid_search': [(f'grid_{i:03d}', c) for i, c in enumerate(grid_search)],
        'current_adaptive': [('current', current_adaptive)],
    }

    results = {}

    for strategy_name, configs in all_strategies.items():
        print(f"\n--- Strategy: {strategy_name} ({len(configs)} configs) ---")
        strategy_results = []

        for config_name, thresholds in tqdm(configs, desc=strategy_name):
            stats = calculate_metrics_at_thresholds(
                all_predictions, all_targets, thresholds, iou_threshold=args.iou
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
    by_f1 = sorted(all_configs, key=lambda x: x['metrics']['mean_f1'], reverse=True)[:5]
    by_map = sorted(all_configs, key=lambda x: x['metrics']['mAP'], reverse=True)[:5]

    # Balanced: F1 * mAP
    by_balanced = sorted(
        all_configs,
        key=lambda x: x['metrics']['mean_f1'] * x['metrics']['mAP'],
        reverse=True
    )[:5]

    # Recall-focused: minimize FN (sum of all fn)
    by_recall = sorted(
        all_configs,
        key=lambda x: sum(x['metrics'][i]['fn'] for i in range(4)),
    )[:5]

    print("\n🏆 Top 5 by Mean F1:")
    for rank, cfg in enumerate(by_f1, 1):
        print(f"{rank}. {cfg['name']}: F1={cfg['metrics']['mean_f1']:.3f}, mAP={cfg['metrics']['mAP']:.3f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 5 by mAP:")
    for rank, cfg in enumerate(by_map, 1):
        print(f"{rank}. {cfg['name']}: mAP={cfg['metrics']['mAP']:.3f}, F1={cfg['metrics']['mean_f1']:.3f}")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 5 by Balanced (F1 × mAP):")
    for rank, cfg in enumerate(by_balanced, 1):
        score = cfg['metrics']['mean_f1'] * cfg['metrics']['mAP']
        print(f"{rank}. {cfg['name']}: Score={score:.4f} (F1={cfg['metrics']['mean_f1']:.3f}, mAP={cfg['metrics']['mAP']:.3f})")
        print(f"   Thresholds: {cfg['thresholds']}")

    print("\n🏆 Top 5 by Total Recall (min FN):")
    for rank, cfg in enumerate(by_recall, 1):
        total_fn = sum(cfg['metrics'][i]['fn'] for i in range(4))
        total_tp = sum(cfg['metrics'][i]['tp'] for i in range(4))
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        print(f"{rank}. {cfg['name']}: Recall={total_recall:.3f}, FN={total_fn}")
        print(f"   Thresholds: {cfg['thresholds']}")

    # Save detailed results
    output_path = Path(args.output)
    output_data = {
        'all_results': results,
        'top_by_f1': [{'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']} for c in by_f1],
        'top_by_map': [{'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']} for c in by_map],
        'top_by_balanced': [{'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']} for c in by_balanced],
        'top_by_recall': [{'name': c['name'], 'thresholds': c['thresholds'], 'metrics': c['metrics']} for c in by_recall],
    }

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved detailed results to {output_path.resolve()}")

    # Print per-class breakdown for best balanced config
    best_config = by_balanced[0]
    print(f"\n=== Per-class breakdown for best balanced config: {best_config['name']} ===")
    class_names = ['caries', 'calculus', 'mouth_ulcer', 'tooth_discoloration']
    for cls_id in range(4):
        m = best_config['metrics'][cls_id]
        thresh = best_config['thresholds'][cls_id]
        print(f"{class_names[cls_id]} (thresh={thresh:.2f}):")
        print(f"  Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}, F1: {m['f1']:.3f}")
        print(f"  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")


if __name__ == '__main__':
    main()