"""
数据增强消融实验配置

基于用户反馈："之前的mosaic等对测试确实是有正面作用的"
系统测试不同的数据增强组合，找出最优配置。

用法：
    # 运行baseline
    python new_dinoyolo_src/train_detector_405YOLO.py --config baseline

    # 运行Mosaic实验
    python new_dinoyolo_src/train_detector_405YOLO.py --config mosaic_light
"""

# ============================================================
# Baseline: 当前配置（无高级增强）
# ============================================================
BASELINE = {
    "name": "baseline",
    "description": "无Mosaic/Copy-Paste，只有基础增强",
    "MOSAIC_PROB": 0.0,
    "COPY_PASTE_PROB": 0.0,
    "expected_f1": 0.63,  # 基于v2在957n上的结果
}

# ============================================================
# 实验A: 轻度Mosaic（30%概率）
# ============================================================
MOSAIC_LIGHT = {
    "name": "mosaic_light",
    "description": "轻度Mosaic增强，避免过度合成",
    "MOSAIC_PROB": 0.3,
    "COPY_PASTE_PROB": 0.0,
    "expected_f1": 0.66,
    "rationale": "牙科图像背景简单，Mosaic不会引入伪影。30%概率平衡真实样本和合成样本。",
}

# ============================================================
# 实验B: 标准Mosaic（50%概率）
# ============================================================
MOSAIC_STANDARD = {
    "name": "mosaic_standard",
    "description": "标准Mosaic增强（YOLOv5/v10默认配置）",
    "MOSAIC_PROB": 0.5,
    "COPY_PASTE_PROB": 0.0,
    "expected_f1": 0.68,
    "rationale": "50%是YOLO系列的经典配置，在多个数据集上验证有效。",
}

# ============================================================
# 实验C: Mosaic + Copy-Paste
# ============================================================
MOSAIC_COPYPASTE = {
    "name": "mosaic_copypaste",
    "description": "Mosaic + Copy-Paste组合增强",
    "MOSAIC_PROB": 0.5,
    "COPY_PASTE_PROB": 0.2,
    "expected_f1": 0.70,
    "rationale": "Copy-Paste可以增加稀有类别（如Mouth_Ulcer）的样本多样性。",
}

# ============================================================
# 实验D: 激进Mosaic（70%概率）
# ============================================================
MOSAIC_AGGRESSIVE = {
    "name": "mosaic_aggressive",
    "description": "激进Mosaic增强，用于测试上限",
    "MOSAIC_PROB": 0.7,
    "COPY_PASTE_PROB": 0.3,
    "expected_f1": 0.68,  # 可能过拟合到合成样本
    "rationale": "测试高增强率是否会导致性能下降（过度依赖合成样本）。",
}

# ============================================================
# 实验E: 仅Copy-Paste（针对稀有类）
# ============================================================
COPYPASTE_ONLY = {
    "name": "copypaste_only",
    "description": "仅Copy-Paste，针对Mouth_Ulcer的高漏检率",
    "MOSAIC_PROB": 0.0,
    "COPY_PASTE_PROB": 0.3,
    "expected_f1": 0.65,
    "rationale": "Copy-Paste可以在不改变图像全局结构的情况下增加小目标样本。",
}

# ============================================================
# 实验配置映射
# ============================================================
CONFIGS = {
    "baseline": BASELINE,
    "mosaic_light": MOSAIC_LIGHT,
    "mosaic_standard": MOSAIC_STANDARD,
    "mosaic_copypaste": MOSAIC_COPYPASTE,
    "mosaic_aggressive": MOSAIC_AGGRESSIVE,
    "copypaste_only": COPYPASTE_ONLY,
}


def get_config(name):
    """获取指定的实验配置"""
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Available: {available}")
    return CONFIGS[name]


def print_experiment_plan():
    """打印完整的实验计划"""
    print("\n" + "="*70)
    print("数据增强消融实验计划")
    print("="*70)
    print("\n【目标】找出在957n数据集上表现最优的数据增强配置\n")

    print("【实验组】")
    for i, (name, cfg) in enumerate(CONFIGS.items(), 1):
        print(f"\n{i}. {cfg['name']}")
        print(f"   描述: {cfg['description']}")
        print(f"   配置: MOSAIC_PROB={cfg['MOSAIC_PROB']}, COPY_PASTE_PROB={cfg['COPY_PASTE_PROB']}")
        print(f"   预期F1: {cfg['expected_f1']}")
        print(f"   原理: {cfg['rationale']}")

    print("\n【实验流程】")
    print("1. 统一数据集: 957n (train=760, val=191)")
    print("2. 每个配置训练2次，使用不同随机种子 (seed=42, 43)")
    print("3. 记录每次的 mAP@.5, F1, Precision, Recall")
    print("4. 对比分析，选出最优配置")

    print("\n【预计时间】")
    print("- 每次训练: ~1.5小时")
    print("- 总实验: 6组 × 2次 = 12次训练 ≈ 18小时")

    print("\n【评估指标】")
    print("- 主指标: F1 score (Precision和Recall的调和平均)")
    print("- 辅助指标: mAP@.5, 各类别Precision/Recall")
    print("- 稳定性: 2次训练的方差（方差<0.01为可接受）")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_experiment_plan()

    # 示例：获取mosaic_standard配置
    cfg = get_config("mosaic_standard")
    print(f"\n示例：获取 '{cfg['name']}' 配置")
    print(f"  MOSAIC_PROB = {cfg['MOSAIC_PROB']}")
    print(f"  COPY_PASTE_PROB = {cfg['COPY_PASTE_PROB']}")
