from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

class Config:
    """配置项只在此处“声明”（类型标注），不赋值。

    实际取值由训练脚本（train_detector_405YOLO.py）继承本类并填入，
    再通过 ``YOLOv10WithDinoV3(..., config=Config)`` 传入模型使用。
    """

    # 路径配置
    REPO_DIR: str
    IMAGE_DIR: str
    TRAIN_JSON: str
    VAL_JSON: str
    SINGLE_CAT_ID: Optional[int]   # None 表示保留 json 中的所有疾病类别（映射为 1~N）
    OUTPUT_DIR: str
    WEIGHTS: str

    # 数据集配置
    DROP_EMPTY: bool               # 是否丢弃没有标注的图片

    # 数据增强：只做几何变换。
    AUG_HFLIP: float               # 水平翻转概率
    AUG_AFFINE: float              # 随机仿射概率（缩放/平移/小角度旋转）
    AUG_SCALE: float               # 缩放抖动幅度：ratio ∈ [1-0.25, 1+0.25]
    AUG_TRANSLATE: float           # 平移幅度，占边长比例
    AUG_ROTATE: float              # 旋转角度上限（度）
    AUG_MIN_BOX_SIZE: float        # 变换后小于该边长（像素）的框丢弃
    AUG_MIN_BOX_KEEP: float        # 变换后保留面积低于原面积该比例的框丢弃
    PAD_VALUE: int                 # letterbox 填充灰度值（YOLO 惯例）

    # Detection augmentation. These are applied to the training set only.
    MOSAIC_PROB: float
    MOSAIC_CENTER_RANGE: Tuple[float, float]
    COPY_PASTE_PROB: float
    COPY_PASTE_MAX_BOX_AREA_RATIO: float
    COPY_PASTE_MAX_OBJECTS: int
    COPY_PASTE_CONTEXT_RATIO: float
    COPY_PASTE_MAX_IOU: float
    OVERSAMPLE_CATEGORY_ID: int
    OVERSAMPLE_FACTOR: float

    # 梯度裁剪。None 表示不裁剪。
    CLIP_GRAD_NORM: Optional[float]

    # 训练超参数
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    BACKBONE_LR: float
    WARMUP_EPOCHS: int
    UNFREEZE_BLOCKS: int
    DEVICE: str

    # 从 ViT 的哪三个 block 取特征构造 P3/P4/P5（升序 = shallow→deep）。
    # 设为 None 则退回旧行为：只用最后一层，三尺度由它重采样派生。
    BACKBONE_OUT_INDICES: Optional[Tuple[int, ...]]

    # 继续训练 (可选)
    RESUME_CHECKPOINT: Optional[str]
    START_EPOCH: int

    # 验证与评估参数
    IOU_THRESHOLD: float           # 用于评估时判断正样本的 IoU 阈值
    SCORE_THRESHOLD: float         # 用于过滤低置信度预测的阈值

    # 模型参数
    MIN_SIZE: int
    MAX_SIZE: int
    NUM_CLASSES: Optional[int]
    CONF_THRESHOLD: float

    # 类别自适应阈值（类别 ID 从 0 开始）
    VAL_CLASS_THRESHOLDS: Dict[int, float]
    VAL_CONF_THRESHOLD_DEFAULT: float  # 不在 VAL_CLASS_THRESHOLDS 中的类别使用此默认阈值

    # Set to a sequence with NUM_CLASSES entries when class reweighting is needed.
    CLASS_WEIGHTS: Optional[Sequence[float]]
    DINO_MEAN: Sequence[float]
    DINO_STD: Sequence[float]
    IMG_SIZE: int
    NUM_WORKERS: int
    SEED: int
