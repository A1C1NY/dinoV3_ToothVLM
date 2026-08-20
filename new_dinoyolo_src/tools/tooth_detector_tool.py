"""
简单的牙齿疾病检测工具，供 LLM 调用。

功能：
1. 加载训练好的模型
2. 对输入图片进行推理
3. 在图片上绘制检测框和标签
4. 返回标注图片 + 文字诊断结果
"""

import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from new_dinoyolo_src.model.yolov10_dinov3 import build_model
from new_dinoyolo_src.config.config import Config


class SimpleToothDetector:
    """简单的牙齿疾病检测器"""

    # 疾病类别配置（与训练时保持一致）
    CATEGORIES = {
        1: {"name": "caries", "display": "龋齿", "color": (0, 255, 0)},
        2: {"name": "calculus", "display": "牙结石", "color": (0, 0, 255)},
        3: {"name": "mouth_ulcer", "display": "口腔溃疡", "color": (255, 165, 0)},
        4: {"name": "tooth_discoloration", "display": "牙齿变色", "color": (0, 255, 255)},
    }

    # 健康建议
    HEALTH_ADVICE = {
        "caries": "建议尽快就诊进行充填治疗，防止龋洞扩大。",
        "calculus": "建议进行专业洗牙，清除牙结石，预防牙周疾病。",
        "mouth_ulcer": "注意口腔卫生，避免刺激性食物，如持续不愈请就医。",
        "tooth_discoloration": "可考虑牙齿美白治疗，建议咨询牙科医生。",
    }

    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化检测器

        Args:
            checkpoint_path: 模型权重路径
            device: 运行设备 (cuda/cpu)
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        # 加载模型
        print(f"正在加载模型: {checkpoint_path}")
        self.model = self._load_model()
        self.model.eval()
        print(f"模型加载完成，运行在: {device}")

    def _load_model(self):
        """加载训练好的模型"""
        # 创建配置（使用评估配置）
        class InferenceConfig(Config):
            REPO_DIR = "."
            IMAGE_DIR = "../Sonata/image"
            TRAIN_JSON = "coco/All_Diseases_Sonata/train.json"
            VAL_JSON = "coco/All_Diseases_Sonata/val.json"
            SINGLE_CAT_ID = None
            OUTPUT_DIR = "inference_output"
            WEIGHTS = "pretrained_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

            DROP_EMPTY = True
            AUG_HFLIP = 0.5
            AUG_AFFINE = 0.7
            AUG_SCALE = 0.25
            AUG_TRANSLATE = 0.10
            AUG_ROTATE = 7.0
            AUG_MIN_BOX_SIZE = 4.0
            AUG_MIN_BOX_KEEP = 0.25
            PAD_VALUE = 114

            MOSAIC_PROB = 0.35
            MOSAIC_CENTER_RANGE = (0.45, 0.55)
            COPY_PASTE_PROB = 0.30
            COPY_PASTE_MAX_BOX_AREA_RATIO = 0.02
            COPY_PASTE_MAX_OBJECTS = 2
            COPY_PASTE_CONTEXT_RATIO = 0.20
            COPY_PASTE_MAX_IOU = 0.10
            OVERSAMPLE_CATEGORY_ID = 3
            OVERSAMPLE_FACTOR = 1.75

            CLIP_GRAD_NORM = 200.0
            BATCH_SIZE = 1
            EPOCHS = 70
            LR = 0.001
            BACKBONE_LR = 0.0001
            WARMUP_EPOCHS = 5
            UNFREEZE_BLOCKS = 6
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

            BACKBONE_OUT_INDICES = (5, 8, 11)
            RESUME_CHECKPOINT = None
            START_EPOCH = 1

            IOU_THRESHOLD = 0.5
            SCORE_THRESHOLD = 0.5
            MIN_SIZE = 1200
            MAX_SIZE = 1200
            NUM_CLASSES = 4
            CONF_THRESHOLD = 0.001

            VAL_CLASS_THRESHOLDS = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30}
            VAL_CONF_THRESHOLD_DEFAULT = 0.3
            CLASS_WEIGHTS = [1.2, 1.3, 2.5, 1.1]

            DINO_MEAN = (0.485, 0.456, 0.406)
            DINO_STD = (0.229, 0.224, 0.225)
            IMG_SIZE = 768
            NUM_WORKERS = 0
            SEED = 42

        # 构建模型
        model = build_model(num_classes=4, config=InferenceConfig).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return model

    def _preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # Letterbox resize (保持宽高比)
        target_size = 768
        ratio = min(target_size / image.width, target_size / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)

        image_resized = image.resize((new_width, new_height), Image.BILINEAR)

        # 创建填充画布
        canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        canvas.paste(image_resized, (pad_x, pad_y))

        # 转换为张量
        tensor = pil_to_tensor(canvas).float() / 255.0

        return tensor.unsqueeze(0), ratio, pad_x, pad_y, original_size

    def detect(self, image_path, confidence_threshold=0.3):
        """
        对图片进行疾病检测

        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值

        Returns:
            dict: 包含检测结果的字典
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 预处理
        tensor, ratio, pad_x, pad_y, original_size = self._preprocess_image(image_path)
        tensor = tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(tensor, conf_threshold=confidence_threshold)

        # 解析结果
        detections = []
        if len(predictions) > 0 and len(predictions[0]) > 0:
            for pred in predictions[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred

                # 转换回原图坐标
                x1_orig = (x1 - pad_x) / ratio
                y1_orig = (y1 - pad_y) / ratio
                x2_orig = (x2 - pad_x) / ratio
                y2_orig = (y2 - pad_y) / ratio

                category_id = int(cls) + 1
                category_info = self.CATEGORIES.get(category_id, {})

                detections.append({
                    "disease": category_info.get("name", "unknown"),
                    "display_name": category_info.get("display", "未知"),
                    "confidence": float(conf),
                    "bbox": [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                    "color": category_info.get("color", (128, 128, 128)),
                })

        return {
            "image_path": str(image_path),
            "detections": detections,
            "total_count": len(detections),
        }

    def draw_results(self, image_path, detections, output_path=None):
        """
        在图片上绘制检测结果

        Args:
            image_path: 原始图片路径
            detections: 检测结果列表
            output_path: 输出路径（可选）

        Returns:
            str: 输出图片路径
        """
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # 尝试加载字体（如果失败使用默认字体）
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_small = font

        # 绘制每个检测框
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 绘制标签背景
            label = f"{det['display_name']} {det['confidence']:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), label, font=font_small)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)

            # 绘制标签文字
            draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font_small)

        # 保存结果
        if output_path is None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_detected.jpg"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)

        return str(output_path)

    def generate_diagnosis_report(self, detections):
        """
        生成诊断报告文字

        Args:
            detections: 检测结果列表

        Returns:
            str: 诊断报告文字
        """
        if not detections:
            return "✅ 未检测到明显的口腔疾病，口腔健康状况良好。建议保持良好的口腔卫生习惯。"

        report = f"📋 **口腔健康诊断报告**\n\n"
        report += f"检测到 **{len(detections)}** 处异常：\n\n"

        # 按疾病类型分组
        disease_groups = {}
        for det in detections:
            disease = det["display_name"]
            if disease not in disease_groups:
                disease_groups[disease] = []
            disease_groups[disease].append(det)

        # 生成每种疾病的报告
        for disease, items in disease_groups.items():
            avg_conf = sum(d["confidence"] for d in items) / len(items)
            report += f"• **{disease}** (检出 {len(items)} 处，平均置信度: {avg_conf:.1%})\n"

            # 添加建议
            disease_key = items[0]["disease"]
            advice = self.HEALTH_ADVICE.get(disease_key, "建议咨询专业牙科医生。")
            report += f"  💡 {advice}\n\n"

        report += "\n⚠️ **注意**: 此诊断仅供参考，请以专业医生的诊断为准。"

        return report

    def process_image(self, image_path, output_dir=None, confidence_threshold=0.3):
        """
        完整的处理流程：检测 + 可视化 + 生成报告

        Args:
            image_path: 输入图片路径
            output_dir: 输出目录（可选）
            confidence_threshold: 置信度阈值

        Returns:
            dict: 包含结果路径和报告的字典
        """
        # 检测
        result = self.detect(image_path, confidence_threshold)

        # 绘制结果
        if output_dir:
            output_path = Path(output_dir) / f"{Path(image_path).stem}_detected.jpg"
        else:
            output_path = None

        annotated_image = self.draw_results(image_path, result["detections"], output_path)

        # 生成报告
        report = self.generate_diagnosis_report(result["detections"])

        return {
            "annotated_image": annotated_image,
            "report": report,
            "detections": result["detections"],
            "total_count": result["total_count"],
        }


# 工具函数：供 LLM Function Calling 使用
def detect_tooth_diseases(image_path: str, confidence_threshold: float = 0.3) -> dict:
    """
    检测牙齿疾病的工具函数，供 LLM 调用

    Args:
        image_path: 图片路径
        confidence_threshold: 置信度阈值 (0-1)

    Returns:
        dict: 包含标注图片路径和诊断报告
    """
    # 使用默认权重路径
    checkpoint_path = Path(__file__).resolve().parents[2] / "res_checkpoints" / "multi_disease_Sonata_expt_v3_1" / "best_map.pth"

    detector = SimpleToothDetector(checkpoint_path)
    result = detector.process_image(image_path, confidence_threshold=confidence_threshold)

    return {
        "status": "success",
        "annotated_image_path": result["annotated_image"],
        "diagnosis_report": result["report"],
        "detections_count": result["total_count"],
    }
