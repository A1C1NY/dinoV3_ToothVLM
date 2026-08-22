from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..settings import PROJECT_ROOT


class PeriodontalService:
    """Lazy-loaded periodontal image classifier used alongside the detector."""

    def __init__(self) -> None:
        self._model = None
        self._class_names: list[str] = []
        self._img_size = 224
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()

    def _load(self) -> None:
        checkpoint = PROJECT_ROOT / "res_checkpoints" / "best_val_acc.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Periodontal checkpoint not found: {checkpoint}")

        # Reuse the exact model construction and preprocessing from the training code.
        from new_dinoyolo_src.infer_classifier_periodontal import load_checkpoint

        model, class_names, img_size = load_checkpoint(checkpoint)
        self._model = model.to(self._device).eval()
        self._class_names = list(class_names)
        self._img_size = img_size

    def analyze(self, image_path: Path) -> dict:
        with self._lock:
            if self._model is None:
                self._load()

            transform = transforms.Compose([
                transforms.Resize(int(self._img_size * 256 / 224)),
                transforms.CenterCrop(self._img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode():
                probabilities = F.softmax(self._model(transform(image).unsqueeze(0).to(self._device)), dim=1)[0]
            ranked = sorted(
                ((name, float(probabilities[index])) for index, name in enumerate(self._class_names)),
                key=lambda item: item[1], reverse=True,
            )
            predicted, confidence = ranked[0]
            positive = predicted.lower() == "periodontitis"
            report = (
                f"牙周炎分类：**{predicted}**（置信度 {confidence:.1%}）。\n"
                f"判定：{'疑似存在牙周炎' if positive else '未检出牙周炎类别'}。\n"
                "此结果仅供辅助筛查，请由牙周科医生结合临床检查确认。"
            )
            return {
                "tool": "periodontal_classifier",
                "prediction": predicted,
                "periodontitis": positive,
                "confidence": confidence,
                "probabilities": {name: probability for name, probability in ranked},
                "report": report,
            }


periodontal_service = PeriodontalService()
