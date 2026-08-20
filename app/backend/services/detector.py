from __future__ import annotations

from pathlib import Path

from ..settings import PROJECT_ROOT, RESULTS_DIR


class DetectionService:
    """Lazy-loads the existing detector once and keeps web concerns out of it."""

    def __init__(self) -> None:
        self._detector = None

    def analyze(self, image_path: Path) -> dict:
        if self._detector is None:
            from new_dinoyolo_src.tools.tooth_detector_tool import SimpleToothDetector

            checkpoint = (
                PROJECT_ROOT
                / "res_checkpoints"
                / "multi_disease_Sonata_expt_v3_1"
                / "best_map.pth"
            )
            self._detector = SimpleToothDetector(checkpoint)

        result_dir = RESULTS_DIR / image_path.stem
        result = self._detector.process_image(image_path, output_dir=result_dir)
        annotated_path = Path(result["annotated_image"])
        return {
            "total_count": result["total_count"],
            "report": result["report"],
            "detections": result["detections"],
            "annotated_image": annotated_path,
        }


detector_service = DetectionService()
