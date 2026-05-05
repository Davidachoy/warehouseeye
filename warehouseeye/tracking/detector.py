"""Person detection wrapper around RT-DETRv2 from Transformers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from warehouseeye.tracking.types import BoundingBox

logger = logging.getLogger(__name__)


class PersonDetector:
    """Load RT-DETRv2 once and run inference on single frames."""

    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_v2_r50vd",
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

        id2label = getattr(self.model.config, "id2label", {})
        self.person_ids = {int(k) for k, v in id2label.items() if str(v).lower() == "person"}
        if not self.person_ids:
            logger.warning("person_label_missing_in_id2label", extra={"model": model_id})
            self.person_ids = {0}

    def detect(self, frame_path: str | Path) -> list[BoundingBox]:
        """Detect only people in the given frame."""
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        detections: list[BoundingBox] = []
        for score, label_id, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
            if int(label_id.item()) not in self.person_ids:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            detections.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(score.item())))
        return detections


def main() -> None:
    """Simple standalone test entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--model-id", default="PekingU/rtdetr_v2_r50vd")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    boxes = PersonDetector(model_id=args.model_id).detect(args.frame)
    logger.info("detections", extra={"count": len(boxes)})


if __name__ == "__main__":
    main()

