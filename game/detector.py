"""
YOLO troop/object detector wrapper.

Usage:
    from game.detector import TroopDetector

    detector = TroopDetector("assets/models/best.pt")
    detections = detector.detect(frame)
    # detections = [
    #     {"name": "Archers", "confidence": 0.87, "bbox": (x1, y1, x2, y2), "center": (cx, cy)},
    #     ...
    # ]
"""

import os
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[int, int]          # (cx, cy) — useful for tile/grid mapping

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
        }


class TroopDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.35, device: str | None = None):
        """
        Args:
            weights_path: path to the trained YOLO .pt file
            conf_threshold: minimum confidence for a detection to be returned
            device: "cuda", "cpu", or None (auto-detect)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names: dict[int, str] = self.model.names  # e.g. {0: "Archers", 1: "Giant", ...}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame and return a list of Detections."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append(Detection(
                name=self.class_names.get(cls_id, f"class_{cls_id}"),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            ))

        return detections


# --- debug ---
if __name__ == "__main__":
    import cv2
    import time
    from .capture import ScreenCapture, TARGET_MONITOR, CROP_REGION

    WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "best.pt")

    detector = TroopDetector(WEIGHTS)
    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("yolo_debug", cv2.WINDOW_NORMAL)

    while True:
        frame = cap.get_frame()
        if frame is None:
            continue

        detections = detector.detect(frame)

        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d.name} {d.confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("yolo_debug", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
