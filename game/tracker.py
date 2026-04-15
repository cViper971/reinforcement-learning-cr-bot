"""
Troop tracker — adds team (ally/enemy) and persistent IDs to YOLO detections.

Team assignment:
  1. On first sight, team inferred from spawn side of the arena (y vs midline).
  2. Once assigned, team sticks — crossing the river doesn't flip it.

Tracking:
  IOU-based greedy matching between consecutive frames.
"""

from dataclasses import dataclass, field
import numpy as np
from .detector import TroopDetector, Detection


# River midline in CROP_REGION coordinates (608 x 1080 frame).
# Enemy king tower is ~y=30, my king is ~y=820 — river sits around y=490.
MIDLINE_Y = 490

# Troop is dropped after this many frames without a match
MAX_MISSED_FRAMES = 8

# Minimum IOU for a detection to match an existing tracked troop
IOU_THRESHOLD = 0.3


@dataclass
class TrackedTroop:
    id: int
    name: str
    team: str                              # "ally" | "enemy"
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    age: int = 0                           # frames since first seen
    missed: int = 0                        # consecutive frames without a match

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "team": self.team,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": self.confidence,
            "age": self.age,
        }


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _infer_team(center_y: int, name: str) -> str:
    """Spawn-side heuristic: top half = enemy, bottom half = ally.

    Edge cases (Miner, Goblin Barrel, Graveyard, spells) can spawn on
    either side — they'll be misclassified ~some of the time. Good enough
    for MVP; refine with health-bar color later.
    """
    return "enemy" if center_y < MIDLINE_Y else "ally"


class TroopTracker:
    def __init__(self, detector: TroopDetector):
        self.detector = detector
        self._tracked: list[TrackedTroop] = []
        self._next_id: int = 0

    def update(self, frame: np.ndarray) -> list[TrackedTroop]:
        detections = self.detector.detect(frame)

        # Greedy IOU matching: pair each detection with best tracked troop of same class
        matched_tracked: set[int] = set()
        matched_det: set[int] = set()

        # Build candidate (iou, det_idx, tracked_idx) list and sort desc
        candidates: list[tuple[float, int, int]] = []
        for di, d in enumerate(detections):
            for ti, t in enumerate(self._tracked):
                if t.name != d.name:
                    continue
                iou = _iou(d.bbox, t.bbox)
                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, di, ti))
        candidates.sort(reverse=True)

        for _iou_score, di, ti in candidates:
            if di in matched_det or ti in matched_tracked:
                continue
            matched_det.add(di)
            matched_tracked.add(ti)
            t = self._tracked[ti]
            d = detections[di]
            t.bbox = d.bbox
            t.center = d.center
            t.confidence = d.confidence
            t.age += 1
            t.missed = 0

        # Unmatched detections → new tracked troops
        for di, d in enumerate(detections):
            if di in matched_det:
                continue
            team = _infer_team(d.center[1], d.name)
            self._tracked.append(TrackedTroop(
                id=self._next_id,
                name=d.name,
                team=team,
                bbox=d.bbox,
                center=d.center,
                confidence=d.confidence,
            ))
            self._next_id += 1

        # Unmatched tracked → increment missed; drop if too many
        for ti, t in enumerate(self._tracked):
            if ti not in matched_tracked:
                t.missed += 1
        self._tracked = [t for t in self._tracked if t.missed <= MAX_MISSED_FRAMES]

        # Only return currently-visible troops (missed == 0)
        return [t for t in self._tracked if t.missed == 0]
