"""GameWrapper: single interface to the running Clash Royale match."""
import os
import time

from .capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
from .detector import TroopDetector
from .perception import extract_state
from .actions import play_card


DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "assets", "models", "best.pt")


class GameWrapper:
    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, conf_threshold: float = 0.2):
        self._cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
        self._detector = TroopDetector(weights_path, conf_threshold=conf_threshold)
        self._cap.start()
        time.sleep(0.3)

    def troop_classes(self) -> list[str]:
        """Class names known to the troop detector (used to build observation vocab)."""
        return list(self._detector.model.names.values())

    def get_state(self) -> dict:
        frame = self._cap.get_frame()
        while frame is None:
            time.sleep(0.01)
            frame = self._cap.get_frame()
        return extract_state(frame, detector=self._detector)

    def act(self, slot: int, col: int, row: int) -> None:
        play_card(slot, col, row)

    def reset_match(self) -> None:
        """Click through end-of-match screens and queue the next battle.
        TODO: implement once the post-match UI templates are calibrated."""
        pass

    def close(self) -> None:
        self._cap.stop()
