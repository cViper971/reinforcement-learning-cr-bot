"""GameWrapper: single interface to the running Clash Royale match."""
import os
import time

from .capture import ScreenCapture
from .detector import TroopDetector
from .game_state import extract_state
from .interact import play_card, reset_match

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "best.pt")
TARGET_MONITOR = 1
CROP_REGION = {'left': 655, 'top': 0, 'width': 608, 'height': 1080}

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

    def reset_match(self, **kwargs) -> None:
        reset_match(self.get_state, **kwargs)

    def close(self) -> None:
        self._cap.stop()
