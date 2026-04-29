"""GameWrapper: single interface to the running Clash Royale match."""
import os
import time

import pydirectinput

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

    def reset_match(self, settle_seconds: float = 5.0, poll_dt: float = 0.25,
                    flicker_grace: int = 3) -> None:
        """Wait until post-match (game_state != 0) holds for `settle_seconds`,
        then press BlueStacks '1' (bound to Play Again). Tolerates up to
        `flicker_grace` consecutive zero readings before treating it as a
        real return-to-running (perception's banner score can dip during
        fade animations)."""
        zero_streak = 0
        stable_since: float | None = None
        while True:
            gs = self.get_state()["game_state"]
            if gs != 0:
                zero_streak = 0
                if stable_since is None:
                    stable_since = time.perf_counter()
                if time.perf_counter() - stable_since >= settle_seconds:
                    pydirectinput.press('1')
                    return
            else:
                if stable_since is None:
                    return  # never entered post-match — match still running
                zero_streak += 1
                if zero_streak >= flicker_grace:
                    return  # genuine return to running before settle — give up
            time.sleep(poll_dt)

    def close(self) -> None:
        self._cap.stop()
