import mss
import numpy as np
import cv2
import threading
import time

TARGET_MONITOR = 1
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

CROP_REGION = {'left': 655, 'top': 0, 'width': 608, 'height': 1080}


class ScreenCapture:
    """Threaded screen capture — always provides the latest frame."""

    def __init__(self, monitor_index: int, crop: dict | None = None):
        self.crop = crop
        self.monitor_index = monitor_index
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]

            if self.crop:
                region = {
                    "left":   monitor["left"] + self.crop["left"],
                    "top":    monitor["top"]  + self.crop["top"],
                    "width":  self.crop["width"],
                    "height": self.crop["height"],
                }
            else:
                region = monitor

            while self._running:
                t0 = time.perf_counter()

                img = np.array(sct.grab(region))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                with self._lock:
                    self._frame = img

                elapsed = time.perf_counter() - t0
                sleep_for = FRAME_INTERVAL - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    