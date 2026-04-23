"""Capture endgame banner templates (VICTORY / DEFEAT).

V = save the _ENDGAME_BOXES["victory"] crop as victory.png
D = save the _ENDGAME_BOXES["defeat"] crop as defeat.png
Q = quit

Calibrate each entry in _ENDGAME_BOXES (game/perception.py) first via
scripts/coords.py — the two banners appear in different positions.
"""

import os
import time
import msvcrt
import cv2

from game.capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
from game.perception import _ENDGAME_BOXES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "templates", "endgame")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(frame, name: str) -> None:
    box = _ENDGAME_BOXES.get(name)
    if box is None:
        print(f"  no box defined for {name}")
        return
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        print(f"  {name} box not calibrated in _ENDGAME_BOXES")
        return
    crop = frame[y1:y2, x1:x2]
    path = os.path.join(OUT_DIR, f"{name}.png")
    cv2.imwrite(path, crop)
    print(f"  saved {path}")


cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
cap.start()
time.sleep(0.3)

cv2.namedWindow("endgame", cv2.WINDOW_NORMAL)
print("V = save victory.png | D = save defeat.png | Q = quit\n")

while True:
    frame = cap.get_frame()
    if frame is None:
        continue

    vis = frame.copy()
    for x1, y1, x2, y2 in _ENDGAME_BOXES.values():
        if x2 > x1 and y2 > y1:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow("endgame", vis)
    cv2.waitKey(1)

    if msvcrt.kbhit():
        key = msvcrt.getch().lower()
        if key == b'v':
            _save(frame, "victory")
        elif key == b'd':
            _save(frame, "defeat")
        elif key == b'q':
            break

cap.stop()
cv2.destroyAllWindows()
