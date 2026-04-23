"""Capture card template screenshots from live frames.

SPACE = save all 4 current card slots to assets/templates/cards/
Q     = quit

Files are named card_<slot>_<timestamp>.png — rename them to the actual
card name (e.g. archers.png) afterwards.
"""

import os
import time
import msvcrt
import cv2

from game.capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
from game.perception import _CARD_BOXES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "templates", "cards")
os.makedirs(OUT_DIR, exist_ok=True)

cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
cap.start()
time.sleep(0.3)

cv2.namedWindow("cards", cv2.WINDOW_NORMAL)
print("SPACE = save 4 slots | Q = quit\n")

while True:
    frame = cap.get_frame()
    if frame is None:
        continue

    vis = frame.copy()
    for x1, y1, x2, y2 in _CARD_BOXES:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow("cards", vis)
    cv2.waitKey(1)

    if msvcrt.kbhit():
        key = msvcrt.getch().lower()
        if key == b' ':
            ts = int(time.time() * 1000)
            for slot_idx, (x1, y1, x2, y2) in enumerate(_CARD_BOXES):
                crop = frame[y1:y2, x1:x2]
                path = os.path.join(OUT_DIR, f"card_{slot_idx}_{ts}.png")
                cv2.imwrite(path, crop)
                print(f"  saved {path}")
        elif key == b'q':
            break

cap.stop()
cv2.destroyAllWindows()
