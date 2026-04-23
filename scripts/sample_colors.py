"""
Sample HSV color at cursor position.
Press SPACE to print HSV at cursor, Q to quit.
"""

import cv2
import ctypes
import ctypes.wintypes
import mss
import msvcrt
import numpy as np
import time
from game_state.capture import ScreenCapture, TARGET_MONITOR, CROP_REGION

MONITOR_INDEX = TARGET_MONITOR

with mss.mss() as sct:
    mon = sct.monitors[MONITOR_INDEX]
    mon_left = mon["left"]
    mon_top = mon["top"]


def get_physical_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def main():
    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    crop_left = CROP_REGION["left"]
    crop_top = CROP_REGION["top"]

    print("SPACE = sample HSV at cursor | Q = quit\n")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b' ':
                frame = cap.get_frame()
                if frame is None:
                    continue
                mx, my = get_physical_pos()
                fx = mx - mon_left - crop_left
                fy = my - mon_top - crop_top
                if 0 <= fx < frame.shape[1] and 0 <= fy < frame.shape[0]:
                    bgr = frame[fy, fx]
                    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                    print(f"pos=({fx},{fy})  BGR={tuple(bgr)}  HSV=({hsv[0]}, {hsv[1]}, {hsv[2]})")
                else:
                    print(f"cursor outside frame ({fx}, {fy})")
            elif key == b'q':
                break
        time.sleep(0.05)

    cap.stop()


if __name__ == "__main__":
    main()
