import mss
import msvcrt
import ctypes
import time

from game_wrapper.wrapper import TARGET_MONITOR, CROP_REGION

with mss.mss() as sct:
    mon = sct.monitors[TARGET_MONITOR]
    mon_left = mon["left"]
    mon_top  = mon["top"]

crop_left = CROP_REGION["left"] if CROP_REGION else 0
crop_top  = CROP_REGION["top"]  if CROP_REGION else 0

def get_physical_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

print("SPACE = print frame coord | Q = quit\n")

while True:
    if msvcrt.kbhit():
        key = msvcrt.getch().lower()
        if key == b' ':
            ax, ay = get_physical_pos()
            fx = ax - mon_left - crop_left
            fy = ay - mon_top  - crop_top
            print(f"({fx}, {fy})")
        elif key == b'q':
            break
    time.sleep(0.05)
