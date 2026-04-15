import mss
import msvcrt
import ctypes
import time

MONITOR_INDEX = 1

with mss.mss() as sct:
    mon = sct.monitors[MONITOR_INDEX]
    mon_left = mon["left"]
    mon_top  = mon["top"]

def get_physical_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

print("SPACE = print coord | Q = quit\n")

while True:
    if msvcrt.kbhit():
        key = msvcrt.getch().lower()
        if key == b' ':
            ax, ay = get_physical_pos()
            print(f"({ax - mon_left}, {ay - mon_top})")
        elif key == b'q':
            break
    time.sleep(0.05)
