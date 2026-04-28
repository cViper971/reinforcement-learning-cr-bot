import time

import mss
import pydirectinput

from .capture import TARGET_MONITOR, CROP_REGION

pydirectinput.FAILSAFE = True
pydirectinput.PAUSE = 0

_CARD_KEYS = ('1', '2', '3', '4')

_GRID_BL = (46, 802)
_GRID_TR = (560, 480)
_GRID_COLS = 18
_GRID_ROWS = 14

_TILE_W = (_GRID_TR[0] - _GRID_BL[0]) / _GRID_COLS
_TILE_H = (_GRID_BL[1] - _GRID_TR[1]) / _GRID_ROWS

def _monitor_offset() -> tuple[int, int]:
    with mss.mss() as sct:
        mon = sct.monitors[TARGET_MONITOR]
        return mon["left"], mon["top"]

_MON_LEFT, _MON_TOP = _monitor_offset()
_CROP_LEFT = CROP_REGION["left"] if CROP_REGION else 0
_CROP_TOP = CROP_REGION["top"] if CROP_REGION else 0

def tile_to_screen(col: int, row: int) -> tuple[int, int]:
    fx = _GRID_BL[0] + (col - 0.5) * _TILE_W
    fy = _GRID_BL[1] - (row - 0.5) * _TILE_H
    return round(_MON_LEFT + _CROP_LEFT + fx), round(_MON_TOP + _CROP_TOP + fy)

_CARD_SELECT_DELAY = 0.05  # s — let the game register the card-select before clicking the tile


def play_card(slot: int, col: int, row: int) -> None:
    sx, sy = tile_to_screen(col, row)
    pydirectinput.press(_CARD_KEYS[slot])
    time.sleep(_CARD_SELECT_DELAY)
    pydirectinput.click(sx, sy)


# --- debug: highlight all RL spots on the live frame ---
if __name__ == "__main__":
    import cv2
    from .capture import ScreenCapture
    from rl.action import SPOTS

    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("spot_overlay", cv2.WINDOW_NORMAL)
    print(f"[actions debug] {len(SPOTS)} spots highlighted. ESC to quit.")

    while True:
        frame = cap.get_frame()
        if frame is None:
            continue
        vis = frame.copy()

        for i, spot in enumerate(SPOTS):
            fx = round(_GRID_BL[0] + (spot.col - 0.5) * _TILE_W)
            fy = round(_GRID_BL[1] - (spot.row - 0.5) * _TILE_H)
            cv2.circle(vis, (fx, fy), 8, (0, 255, 255), 2)
            cv2.putText(vis, f"{i}:{spot.name}", (fx + 10, fy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("spot_overlay", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
