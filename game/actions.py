import mss
import pyautogui

from .capture import TARGET_MONITOR, CROP_REGION

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

_CARD_KEYS = ('1', '2', '3', '4')

_GRID_BL = (-91, 642)
_GRID_TR = (315, 382)
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


def play_card(slot: int, col: int, row: int) -> None:
    sx, sy = tile_to_screen(col, row)
    pyautogui.press(_CARD_KEYS[slot])
    pyautogui.click(sx, sy)
