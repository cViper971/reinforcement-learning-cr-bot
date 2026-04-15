"""
Action layer for controlling Clash Royale in BlueStacks.

Exposes a single function:
    play(card_slot: int, grid_pos: tuple[int, int])

where card_slot is 0-3 and grid_pos is (x, y) on the arena tile grid.
"""

import time
import pydirectinput

# --- Calibration (monitor-space pixel coords) ---

# Centers of the 4 card slots in your hand (left to right)
CARD_SLOTS: list[tuple[int, int]] = [
    (0, 0),  # slot 0 — TODO calibrate
    (0, 0),  # slot 1 — TODO calibrate
    (0, 0),  # slot 2 — TODO calibrate
    (0, 0),  # slot 3 — TODO calibrate
]

# Arena bounds (your playable half: top-left and bottom-right in monitor coords)
ARENA_TOP_LEFT = (0, 0)      # TODO calibrate
ARENA_BOTTOM_RIGHT = (0, 0)  # TODO calibrate

# Grid dimensions — CR arena is 18 wide x 32 tall total.
# Your deployable half is bottom 15 rows (enemy side also deployable after tower kill).
GRID_WIDTH = 18
GRID_HEIGHT = 32

CLICK_DELAY = 0.05   # between card click and deploy click
POST_DELAY = 0.1     # after full action


def grid_to_pixel(gx: int, gy: int) -> tuple[int, int]:
    """Convert grid coords to monitor pixel coords."""
    x1, y1 = ARENA_TOP_LEFT
    x2, y2 = ARENA_BOTTOM_RIGHT
    tile_w = (x2 - x1) / GRID_WIDTH
    tile_h = (y2 - y1) / GRID_HEIGHT
    px = int(x1 + (gx + 0.5) * tile_w)
    py = int(y1 + (gy + 0.5) * tile_h)
    return px, py


def play(card_slot: int, grid_pos: tuple[int, int]) -> None:
    """Play a card at a grid position.

    Args:
        card_slot: 0-3, which card in your hand
        grid_pos: (x, y) on the arena grid
    """
    if not (0 <= card_slot < 4):
        raise ValueError(f"card_slot must be 0-3, got {card_slot}")

    card_x, card_y = CARD_SLOTS[card_slot]
    deploy_x, deploy_y = grid_to_pixel(*grid_pos)

    pydirectinput.click(card_x, card_y)
    time.sleep(CLICK_DELAY)
    pydirectinput.click(deploy_x, deploy_y)
    time.sleep(POST_DELAY)


# --- debug ---
if __name__ == "__main__":
    # Dry-run: print what play() would do without clicking
    for slot in range(4):
        for pos in [(0, 0), (9, 16), (17, 31)]:
            cx, cy = CARD_SLOTS[slot]
            px, py = grid_to_pixel(*pos)
            print(f"play({slot}, {pos}) -> click({cx},{cy}) then click({px},{py})")
