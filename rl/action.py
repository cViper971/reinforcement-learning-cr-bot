"""Action space for the RL agent.

Action = MultiDiscrete([5, N_SPOTS]) = (slot, spot_idx).
  slot in {0,1,2,3} plays card from that hand slot
  slot == NOOP_SLOT (4) does nothing (spot ignored)

Spots are a curated list of canonical placement locations rather than a free
18x14 grid — see plan: real CR play uses a small set of spots, free grid
wastes policy capacity and explodes credit assignment.
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


NOOP_SLOT = 4
N_SLOTS = 5  # 4 cards + noop


@dataclass(frozen=True)
class Spot:
    name: str
    col: int
    row: int


# Coords on the my-side action grid (1-indexed, (1,1) = bottom-left).
# Calibrate with the debug overlay (`python -m rl.action`) before training.
SPOTS: list[Spot] = [
    Spot("left_bridge",     4, 14),
    Spot("right_bridge",   15, 14),
    Spot("left_mid",        9,  10),
    Spot("right_mid",      10,  10),
    Spot("left_princess",   4,  8),
    Spot("right_princess", 15,  8),
    Spot("left_back",       9, 0),
    Spot("right_back",     10, 0),
]
N_SPOTS = len(SPOTS)


# Card -> elixir cost. Extend as more card templates are added under
# assets/templates/cards/. Keys must match template filename stem.
CARD_COSTS: dict[str, int] = {
    "archers":       3,
    "giant":         5,
    "goblins":       2,
    "knight":        3,
    "mini_pekka":    4,
    "minions":       3,
    "musketeer":     4,
    "spear_goblins": 2,
}
UNKNOWN_CARD_COST = 99  # treat unknown cards as unplayable


class Action(NamedTuple):
    slot: int
    spot_idx: int

    @property
    def is_noop(self) -> bool:
        return self.slot == NOOP_SLOT


def decode(action_array: np.ndarray) -> Action:
    """Convert MultiDiscrete sample [slot, spot_idx] into an Action."""
    return Action(slot=int(action_array[0]), spot_idx=int(action_array[1]))


def execute(action: Action, game) -> None:
    """Send the action to the game via a GameWrapper. No-op slot does nothing."""
    if action.is_noop:
        return
    spot = SPOTS[action.spot_idx]
    game.act(action.slot, spot.col, spot.row)


def card_cost(card_name: str) -> int:
    return CARD_COSTS.get(card_name, UNKNOWN_CARD_COST)


def build_mask(elixir: int, hand: list[str]) -> dict[str, np.ndarray]:
    """Per-component mask for sb3-contrib MaskablePPO over MultiDiscrete([5, N_SPOTS]).

    Returns a dict with 'slot' (shape [5]) and 'spot' (shape [N_SPOTS]).
    A slot is valid if its card is known and affordable. No-op is always valid.
    All spots are always valid (placement-legality varies per card and changes
    after first tower drop — left as future work).
    """
    slot_mask = np.zeros(N_SLOTS, dtype=bool)
    for s in range(4):
        card = hand[s]
        if card == "unknown":
            continue
        if card_cost(card) <= elixir:
            slot_mask[s] = True
    slot_mask[NOOP_SLOT] = True

    spot_mask = np.ones(N_SPOTS, dtype=bool)
    return {"slot": slot_mask, "spot": spot_mask}


# --- debug: overlay spot markers on the live frame to calibrate coords ---
if __name__ == "__main__":
    import time
    import cv2
    from game_wrapper.capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
    from game_wrapper.actions import _GRID_BL, _GRID_TR, _GRID_COLS, _GRID_ROWS

    tile_w = (_GRID_TR[0] - _GRID_BL[0]) / _GRID_COLS
    tile_h = (_GRID_BL[1] - _GRID_TR[1]) / _GRID_ROWS

    def spot_to_frame_xy(spot: Spot) -> tuple[int, int]:
        # Inverse of game_wrapper.actions.tile_to_screen, dropping the monitor
        # offset so we get coords in the cropped frame's space.
        fx = _GRID_BL[0] + (spot.col - 0.5) * tile_w
        fy = _GRID_BL[1] - (spot.row - 0.5) * tile_h
        return round(fx), round(fy)

    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("spot_overlay", cv2.WINDOW_NORMAL)

    print(f"{N_SPOTS} spots loaded. ESC to quit.")
    while True:
        frame = cap.get_frame()
        if frame is None:
            continue

        vis = frame.copy()
        for i, spot in enumerate(SPOTS):
            x, y = spot_to_frame_xy(spot)
            color = (0, 255, 255)
            cv2.circle(vis, (x, y), 8, color, 2)
            cv2.putText(vis, f"{i}:{spot.name}", (x + 10, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("spot_overlay", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
