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
N_SLOTS = 5

@dataclass(frozen=True)
class Spot:
    name: str
    col: int
    row: int

# Coords on the my-side action grid (1-indexed, (1,1) = bottom-left).
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


# Card -> elixir cost.
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

def build_mask(elixir: int, hand: list[str]) -> np.ndarray:
    mask = np.zeros(N_SLOTS + N_SPOTS, dtype=bool)
    for s in range(4):
        card = hand[s]
        if card != "unknown" and card_cost(card) <= elixir:
            mask[s] = True
    mask[NOOP_SLOT] = True
    mask[N_SLOTS:] = True  # all spots always valid
    return mask
