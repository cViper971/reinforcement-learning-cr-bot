"""Encode the perception state dict into observation arrays.

Output shape:
  spatial: (C, GRID_ROWS=14, GRID_COLS=18) float32 in [0, 1]
  scalar:  (D,)                            float32 in [0, 1]

Channels (4 to start, expand later):
  0: ally   troop count per cell, normalized by MAX_PER_CELL
  1: enemy  troop count per cell, normalized by MAX_PER_CELL
  2: ally   troop "presence" mask (0/1) per cell
  3: enemy  troop "presence" mask (0/1) per cell

Scalar features (in this order):
  elixir / 10
  6 tower HPs / 100  (in TOWER_ORDER)
  4 hand-card cost / 10
  4 * len(CARD_VOCAB+1) one-hot for hand cards
"""
import numpy as np

from game_state.actions import _GRID_BL, _GRID_TR, _GRID_COLS, _GRID_ROWS
from rl.action import CARD_COSTS, card_cost


GRID_ROWS = _GRID_ROWS  # 14
GRID_COLS = _GRID_COLS  # 18

_TILE_W_PX = (_GRID_TR[0] - _GRID_BL[0]) / _GRID_COLS
_TILE_H_PX = (_GRID_BL[1] - _GRID_TR[1]) / _GRID_ROWS

N_CHANNELS = 4
MAX_PER_CELL = 4.0  # for normalization; >MAX clipped

# Stable card vocabulary order. The +1 slot at the end is for "unknown".
CARD_VOCAB: list[str] = sorted(CARD_COSTS.keys())
CARD_VOCAB_SIZE = len(CARD_VOCAB) + 1  # +1 for unknown
_UNKNOWN_IDX = len(CARD_VOCAB)
_CARD_TO_IDX = {name: i for i, name in enumerate(CARD_VOCAB)}

TOWER_ORDER = [
    "my_left_princess", "my_right_princess", "my_king",
    "enemy_left_princess", "enemy_right_princess", "enemy_king",
]

SCALAR_DIM = 1 + len(TOWER_ORDER) + 4 + 4 * CARD_VOCAB_SIZE


def _troop_to_cell(cx: int, cy: int) -> tuple[int, int] | None:
    """Map troop center pixel (in cropped frame) to (col, row) on the 14x18 grid.
    Returns None if outside grid bounds."""
    col_f = (cx - _GRID_BL[0]) / _TILE_W_PX
    row_f = (_GRID_BL[1] - cy) / _TILE_H_PX
    col = int(col_f)
    row = int(row_f)
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        return None
    return col, row


def _spatial(state: dict) -> np.ndarray:
    out = np.zeros((N_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32)
    for t in state.get("troops", []):
        cell = _troop_to_cell(*t["center"])
        if cell is None:
            continue
        col, row = cell
        if t["team"] == "ally":
            out[0, row, col] += 1.0
            out[2, row, col] = 1.0
        elif t["team"] == "enemy":
            out[1, row, col] += 1.0
            out[3, row, col] = 1.0
    out[0] = np.clip(out[0] / MAX_PER_CELL, 0.0, 1.0)
    out[1] = np.clip(out[1] / MAX_PER_CELL, 0.0, 1.0)
    return out


def _hand_one_hot(card_name: str) -> np.ndarray:
    v = np.zeros(CARD_VOCAB_SIZE, dtype=np.float32)
    v[_CARD_TO_IDX.get(card_name, _UNKNOWN_IDX)] = 1.0
    return v


def _scalar(state: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    parts.append(np.array([state["elixir"] / 10.0], dtype=np.float32))
    parts.append(np.array(
        [state[t] / 100.0 for t in TOWER_ORDER], dtype=np.float32))
    hand: list[str] = state["hand"]
    parts.append(np.array(
        [card_cost(c) / 10.0 if c != "unknown" else 0.0 for c in hand],
        dtype=np.float32))
    for c in hand:
        parts.append(_hand_one_hot(c))
    return np.concatenate(parts)


def encode(state: dict) -> dict[str, np.ndarray]:
    return {"spatial": _spatial(state), "scalar": _scalar(state)}
