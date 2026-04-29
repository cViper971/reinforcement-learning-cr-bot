import numpy as np
from rl.action import CARD_COSTS

MAX_ALLIES  = 10
MAX_ENEMIES = 10
GRID_COLS   = 18
GRID_ROWS   = 29
N_SECTIONS  = 6  # ally_left/right, bridge_left/right, enemy_left/right

# Perception emits 1-indexed (col, row) with (1,1) = bottom-left, max (18,29).
_ALLY_ROW_MAX   = 12   # rows 1-12 = ally side
_BRIDGE_ROW_MAX = 17   # rows 13-17 = bridge band, 18-29 = enemy side
_MID_COL        = 10   # col < 10 = left, >= 10 = right (true midline at 9.5)

TOWER_ORDER = [
    "my_left_princess", "my_right_princess", "my_king",
    "enemy_left_princess", "enemy_right_princess", "enemy_king",
]

CARD_VOCAB       = sorted(CARD_COSTS.keys())
CARD_VOCAB_SIZE  = len(CARD_VOCAB)  # always exactly your 8 deck cards
_CARD_TO_IDX     = {name: i for i, name in enumerate(CARD_VOCAB)}

TROOP_VOCAB      = []
TROOP_VOCAB_SIZE = 0
_TROOP_TO_IDX    = {}
SCALAR_DIM       = 0


def _set_troop_vocab(class_names) -> None:
    global TROOP_VOCAB, TROOP_VOCAB_SIZE, _TROOP_TO_IDX, SCALAR_DIM
    TROOP_VOCAB      = sorted(name.lower() for name in class_names)
    TROOP_VOCAB_SIZE = len(TROOP_VOCAB) + 1
    _TROOP_TO_IDX    = {name: i for i, name in enumerate(TROOP_VOCAB)}
    SCALAR_DIM       = (
        1 + len(TOWER_ORDER) + 4 * CARD_VOCAB_SIZE +
        MAX_ALLIES  * (CARD_VOCAB_SIZE  + N_SECTIONS) +
        MAX_ENEMIES * (TROOP_VOCAB_SIZE + N_SECTIONS)
    )


def _card_vec(name: str) -> np.ndarray:
    v = np.zeros(CARD_VOCAB_SIZE, dtype=np.float32)
    idx = _CARD_TO_IDX.get(name.lower())
    if idx is not None:
        v[idx] = 1.0
    return v


def _one_hot(name: str, vocab: list[str], vocab_to_idx: dict) -> np.ndarray:
    v = np.zeros(len(vocab) + 1, dtype=np.float32)
    v[vocab_to_idx.get(name.lower(), len(vocab))] = 1.0
    return v


def _to_section(col: int, row: int) -> int:
    side = 0 if col < _MID_COL else 1
    zone = 0 if row <= _ALLY_ROW_MAX else (1 if row <= _BRIDGE_ROW_MAX else 2)
    return zone * 2 + side


def _encode_troop_list(troops: list[dict], team: str, max_size: int, name_vec_fn) -> np.ndarray:
    filtered = sorted(
        (t for t in troops if t["team"] == team and "col" in t),
        key=lambda t: t["row"],
    )
    name_dim = len(name_vec_fn(""))
    out = np.zeros((max_size, name_dim + N_SECTIONS), dtype=np.float32)
    for i, t in enumerate(filtered[:max_size]):
        sec = np.zeros(N_SECTIONS, dtype=np.float32)
        sec[_to_section(t["col"], t["row"])] = 1.0
        out[i] = np.concatenate([name_vec_fn(t["name"]), sec])
    return out


def encode(state: dict) -> np.ndarray:
    troops = state.get("troops", [])
    ally_name  = _card_vec
    enemy_name = lambda n: _one_hot(n, TROOP_VOCAB, _TROOP_TO_IDX)
    parts = [
        np.array([state["elixir"] / 10.0], dtype=np.float32),
        np.array([state[t] / 100.0 for t in TOWER_ORDER], dtype=np.float32),
        *[_card_vec(card) for card in state["hand"]],
        _encode_troop_list(troops, "ally",  MAX_ALLIES,  ally_name).flatten(),
        _encode_troop_list(troops, "enemy", MAX_ENEMIES, enemy_name).flatten(),
    ]
    return np.concatenate(parts)
