"""Reward shaping from per-step state delta.

  r = HP_DELTA_COEF * (delta_enemy_hp_lost - delta_my_hp_lost)
    + PRINCESS_COEF * (enemy_princess_destroyed - my_princess_destroyed) this step
    + KING_COEF     * (enemy_king_destroyed     - my_king_destroyed)     this step
    + TERMINAL on game_state transition: +TERMINAL win, -TERMINAL loss

Tower-destroyed events are edge-detected on HP crossing zero, so a tower
sitting at 0% across many steps doesn't get re-rewarded.

Coefficients are starting points; tune as training data accumulates.
"""
HP_DELTA_COEF = 0.01
PRINCESS_COEF = 0.5
KING_COEF = 1.0
TERMINAL_COEF = 1.0

MY_PRINCESSES = ("my_left_princess", "my_right_princess")
ENEMY_PRINCESSES = ("enemy_left_princess", "enemy_right_princess")
MY_KING = "my_king"
ENEMY_KING = "enemy_king"


def _just_destroyed(prev: dict, curr: dict, tower: str) -> bool:
    return prev[tower] > 0 and curr[tower] == 0


def _hp_loss(prev: dict, curr: dict, towers) -> int:
    return sum(max(0, prev[t] - curr[t]) for t in towers)


def compute(prev: dict, curr: dict) -> float:
    enemy_lost = _hp_loss(prev, curr, ENEMY_PRINCESSES + (ENEMY_KING,))
    my_lost = _hp_loss(prev, curr, MY_PRINCESSES + (MY_KING,))

    r = HP_DELTA_COEF * (enemy_lost - my_lost)

    for t in ENEMY_PRINCESSES:
        if _just_destroyed(prev, curr, t):
            r += PRINCESS_COEF
    for t in MY_PRINCESSES:
        if _just_destroyed(prev, curr, t):
            r -= PRINCESS_COEF
    if _just_destroyed(prev, curr, ENEMY_KING):
        r += KING_COEF
    if _just_destroyed(prev, curr, MY_KING):
        r -= KING_COEF

    # Terminal — only on the transition step (prev was running, curr isn't)
    if prev["game_state"] == 0 and curr["game_state"] != 0:
        if curr["game_state"] == 1:
            r += TERMINAL_COEF
        elif curr["game_state"] == -1:
            r -= TERMINAL_COEF

    return float(r)
