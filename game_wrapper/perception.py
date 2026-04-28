import glob
import numpy as np
import cv2
import os

_GRID_BL = (48, 804)
_GRID_TR = (558, 104)
_GRID_COLS = 18
_GRID_ROWS = 29

# Grid tile dimensions (pixels)
_TILE_W_PX = (_GRID_TR[0] - _GRID_BL[0]) / _GRID_COLS
_TILE_H_PX = (_GRID_BL[1] - _GRID_TR[1]) / _GRID_ROWS

# Elixir number bounding box (frame-relative coords)
_ELIXIR_NUM_BOX = (153, 1014, 199, 1061)  # x1, y1, x2, y2

# Card slot bounding boxes (frame-relative). Derived from calibration:
#   card 1 top-left = (138, 898), card 1 bottom-right = (239, 1024),
#   card 4 bottom-right = (581, 1024). Pitch = (581-239)/3 = 114.
_CARD_BOXES = [
    (138, 898, 239, 1024),
    (252, 898, 353, 1024),
    (366, 898, 467, 1024),
    (480, 898, 581, 1024),
]

# End-game banner bounding boxes (frame-relative). VICTORY and DEFEAT banners
# appear at different positions, so they need separate boxes. Each template is
# matched only against its own box.
_ENDGAME_BOXES = {
    "victory": (214, 416, 387, 453),
    "defeat":  (218,  94, 381, 137),
}

# Tower health bar bounding boxes (frame-relative coords: x1, y1, x2, y2)
_TOWER_BOXES = {
    "enemy_left_princess":  (122, 156, 188, 166),
    "enemy_right_princess": (440, 157, 506, 166),
    "enemy_king":           (270,  28, 361,  47),
    "my_left_princess":     (122, 668, 188, 679),
    "my_right_princess":    (440, 668, 504, 679),
    "my_king":              (270, 808, 364, 830),
}

# HSV ranges for health bar fill colors
_MY_HP_LOW  = np.array([85, 100, 150], dtype=np.uint8)
_MY_HP_HIGH = np.array([115, 200, 255], dtype=np.uint8)
_ENEMY_HP_LOW  = np.array([155, 100, 150], dtype=np.uint8)
_ENEMY_HP_HIGH = np.array([180, 200, 255], dtype=np.uint8)

# Load elixir digit templates (0-10)
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "templates")
_ELIXIR_TEMPLATE_DIR = os.path.join(_TEMPLATE_DIR, "elixir")
_ELIXIR_TEMPLATES: dict[int, np.ndarray] = {}

for i in range(11):
    path = os.path.join(_ELIXIR_TEMPLATE_DIR, f"elixir_{i}.png")
    if os.path.exists(path):
        _ELIXIR_TEMPLATES[i] = cv2.imread(path)

# Load card templates — filename (without .png) = card name. Stored as
# grayscale so matching is invariant to the desaturated "not enough elixir"
# look (cards render black-and-white when uncastable).
_CARD_TEMPLATE_DIR = os.path.join(_TEMPLATE_DIR, "cards")
_CARD_TEMPLATES: dict[str, np.ndarray] = {}

if os.path.isdir(_CARD_TEMPLATE_DIR):
    for path in glob.glob(os.path.join(_CARD_TEMPLATE_DIR, "*.png")):
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        _CARD_TEMPLATES[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Load endgame banner templates — expects victory.png and defeat.png
_ENDGAME_TEMPLATE_DIR = os.path.join(_TEMPLATE_DIR, "endgame")
_ENDGAME_TEMPLATES: dict[str, np.ndarray] = {}

if os.path.isdir(_ENDGAME_TEMPLATE_DIR):
    for path in glob.glob(os.path.join(_ENDGAME_TEMPLATE_DIR, "*.png")):
        name = os.path.splitext(os.path.basename(path))[0]
        _ENDGAME_TEMPLATES[name] = cv2.imread(path)

_CARD_CONFIDENCE_THRESHOLD = 0.4
_ENDGAME_CONFIDENCE_THRESHOLD = 0.7
_last_elixir = 0
_last_hand: list[str] = ["unknown"] * 4


def detect_elixir(frame: np.ndarray) -> int:
    """Match the elixir digit region against saved templates.
    Only updates if the best match exceeds the confidence threshold."""
    global _last_elixir

    if not _ELIXIR_TEMPLATES:
        raise RuntimeError("No elixir templates found. Run capture_elixir_templates.py first.")

    x1, y1, x2, y2 = _ELIXIR_NUM_BOX
    digit = frame[y1:y2, x1:x2]

    best_val = -1.0
    best_n = 0

    for n, tmpl in _ELIXIR_TEMPLATES.items():
        if tmpl.shape != digit.shape:
            tmpl = cv2.resize(tmpl, (digit.shape[1], digit.shape[0]))
        result = cv2.matchTemplate(digit, tmpl, cv2.TM_CCOEFF_NORMED)
        val = result[0][0]
        if val > best_val:
            best_val = val
            best_n = n

    # Always commit the best match — the previous "only update above threshold"
    # logic could lock _last_elixir at 0 forever when matches consistently
    # scored just below the bar (icon glow / transitions). best_n is by
    # construction the closest template, so ambiguity-handling is better done
    # via temporal smoothing if needed.
    _last_elixir = best_n
    return _last_elixir


def detect_tower_health(frame: np.ndarray, tower_name: str) -> int:
    """Detect tower health as a percentage (0-100) by measuring bar fill ratio.
    King towers show 100% when their bar is not visible (hidden until damaged)."""
    x1, y1, x2, y2 = _TOWER_BOXES[tower_name]
    bar = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)

    if tower_name.startswith("my"):
        mask = cv2.inRange(hsv, _MY_HP_LOW, _MY_HP_HIGH)
    else:
        mask = cv2.inRange(hsv, _ENEMY_HP_LOW, _ENEMY_HP_HIGH)

    filled_cols = (mask.sum(axis=0) > 0).sum()
    total_cols = mask.shape[1]

    if total_cols == 0:
        return 100 if "king" in tower_name else 0

    ratio = filled_cols / total_cols

    # No bar detected — king towers default to 100%, princess to 0% (destroyed)
    if ratio < 0.02:
        return 100 if "king" in tower_name else 0

    return round(ratio * 100)


def detect_hand(frame: np.ndarray) -> list[str]:
    """Template-match each of the 4 card slots. Returns card names in slot order.
    A slot below the confidence threshold is reported as "unknown" so action
    masking won't keep treating a played-away card as still in hand."""
    global _last_hand

    if not _CARD_TEMPLATES:
        return list(_last_hand)

    for slot_idx, (x1, y1, x2, y2) in enumerate(_CARD_BOXES):
        slot_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        best_val = -1.0
        best_name = "unknown"

        for name, tmpl in _CARD_TEMPLATES.items():
            if tmpl.shape != slot_gray.shape:
                tmpl = cv2.resize(tmpl, (slot_gray.shape[1], slot_gray.shape[0]))
            result = cv2.matchTemplate(slot_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            val = result[0][0]
            if val > best_val:
                best_val = val
                best_name = name

        _last_hand[slot_idx] = best_name if best_val >= _CARD_CONFIDENCE_THRESHOLD else "unknown"

    return list(_last_hand)


def detect_game_state(frame: np.ndarray) -> int:
    """Match each endgame banner region against its template.
    Returns: 0 = running, 1 = won, -1 = lost.
    If both below threshold, assume match is still going."""
    scores: dict[str, float] = {}

    for name, tmpl in _ENDGAME_TEMPLATES.items():
        box = _ENDGAME_BOXES.get(name)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        region = frame[y1:y2, x1:x2]
        if tmpl.shape != region.shape:
            tmpl = cv2.resize(tmpl, (region.shape[1], region.shape[0]))
        result = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        scores[name] = float(result[0][0])

    if not scores:
        return 0

    best_name = max(scores, key=scores.get)
    if scores[best_name] < _ENDGAME_CONFIDENCE_THRESHOLD:
        return 0
    if best_name == "victory":
        return 1
    if best_name == "defeat":
        return -1
    return 0


def _pixel_to_game_coords(cx: int, cy: int) -> tuple[int, int] | None:
    """Convert pixel center (cx, cy) to 1-indexed game grid (col, row).
    (1, 1) = bottom-left tile. Returns None if outside grid bounds."""
    col = int((cx - _GRID_BL[0]) / _TILE_W_PX) + 1
    row = int((_GRID_BL[1] - cy) / _TILE_H_PX) + 1
    if 1 <= col <= _GRID_COLS and 1 <= row <= _GRID_ROWS:
        return col, row
    return None


def extract_state(frame: np.ndarray, detector=None) -> dict:
    """Extract game state from a frame. If a TroopDetector is provided,
    troops (with ally/enemy team) are included. Troop pixel coords converted to game coords."""
    state = {
        "elixir": detect_elixir(frame),
        "hand": detect_hand(frame),
        "game_state": detect_game_state(frame),
    }
    for name in _TOWER_BOXES:
        state[name] = detect_tower_health(frame, name)
    if detector is not None:
        troops = []
        for d in detector.detect(frame):
            troop_dict = d.as_dict()
            # Convert pixel center to game coords
            game_coords = _pixel_to_game_coords(*d.center)
            if game_coords:
                col, row = game_coords
                troop_dict["col"] = col
                troop_dict["row"] = row
            troops.append(troop_dict)
        state["troops"] = troops
    return state


# --- debug ---
if __name__ == "__main__":
    import os
    import time
    from .capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
    from .detector import TroopDetector

    weights_path = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "best.pt")
    detector = TroopDetector(weights_path, conf_threshold=0.3)

    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    last_print = 0
    while True:
        frame = cap.get_frame()
        if frame is None:
            continue

        state = extract_state(frame, detector=detector)
        now = time.time()
        if now - last_print >= 5:
            print(f"Elixir: {state['elixir']}  game_state: {state['game_state']}")
            print(f"  hand: {state['hand']}")
            for name in _TOWER_BOXES:
                print(f"  {name}: {state[name]}%")
            print(f"  troops ({len(state['troops'])}):")
            for t in state["troops"]:
                if "col" in t:
                    print(f"    {t['team']:5s} {t['name']:20s} @ grid({t['col']},{t['row']})  conf={t['confidence']:.2f}")
                else:
                    print(f"    {t['team']:5s} {t['name']:20s} @ out-of-bounds  conf={t['confidence']:.2f}")
            print()
            last_print = now

        vis = frame.copy()

        # Elixir
        x1, y1, x2, y2 = _ELIXIR_NUM_BOX
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f"Elixir: {state['elixir']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 255), 2)

        # Endgame banner regions + state
        for ex1, ey1, ex2, ey2 in _ENDGAME_BOXES.values():
            if ex2 > ex1 and ey2 > ey1:
                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
        gs = state["game_state"]
        gs_label = {1: "WON", -1: "LOST", 0: "running"}[gs]
        gs_color = (0, 255, 0) if gs == 1 else (0, 0, 255) if gs == -1 else (200, 200, 200)
        cv2.putText(vis, f"Game: {gs_label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, gs_color, 2)

        # Tower health
        for name, (tx1, ty1, tx2, ty2) in _TOWER_BOXES.items():
            hp = state[name]
            color = (0, 255, 0) if hp > 50 else (0, 255, 255) if hp > 25 else (0, 0, 255)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(vis, f"{name}: {hp}%", (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Cards
        for slot_idx, (cx1, cy1, cx2, cy2) in enumerate(_CARD_BOXES):
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.putText(vis, state["hand"][slot_idx], (cx1, cy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Troops — blue = ally, red = enemy, green = unknown team
        for t in state["troops"]:
            tx1, ty1, tx2, ty2 = t["bbox"]
            color = (255, 128, 0) if t["team"] == "ally" else (0, 0, 255) if t["team"] == "enemy" else (0, 255, 0)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2)
            label = f"{t['team'] or '?'} {t['name']}"
            cv2.putText(vis, label, (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("debug", vis)

        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
