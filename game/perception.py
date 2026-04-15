import numpy as np
import cv2
import os

# Elixir number bounding box (frame-relative coords)
_ELIXIR_NUM_BOX = (153, 1014, 199, 1061)  # x1, y1, x2, y2

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
_ELIXIR_TEMPLATES: dict[int, np.ndarray] = {}

for i in range(11):
    path = os.path.join(_TEMPLATE_DIR, f"elixir_{i}.png")
    if os.path.exists(path):
        _ELIXIR_TEMPLATES[i] = cv2.imread(path)


_ELIXIR_CONFIDENCE_THRESHOLD = 0.8
_last_elixir = 0


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

    if best_val >= _ELIXIR_CONFIDENCE_THRESHOLD:
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


def extract_state(frame: np.ndarray, tracker=None) -> dict:
    """Extract game state from a frame. If a TroopTracker is provided,
    team-aware tracked troops are included under 'troops'."""
    state = {"elixir": detect_elixir(frame)}
    for name in _TOWER_BOXES:
        state[name] = detect_tower_health(frame, name)
    if tracker is not None:
        state["troops"] = [t.as_dict() for t in tracker.update(frame)]
    return state


# --- debug ---
if __name__ == "__main__":
    import os
    import time
    from .capture import ScreenCapture, TARGET_MONITOR, CROP_REGION
    from .detector import TroopDetector
    from .tracker import TroopTracker, MIDLINE_Y

    weights_path = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "best.pt")
    detector = TroopDetector(weights_path, conf_threshold=0.3)
    tracker = TroopTracker(detector)

    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    last_print = 0
    while True:
        frame = cap.get_frame()
        if frame is None:
            continue

        state = extract_state(frame, tracker=tracker)
        now = time.time()
        if now - last_print >= 5:
            print(f"Elixir: {state['elixir']}")
            for name in _TOWER_BOXES:
                print(f"  {name}: {state[name]}%")
            print(f"  troops ({len(state['troops'])}):")
            for t in state["troops"]:
                cx, cy = t["center"]
                print(f"    #{t['id']} {t['team']:5s} {t['name']:15s} @ ({cx},{cy})  conf={t['confidence']:.2f}")
            print()
            last_print = now

        vis = frame.copy()

        # Midline (for debugging team assignment)
        cv2.line(vis, (0, MIDLINE_Y), (vis.shape[1], MIDLINE_Y), (128, 128, 128), 1)

        # Elixir
        x1, y1, x2, y2 = _ELIXIR_NUM_BOX
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f"Elixir: {state['elixir']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 255), 2)

        # Tower health
        for name, (tx1, ty1, tx2, ty2) in _TOWER_BOXES.items():
            hp = state[name]
            color = (0, 255, 0) if hp > 50 else (0, 255, 255) if hp > 25 else (0, 0, 255)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(vis, f"{name}: {hp}%", (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Troops — blue = ally, red = enemy
        for t in state["troops"]:
            tx1, ty1, tx2, ty2 = t["bbox"]
            color = (255, 128, 0) if t["team"] == "ally" else (0, 0, 255)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2)
            label = f"#{t['id']} {t['team']} {t['name']}"
            cv2.putText(vis, label, (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("debug", vis)

        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
