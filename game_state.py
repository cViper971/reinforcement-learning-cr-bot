import numpy as np
import cv2

# Frame-relative bounding box of the elixir bar
# (monitor coords minus CROP_REGION left/top)
_ELIXIR_BOX = (153, 1035, 590, 1066)  # x1, y1, x2, y2

# HSV range for the purple/pink fill — tune if needed
_ELIXIR_LOW  = np.array([130, 60, 60],  dtype=np.uint8)
_ELIXIR_HIGH = np.array([170, 255, 255], dtype=np.uint8)


def detect_elixir(frame: np.ndarray) -> int:
    x1, y1, x2, y2 = _ELIXIR_BOX
    bar = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ELIXIR_LOW, _ELIXIR_HIGH)
    # collapse rows — True for each column that has any purple pixel
    filled_cols = (mask.sum(axis=0) > 0).sum()
    fill_ratio = filled_cols / mask.shape[1]
    return int(fill_ratio * 10)


def extract_state(frame: np.ndarray) -> dict:
    return {
        "elixir": detect_elixir(frame),
    }


# --- debug ---
if __name__ == "__main__":
    from capture_test import ScreenCapture, TARGET_MONITOR, CROP_REGION
    import time

    cap = ScreenCapture(monitor_index=TARGET_MONITOR, crop=CROP_REGION)
    cap.start()
    time.sleep(0.3)

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    last_elixir = None
    while True:
        frame = cap.get_frame()
        if frame is None:
            continue

        state = extract_state(frame)
        if state["elixir"] != last_elixir:
            print(f"Elixir: {state['elixir']}")
            last_elixir = state["elixir"]
        vis = frame.copy()

        x1, y1, x2, y2 = _ELIXIR_BOX
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f"Elixir: {state['elixir']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 255), 2)
        cv2.imshow("debug", vis)

        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()
