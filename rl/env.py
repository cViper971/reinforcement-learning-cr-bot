"""Gymnasium env wrapping a GameWrapper.

Phase 1 reset() waits for a manually-started match: it polls until game_state
stabilizes at 0 (running) for several consecutive frames. Auto-queueing comes
in Phase 3 once GameWrapper.reset_match() is filled in.

Step rate: 2 Hz (500 ms boundaries). Each step: optionally place a card,
sleep until next boundary, capture, perceive, compute reward.
"""
import ctypes
import threading
import time
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


_kill = threading.Event()


def _kill_watcher() -> None:
    """Background thread that sets _kill the moment Q is pressed.
    Uses GetAsyncKeyState so it works globally (BlueStacks can have focus)."""
    while not _kill.is_set():
        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            print("\n[env] Q pressed — killing bot")
            _kill.set()
            return
        time.sleep(0.03)


threading.Thread(target=_kill_watcher, daemon=True).start()

from game_wrapper import GameWrapper, DEFAULT_WEIGHTS

from rl.action import (
    Action, N_SLOTS, N_SPOTS, SPOTS, build_mask, card_cost, decode, execute,
)
import rl.obs as obs_module
from rl.obs import _set_troop_vocab, encode
from rl.reward import compute as compute_reward


STEP_HZ = 2.0
STEP_DT = 1.0 / STEP_HZ  # 0.5 s
RESET_STABLE_FRAMES = 4   # # of consecutive running-state frames to confirm match start
RESET_POLL_DT = 0.5


class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, conf_threshold: float = 0.2):
        super().__init__()

        self.action_space = spaces.MultiDiscrete([N_SLOTS, N_SPOTS])

        self._game = GameWrapper(weights_path=weights_path, conf_threshold=conf_threshold)

        # Initialize troop vocabulary from detector
        _set_troop_vocab(self._game.troop_classes())

        # Create observation space with updated SCALAR_DIM
        self.observation_space = spaces.Box(0.0, 1.0, (obs_module.SCALAR_DIM,), np.float32)

        self._prev_state: dict | None = None
        self._next_step_deadline: float = 0.0

    # --- gym API ---

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Phase 1: wait for human to queue a match. Poll until we see N
        # consecutive running-state frames.
        print("[env] waiting for match start (queue manually)... press Q to kill, Ctrl+C to abort")
        running_streak = 0
        last_print = 0.0
        while running_streak < RESET_STABLE_FRAMES:
            if _kill.is_set():
                raise KeyboardInterrupt("Q pressed during reset")
            state = self._game.get_state()
            if state["game_state"] == 0:
                running_streak += 1
            else:
                running_streak = 0
            now = time.time()
            if now - last_print >= 5.0:
                print(f"[env] reset polling: game_state={state['game_state']}  elixir={state['elixir']}")
                last_print = now
            time.sleep(RESET_POLL_DT)

        self._prev_state = self._game.get_state()
        self._next_step_deadline = time.perf_counter() + STEP_DT
        obs = encode(self._prev_state)
        info = self._info(self._prev_state)
        print("[env] match running, episode begin")
        return obs, info

    def step(self, action_array: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = decode(action_array)

        # Validate against current mask. If invalid, treat as no-op rather than
        # erroring — keeps random/exploration policies happy.
        if not action.is_noop:
            mask = build_mask(self._prev_state["elixir"], self._prev_state["hand"])
            if mask["slot"][action.slot]:
                now_pc = time.perf_counter()
                last_pc = getattr(self, "_last_play_t", None)
                dt_since_last = (now_pc - last_pc) if last_pc is not None else 0.0
                card = self._prev_state['hand'][action.slot]
                ts = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"
                print(f"[{ts} +{dt_since_last:.3f}s] "
                      f"played {card} (cost {card_cost(card)}) at {SPOTS[action.spot_idx].name} "
                      f"| elixir={self._prev_state['elixir']}",
                      flush=True)
                self._last_play_t = now_pc
                execute(action, self._game)

        # Sleep to next 500ms boundary
        now = time.perf_counter()
        sleep_for = self._next_step_deadline - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._next_step_deadline += STEP_DT

        state = self._game.get_state()
        ts = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"
        print(f"[{ts}] elixir={state['elixir']} hand={state['hand']}", flush=True)
        reward = compute_reward(self._prev_state, state)
        terminated = state["game_state"] != 0
        truncated = False
        obs = encode(state)
        info = self._info(state)
        self._prev_state = state
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._game.close()

    # --- internals ---

    def _info(self, state: dict) -> dict[str, Any]:
        mask = build_mask(state["elixir"], state["hand"])
        return {
            "raw": state,
            "action_mask": mask,
        }


# --- debug: random policy plays one episode ---
if __name__ == "__main__":
    env = ClashRoyaleEnv()
    obs, info = env.reset()
    total_r = 0.0
    steps = 0
    t0 = time.time()
    try:
        print("[env] press Q at any time to kill the bot")
        done = False
        while not done:
            if _kill.is_set():
                break
            mask = info["action_mask"]
            # Sample respecting per-component mask (so random play is at least
            # affordable — otherwise nothing ever gets placed early in a match).
            valid_slots = np.flatnonzero(mask["slot"])
            slot = int(np.random.choice(valid_slots))
            spot = int(np.random.randint(N_SPOTS))
            obs, reward, terminated, truncated, info = env.step(np.array([slot, spot]))
            total_r += reward
            steps += 1
            done = terminated or truncated
    finally:
        dt = time.time() - t0
        print(f"\n[done] steps={steps} total_reward={total_r:+.3f} "
              f"elapsed={dt:.1f}s ({steps/max(dt,0.01):.2f} steps/s)")
        env.close()
