# Attribution

## AI assistance

This project was built with substantial assistance from **Claude (Anthropic)** via the Claude Code CLI. The collaboration model was iterative pair-programming: the human author drove design decisions, calibration, debugging, and integration; Claude generated code on request and helped diagnose runtime issues.

### What Claude generated

- Initial scaffolds for [src/game_wrapper/capture.py](src/game_wrapper/capture.py), [src/game_wrapper/detector.py](src/game_wrapper/detector.py), [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py), [src/game_wrapper/interact.py](src/game_wrapper/interact.py), and [src/game_wrapper/wrapper.py](src/game_wrapper/wrapper.py) (threaded screen capture, YOLO inference wrapper, frame-to-state extraction, mouse/keyboard input layer, and the unified `GameWrapper` interface).
- The `MaskablePPO` training loop scaffold in [src/rl/train.py](src/rl/train.py), the Gymnasium env wrapper in [src/rl/env.py](src/rl/env.py), and the action / observation / reward modules in [src/rl/action.py](src/rl/action.py), [src/rl/obs.py](src/rl/obs.py), and [src/rl/reward.py](src/rl/reward.py).
- The `reset_match()` settle-and-press logic that auto-queues new matches.
- The `pyproject.toml` setup for editable install.

### What the human modified, debugged, or substantially reworked

- **All calibration constants** — every pixel coordinate (grid corners, tower boxes, elixir box, card boxes, endgame boxes, monitor offset, crop region) was measured against the live game by the human using `scripts/coords.py`. Initial AI guesses were wrong by full tiles in both axes.
- **Grid coordinate convention** — multiple back-and-forth iterations on whether grid indices are 0- or 1-indexed and whether row/col offsets are `+0.5` or `−0.5`. Human caught the mismatch between perception's index convention and the action layer's, which was causing every click to land one tile off.
- **Action grid vs. perception grid split** — separating the 18×14 my-side action grid from the 18×29 full-arena perception grid (initial single-grid version was off by a half-tile around the bridge band, which is taller than a regular tile). Human-driven design decision after the agent kept landing cards in the wrong place.
- **Template matching robustness** — the initial card / banner template matching was color-based and broke whenever the user's elixir was too low (cards desaturate to grayscale) or arena banners changed color across arenas. Human debugged the symptoms; both human and Claude iterated to grayscale-on-both-sides matching as the fix.
- **Elixir lock-at-zero bug** — the initial threshold-gated update logic could leave `_last_elixir = 0` indefinitely if the digit template never crossed the confidence bar. Human diagnosed; Claude swapped in unconditional best-match commit.
- **Hand "stuck on played card" bug** — the initial detect_hand fell back to the previous card name when matching was uncertain. Human noticed the agent was trying to play cards that no longer existed in hand; Claude switched the fallback to `"unknown"`.
- **Auto-queue settle logic** — the initial implementation locked onto the exact `game_state` value at function entry, which got stuck on perception flicker between victory and running states. Human reported the hang; Claude rewrote with a flicker-tolerance counter.
- **Project structure refactor** — moving everything under `src/`, adding `pyproject.toml`, restructuring directories per the rubric, fixing all path references.
- **YOLO troop detection model** — fine-tuned on a custom dataset of Clash Royale screenshots manually annotated by the human; Claude wrote no labels, only the training script scaffold.
- **All design decisions** about action space (curated 8-spot list vs. free 18×14 grid), reward shaping coefficients, observation layout, step rate, and architecture choices were human-led.

The author understands the operation of every component and has personally exercised every code path during development. AI suggestions were rejected, modified, or reworked whenever they didn't match the project's needs.

## External libraries

| Library | Purpose | License |
| --- | --- | --- |
| `mss` | Fast cross-platform screen capture | MIT |
| `numpy` | Array operations | BSD |
| `opencv-python` | Frame format conversion, template matching | Apache 2.0 |
| `pydirectinput` | Win32 SendInput-based mouse and keyboard control (DPI-correct) | MIT |
| `torch` / `torchvision` | YOLO backend | BSD |
| `ultralytics` | YOLOv8 detection model and training framework | AGPL-3.0 |
| `gymnasium` | Standard RL environment API | MIT |
| `stable-baselines3` | PPO base implementation | MIT |
| `sb3-contrib` | `MaskablePPO` for action-masked policy training | MIT |
| `tensorboard` | Training metric visualization | Apache 2.0 |

## Models

- **YOLOv8 (small variant)** from Ultralytics — used as the pretrained starting point, then **fine-tuned by the author** on a custom Clash Royale troop-detection dataset. The shipped weights at `models/best.pt` are the fine-tuned checkpoint.

## Datasets

- **Custom-annotated Clash Royale screenshot dataset** — collected and labeled by the author. Used solely to fine-tune the YOLO troop detector. Not redistributed; only the trained weights ship with this repository.
- No external datasets are used.

## Reference materials

- Clash Royale post-match UI behavior verified against [Supercell's official announcement](https://x.com/ClashRoyale/status/1934600054704075062) (June 2025) about Play Again being disabled in 2v2 modes.
- PPO hyperparameter defaults follow the recommendations in the Stable-Baselines3 documentation.
