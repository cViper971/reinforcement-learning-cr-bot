# Attribution

## How I used AI on this project

I used Claude (via the Claude Code CLI) to assist in the development of this project. Claude helped me get drafts of code on the page faster, but the project plan, calibration, and most of the debugging were my own work.

### Code that Claude drafted

- The threaded screen capture in [src/game_wrapper/capture.py](src/game_wrapper/capture.py).
- The YOLO inference wrapper in [src/game_wrapper/detector.py](src/game_wrapper/detector.py).
- The template-matching code for elixir, cards, and end-game banners in [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py).
- The `tile_to_screen` math and the input layer in [src/game_wrapper/interact.py](src/game_wrapper/interact.py).
- The `GameWrapper` class in [src/game_wrapper/wrapper.py](src/game_wrapper/wrapper.py).
- Initial scaffolds for the Gymnasium env in [src/rl/env.py](src/rl/env.py) and the MaskablePPO training loop in [src/rl/train.py](src/rl/train.py).
- First versions of the action mask, observation encoding, and reward function in [src/rl/action.py](src/rl/action.py), [src/rl/obs.py](src/rl/obs.py), and [src/rl/reward.py](src/rl/reward.py).
- The `pyproject.toml` for the editable install.

### What I had to debug, fix, or substantially rework

The first drafts almost always had something off. Most of my time on the project was spent in the loop below:

- **All pixel calibration constants.** Grid corners, tower bounding boxes, elixir region, card slot regions, end-game banner regions, monitor offset, and crop region — I measured all of these by hand against the live BlueStacks window using my [src/scripts/coords.py](src/scripts/coords.py) helper. The initial AI-suggested values were off by full tiles in both axes.
- **Grid coordinate convention.** This took several iterations. Perception used 0-indexed coordinates while the action layer used 1-indexed, so every click landed exactly one tile off from the cell perception was reporting. I caught the mismatch by noticing that my intended "bridge" placements were landing on top of my king tower.
- **Splitting the perception grid from the action grid.** The bridge in Clash Royale is taller than a regular tile, so a single uniform 18×29 grid did not map cleanly to playable rows on my side. I made the design call to split it into an 18×14 action grid (my side only) and an 18×29 perception grid (full arena).
- **Color-invariant template matching.** Cards desaturate to grayscale when you don't have enough elixir to play them, and the original template matcher returned `"unknown"` during those frames, which made the bot think its hand was empty. The same issue surfaced later when arena-themed banner recolors broke the victory/defeat detection. I diagnosed both by watching the live overlay, and converted both the templates and the input frames to grayscale before matching.
- **Elixir reading stuck at zero.** The initial code only updated `_last_elixir` when the template match crossed a confidence threshold. When the threshold was not consistently crossed, the value never updated and remained zero, which caused the action mask to mark every card as unaffordable. I changed the logic to commit the best match unconditionally.
- **Hand stuck on a played card.** Same threshold-fallback issue, different module. The agent kept trying to play cards that were no longer in hand. I switched the fallback so any below-threshold match resolves to `"unknown"`, which the action mask correctly excludes.
- **Auto-queue between matches.** The initial implementation locked onto the exact `game_state` value at function entry and got stuck when perception briefly flickered between victory and running states. I rewrote it with a flicker-tolerance counter that requires several consecutive contradictory reads before giving up.
- **DPI scaling.** The first input layer used `pyautogui`, which operates in logical pixels. My display is at 125% scaling, so every click landed about 25% off across the arena. I diagnosed it by writing a script that moved the cursor along the diagonal of the action grid and watched the offset accumulate. I replaced `pyautogui` with `pydirectinput`, which sends physical pixel coordinates via Win32 SendInput.
- **Apparent multiple-plays-per-step bug.** I thought the agent was firing multiple actions per step because consecutive log lines showed the same wallclock timestamp. After adding millisecond precision to the timestamps I confirmed the plays were correctly spaced 500 ms apart and just landing in the same second.
- **Project structure and packaging.** I did the migration into `src/` and added `pyproject.toml` for the editable install, then traced and fixed all path references that broke from the move (`DEFAULT_WEIGHTS`, `_TEMPLATE_DIR`, the runs/checkpoints directories, etc.).
- **Action mask shape.** The initial implementation returned a `dict` with separate `"slot"` and `"spot"` keys. MaskablePPO actually expects a flat 1-D boolean array of length `sum(action_space.nvec)`, so I flattened the output.
- **Off-by-one in obs.py zoning.** Perception emits 1-indexed coordinates but the observation encoder's row thresholds were written for 0-indexed. I corrected the zone thresholds.

### YOLO model

I fine-tuned the YOLO troop detector myself on a custom Clash Royale screenshot dataset that I collected and annotated by hand. Claude drafted the training script ([src/scripts/train_yolo.py](src/scripts/train_yolo.py)) but did not generate any of the labels.

### What I want graders to take away

I understand every line of code in this repository. I can walk through any module without notes. AI assistance accelerated implementation, but where the AI suggested something I disagreed with, I pushed back or rewrote it. I exercised every code path personally during development.

## External libraries

| Library | Purpose | License |
| --- | --- | --- |
| `mss` | Fast screen capture | MIT |
| `numpy` | Array operations | BSD |
| `opencv-python` | Color conversion and template matching | Apache 2.0 |
| `pydirectinput` | Win32 SendInput-based mouse and keyboard control (DPI-correct) | MIT |
| `torch` / `torchvision` | YOLO backend | BSD |
| `ultralytics` | YOLOv8 | AGPL-3.0 |
| `gymnasium` | Standard RL environment API | MIT |
| `stable-baselines3` | PPO base implementation | MIT |
| `sb3-contrib` | `MaskablePPO` (action-masked policy training) | MIT |
| `tensorboard` | Training metric visualization | Apache 2.0 |

## Models

- **YOLOv8 (small)** from Ultralytics, used as the pretrained starting point and fine-tuned by me on my own dataset. The shipped weights at `models/best.pt` are my fine-tuned checkpoint.

## Datasets

- **Custom Clash Royale screenshot dataset.** Collected and annotated by me. Not redistributed; only the fine-tuned weights ship with this repository.
- No external datasets are used.

## Reference materials

- [Supercell's June 2025 announcement](https://x.com/ClashRoyale/status/1934600054704075062) about disabling Play Again in 2v2 modes, which I checked to confirm Play Again still works in 1v1 Trophy Road.
- The Stable-Baselines3 documentation for default PPO hyperparameter recommendations.
