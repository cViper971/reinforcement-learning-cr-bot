# Setup

This project drives a live BlueStacks instance running Clash Royale. Setup has two parts: **software install** and **environment calibration**.

## 1. Software prerequisites

- **OS**: Windows 10/11 (the input layer uses Win32 APIs via `pydirectinput`, and the screen capture uses `mss` with Win32 monitor enumeration).
- **Python 3.10+** with `pip` and `venv`.
- **NVIDIA GPU recommended** for YOLO inference at 2 Hz. CPU works but adds ~200 ms per step; CUDA 11+ + matching `torch` build is strongly preferred.
- **BlueStacks 5+** with Clash Royale installed.

### Install

```bash
git clone <this-repo-url> cr-bot
cd cr-bot
python -m venv .venv
.venv\Scripts\activate         # Windows PowerShell or cmd
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                # registers rl/ and game_wrapper/ as importable packages
```

`pip install -e .` reads `pyproject.toml` and registers `rl` and `game_wrapper` from `src/`. After this, `python -m rl.train` works from any directory.

## 2. Game-side prerequisites

Before training, your Clash Royale account must satisfy these conditions or the bot won't be able to play matches back-to-back:

- **Account is past Arena 1.** New-account onboarding cutscenes block the Play Again button until you've reached Arena 2 (~300 trophies). Play matches manually until the post-match screen shows "Play Again" before training.
- **Deck contains exactly the 8 cards listed in [src/rl/action.py](src/rl/action.py)'s `CARD_COSTS` dict** (archers, giant, goblins, knight, mini_pekka, minions, musketeer, spear_goblins). Other cards will register as `"unknown"` in perception and be masked out of the action space.

## 3. BlueStacks configuration

- **Resolution**: 1080×1920 portrait (the perception bounding boxes in [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py) are calibrated against this).
- **Window position**: place the BlueStacks window so its top-left aligns with screen pixel `(655, 0)` on monitor 1 (default `CROP_REGION` in [src/game_wrapper/wrapper.py](src/game_wrapper/wrapper.py)). If your setup differs, adjust `CROP_REGION` and `TARGET_MONITOR` in `wrapper.py`.
- **Key bindings**: BlueStacks must have key `1` mapped to the Play Again button (Settings → Game Controls → Click & Tap, bind key `1` over the Play Again button location). The bot relies on this for auto-queue.
- **Card hotkeys**: keys `1`–`4` should drop the corresponding hand slot (BlueStacks usually does this by default).

## 4. Calibration

If the perception bounding boxes or grid corners don't line up with your window, recalibrate using the coords helper. From repo root:

```bash
python -m scripts.coords
```

Hover the cursor over a target pixel and press SPACE to print its frame-relative coords. Q to quit. Use this to update:

- `_GRID_BL`, `_GRID_TR` in [src/game_wrapper/interact.py](src/game_wrapper/interact.py) (action grid: bottom-left and top-right corners of YOUR side of the arena).
- `_GRID_BL`, `_GRID_TR` in [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py) (perception grid: full 18×29 arena).
- `_TOWER_BOXES`, `_ELIXIR_NUM_BOX`, `_CARD_BOXES`, `_ENDGAME_BOXES` in [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py) if your window position differs.

## 5. Run

```bash
# manual: queue a match in BlueStacks first, then start the trainer
python -m rl.train

# or with custom args
python -m rl.train --total-steps 5000 --run-name smoke-test
```

Press **Q** at any time (works globally; BlueStacks can have focus) to stop training cleanly. The final model saves to `models/checkpoints/<run-name>/last.zip`.

Tensorboard:

```bash
tensorboard --logdir models/runs
```

Open http://localhost:6006 in a browser to see live training curves.
