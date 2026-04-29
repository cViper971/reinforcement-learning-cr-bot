# Attribution

## How I used AI on this project

I used Claude (via the Claude Code CLI) to assist in the development of this project. Claude helped me create drafts of files quickly, but the project plan, calibration, and most of the debugging were my own work.

### Code that Claude drafted

- The threaded screen capture in [src/game_wrapper/capture.py](src/game_wrapper/capture.py).
- The YOLO inference wrapper in [src/game_wrapper/detector.py](src/game_wrapper/detector.py).
- The template-matching code for elixir, cards, and end-game banners in [src/game_wrapper/game_state.py](src/game_wrapper/game_state.py).
- The `tile_to_screen` math and the input layer in [src/game_wrapper/interact.py](src/game_wrapper/interact.py).
- The `GameWrapper` class in [src/game_wrapper/wrapper.py](src/game_wrapper/wrapper.py).
- Initial scaffolds for the Gymnasium env in [src/rl/env.py](src/rl/env.py) and the MaskablePPO training loop in [src/rl/train.py](src/rl/train.py).
- First versions of the action mask, observation encoding, and reward function in [src/rl/action.py](src/rl/action.py), [src/rl/obs.py](src/rl/obs.py), and [src/rl/reward.py](src/rl/reward.py).
- The `pyproject.toml` for the editable install.

### Design decisions, reworks, and the actual work I did

- **Action space.** I decided the list of named placement spots (left/right bridge, mid, princess, back) over a free 18×14 grid. The full grid wastes policy capacity on tactically equivalent positions and explodes credit assignment. I picked the 8 spots and tuned each spot's coordinates against the live overlay until they lined up with where I'd actually want to play that card.
- **Observation space.** I designed the encoding in [src/rl/obs.py](src/rl/obs.py): normalized elixir, six tower HPs, four one-hot hand slots, and a fixed-size padded list of ally + enemy troop entries (one-hot + 6-section spatial zone each). I picked the 5-row "bridge contention band" (rows 13–17) for the zoning instead of the strict 1-row bridge, because the area immediately in front of the bridge is tactically distinct from the back area and that's what matters for placement decisions.
- **Action masking.** I specified that the mask gates slots by elixir affordability and unknown-card status so the policy can't waste rollout steps on illegal actions. Configured the flat-array shape to match what `MaskablePPO` consumes.
- **Reward shaping.** I picked the components and weights in [src/rl/reward.py](src/rl/reward.py): a small per-step term on net tower HP delta, a medium event term on princess kills, a large event term on king kills, and a terminal win/loss bonus. Edge-detected tower-destroyed events on HP crossing zero so a destroyed tower doesn't get re-rewarded every frame.
- **Step rate and 50 ms key-to-click delay.** I picked 2 Hz as the action rate (fast enough to react to elixir changes, slow enough that perception + YOLO finish in budget) and added the 50 ms gap between card-key press and tile click after I noticed BlueStacks dropping clicks when the events were fired back-to-back.
- **`GameWrapper` abstraction.** I designed the single class exposing only `get_state()`, `act()`, `reset_match()`, and `close()`. Everything else — screen capture, YOLO inference, perception extraction, input simulation, auto-queue — is hidden. The RL side has no idea how the game is being read or driven.
- **Module split inside `game_wrapper`.** I reorganized the package into `capture` (screen grab), `detector` (YOLO), `game_state` (perception), `interact` (input + auto-queue), and `wrapper` (the public class). I refactored and regrouped these multiple times until each file had one responsibility
- **Color-based template matching.** I noticed cards were registering as `"unknown"` whenever I couldn't afford them, because the matcher was failing on the desaturated highlight. Same problem hit the victory/defeat banners later when the arena recolor changed them. Asked for grayscale on both sides.
- **Auto-queue hang.** Bot would sit forever after a match because the settle loop was tracking the exact `game_state` value and getting stuck on perception flicker. I asked for a flicker-tolerance counter that needs several consecutive contradictory reads before exiting.
l times when something looked off.
- **BlueStacks setup.** Bound key `1` over the Play Again button, set window position and resolution, configured card hotkeys, played manually past Arena 1 to unlock the Play Again UI in the first place.

### YOLO troop detector

I fine-tuned the YOLO troop detector myself. I used a publicly available Clash Royale dataset from Roboflow ([clash-royale-qexh0](https://universe.roboflow.com/enrique-uu9rp/clash-royale-qexh0)) as the training data, ran the training on Google Colab to produce `models/best.pt`, and integrated the resulting weights into perception. Claude drafted the training script ([src/scripts/train_yolo.py](src/scripts/train_yolo.py)) but did not select the dataset, configure the run, or generate any labels.

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

- **Custom Clash Royale screenshot dataset.** Found on Roboflow: https://universe.roboflow.com/enrique-uu9rp/clash-royale-qexh0
