# Clash Royale Autonomous RL Agent

This project implements a reinforcement-learning (RL) agent that plays the live mobile game **Clash Royale** in real time. The agent operates entirely from pixel data, capturing the BlueStacks emulator window, parsing the game state through a custom computer vision pipeline, and executing actions via synthesized hardware-level inputs.

## Architecture and Pipeline

The system is designed as an end-to-end pipeline bridging real-time computer vision with deep reinforcement learning. It acts as a fully autonomous agent capable of continuous self-play and training against live opponents.

### High-Fidelity Perception Layer

The perception layer extracts a structured game state from the raw video feed at 30 FPS.
- **Screen Capture**: A dedicated background thread utilizes `mss` to stream the emulator window with minimal latency.
- **Game State Extraction**:
  - **Elixir Tracking**: Implements OpenCV template matching against digit cutouts to monitor the continuous elixir pool.
  - **Hand State**: Uses grayscale template matching to identify the four currently available cards, utilizing normalized cross-correlation to remain robust against the "not enough elixir" desaturation UI overlay.
  - **Tower Health**: Applies HSV color space thresholding to isolate red and blue health bars, calculating fill ratios to determine continuous health percentages.
  - **Spatial Troop Tracking**: Deploys a fine-tuned **YOLO** (Ultralytics) object detection model to actively locate, classify, and track the bounding boxes and team affiliation (ally/enemy) of troops on the battlefield.

### Reinforcement Learning Environment

The extracted state feeds into a custom Gymnasium environment operating at a fixed 2 Hz control rate.
- **Observation Space**: The perception dictionary is encoded into a continuous 1D vector (Box space), capturing normalized elixir, tower health, one-hot encoded hand configurations, and spatial grids of active troops.
- **Action Space**: Implements a `MultiDiscrete([5, 8])` space. This corresponds to 5 slot choices (4 cards + 1 No-Op) and 8 curated canonical placement coordinates (e.g., left/right bridge, center, princess towers), significantly reducing the action space complexity compared to a full coordinate grid.
- **Dynamic Action Masking**: A deterministic action mask evaluates the agent's current hand and elixir pool. It zeroes out logits for unaffordable or unavailable cards, which is integrated directly into the `MaskablePPO` policy network to prevent invalid state transitions and accelerate exploration.
- **Reward Shaping**: The reward function calculates dense per-step signals based on the delta in tower HP between the agent and the opponent, augmented by sparse terminal rewards for tower destruction events and match outcomes.

### Autonomous Training Loop

- **Policy Optimization**: The agent is trained using `MaskablePPO` (Proximal Policy Optimization with action masking) via `sb3-contrib`. 
- **Continuous Execution**: Upon detecting end-game victory or defeat banners, the environment automatically synthesizes the required inputs to queue into the next match. This enables indefinite, uninterrupted rollouts and gradient updates.

## Quick Start

See [SETUP.md](SETUP.md) for full system prerequisites, including BlueStacks configuration, deck constraints, and monitor calibration. 

```bash
# From the repository root
pip install -r requirements.txt
pip install -e .            # Registers rl and game_wrapper as packages
python -m rl.train          # Initializes the training loop; press Q anywhere to terminate
```

To resume training from the most recent checkpoint:

```bash
python -m rl.train --resume models/checkpoints/cr-mppo/last.zip
```

## Video Documentation

- **Gameplay Demo:** [videos/demo.mp4](videos/demo.mp4)
- **Technical Walkthrough:** [videos/technical.mp4](videos/technical.mp4)

## Evaluation

The architecture demonstrates robust foundations for real-time imperfect-information games:

- **Vision Model Performance**: The custom-trained YOLO troop detector achieves strong validation metrics on the hold-out set, enabling resilient parsing of chaotic battlefield states.
- **Policy Convergence**: The `MaskablePPO` agent exhibits stable gradient updates characterized by bounded KL divergence. The integration of action masking heavily prunes the exploration tree, enabling the critic network to efficiently map the complex return landscapes of Clash Royale mechanics.
