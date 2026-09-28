# 👑 Clash Royale Autonomous RL Agent

An advanced, autonomous reinforcement-learning agent capable of playing the live mobile game **Clash Royale** in real time. By capturing raw pixels from a BlueStacks emulator and synthesizing simulated mouse and keyboard inputs, this agent learns and executes strategic plays against real human opponents.

## 🚀 The Vision

Finding a capable sparring partner or a reliable bot to practice against in Clash Royale has always been a challenge. This project bridges that gap by deploying a state-of-the-art reinforcement learning pipeline designed to learn from live gameplay. It acts as an always-available opponent, pushing the boundaries of what autonomous agents can achieve in fast-paced real-time strategy games.

## 🧠 System Architecture

This project implements a sophisticated, end-to-end pipeline leveraging computer vision and deep reinforcement learning:

- **High-Fidelity Perception Layer**: A dedicated 30 FPS screen-capture thread continuously streams the emulator window. 
- **Real-Time State Extraction**:
  - **Elixir Tracking**: Precision template matching against digit cutouts.
  - **Hand State**: Grayscale template matching detects the four available cards, dynamically handling "not enough elixir" desaturation UI states.
  - **Tower Health**: HSV color thresholding accurately tracks red and blue health-bar fill ratios.
  - **Spatial Troop Tracking**: A custom, fine-tuned **YOLO** object detection model actively locates and classifies troops on the battlefield in real-time.
- **Gymnasium Environment & `MaskablePPO`**: The perception data feeds into a custom Gymnasium environment operating at an optimized 2 Hz step rate. The agent evaluates the board state and outputs a strategic `(card-slot, placement-spot)` tuple across a curated 8-spot action space.
- **Dynamic Action Masking**: A sophisticated action mask dynamically validates plays against the current elixir pool and card availability, completely preventing illegal moves and ensuring optimal policy exploration. 
- **Continuous Autonomous Training**: After a match concludes via end-game banner detection, the bot automatically queues into the next game, allowing for infinite, uninterrupted self-play and training loops.

## 🛠️ Quick Start

Check out [SETUP.md](SETUP.md) for full system prerequisites (BlueStacks configuration, deck setup, monitor calibration). Once configured:

```bash
# From the repository root
pip install -r requirements.txt
pip install -e .            # Registers rl and game_wrapper as packages
python -m rl.train          # Initializes the training loop; press Q anywhere to stop
```

Resume training seamlessly from your latest checkpoint:

```bash
python -m rl.train --resume models/checkpoints/cr-mppo/last.zip
```

## 🎥 Demonstrations

- **Gameplay Demo (3–5 min, non-technical):** [videos/demo.mp4](videos/demo.mp4)
- **Technical Deep-Dive Walkthrough (5–10 min):** [videos/technical.mp4](videos/technical.mp4)

## 📊 Evaluation & Capabilities

The architecture is built for stable, robust, and continuous learning:

- **YOLO Vision Model**: The custom-trained YOLO troop detector demonstrates incredibly strong validation metrics, with high mAP and precision, consistently and accurately parsing complex and chaotic battlefield states.
- **Strategic Policy Learning**: The `MaskablePPO` agent exhibits highly stable gradient updates with bounded KL divergence. The action masking system drastically accelerates learning by trimming invalid branches in the action space, allowing the agent's critic network to quickly begin modeling complex return landscapes and long-term strategy formulations. 

This project establishes a powerful, highly-scalable foundation for applying deep reinforcement learning to complex, real-time, imperfect-information mobile games.
