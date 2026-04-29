# Clash Royale Reinforcement-Learning Bot

I created a reinforcement-learning agent that plays the live mobile game **Clash Royale** in real time, by reading raw pixels from a BlueStacks emulator window and controlling the game with simulated mouse + keyboard input.

## What it Does

This project builds an end-to-end pipeline that lets a `MaskablePPO` agent train on actual matches against live opponents. A 30 fps screen-capture thread streams the BlueStacks window into a perception layer that extracts the game state every step: elixir count (template-matched against digit cutouts), the four cards in hand (grayscale template matching to detect over the desaturated "not enough elixir" highlight), each tower's HP percentage (HSV health-bar fill ratio), end-of-match victory/defeat banners, and live troop positions from a fine-tuned YOLO detector. That perception dict feeds into a Gymnasium environment running at a 2 Hz step rate. The agent picks a (card-slot, placement-spot) tuple from a curated 8-spot action space; an action mask gated by elixir cost prevents illegal plays. After each match ends, the bot auto-queues the next one via the BlueStacks Play Again hotkey, and training runs continuously match-after-match until killed.

## Quick Start

See [SETUP.md](SETUP.md) for full prerequisites (BlueStacks configuration, deck setup, monitor calibration). Once that's done:

```bash
# from repo root
pip install -r requirements.txt
pip install -e .            # registers rl and game_wrapper as packages
python -m rl.train          # starts training; press Q anywhere to stop
```

Tensorboard:

```bash
tensorboard --logdir models/runs
```

Resume from a checkpoint:

```bash
python -m rl.train --resume models/checkpoints/cr-mppo/last.zip
```

## Video Links

- **Demo (3–5 min, non-technical):** [link to videos/demo.mp4]
- **Technical walkthrough (5–10 min):** [link to videos/walkthrough.mp4]

## Evaluation

The bot is evaluated on three axes:

1. **Pipeline correctness** — every system component runs end-to-end on the live game without intervention. Perception correctly identifies elixir, cards, tower HP, and troop positions; actions land on intended tiles; matches auto-queue on completion.
2. **Training signal** — `MaskablePPO` rollouts produce non-degenerate gradients with bounded value loss and decreasing entropy over time. Tensorboard logs in `models/runs/cr-mppo/` show `ep_rew_mean`, `entropy_loss`, `value_loss`, and `clip_fraction` curves.
3. **Match outcomes** — per-episode reward (combination of tower HP delta + tower-destroyed events + terminal win/loss) is logged to tensorboard for every match. After ~20 minutes of training, the agent transitions from random play (~−0.5 mean reward) to placing affordable cards each step without burning elixir uselessly.

Real strategic competence requires substantially more wall-clock training (training is bottlenecked at the live 2 Hz step rate of the actual game — ~170 K env steps per 24 h on a single BlueStacks instance), but the pipeline demonstrates that all components compose correctly.
