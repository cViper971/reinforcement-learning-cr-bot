# Clash Royale Reinforcement-Learning Bot

I created a reinforcement-learning agent that plays the live mobile game **Clash Royale** in real time, by reading raw pixels from a BlueStacks emulator window and controlling the game with simulated mouse + keyboard input.

## What it Does

If there's ever been a time you've wanted to play Clash Royale and had no friends online, or if you've ever just wanted to train against a bot to improve your own skills, there currently aren't many options. That's why I created a reinforcement learning bot to play against!

This project builds an end-to-end pipeline that lets a `MaskablePPO` agent train on actual matches against live opponents. A 30 fps screen-capture thread streams the BlueStacks window into a perception layer that extracts the game state every step: elixir count (template-matched against digit cutouts), the four cards in hand (grayscale template matching to detect over the desaturated "not enough elixir" highlight), each tower's HP percentage (HSV health-bar fill ratio), end-of-match victory/defeat banners, and live troop positions from a fine-tuned YOLO detector. That perception dict feeds into a Gymnasium environment running at a 2 Hz step rate. The agent picks a (card-slot, placement-spot) tuple from a curated 8-spot action space; an action mask gated by elixir cost prevents illegal plays. After each match ends, the bot auto-queues the next one via the BlueStacks Play Again hotkey, and training runs continuously match-after-match until killed.

## Quick Start

See [SETUP.md](SETUP.md) for full prerequisites (BlueStacks configuration, deck setup, monitor calibration). Once that's done:

```bash
# from repo root
pip install -r requirements.txt
pip install -e .            # registers rl and game_wrapper as packages
python -m rl.train          # starts training; press Q anywhere to stop
```

Resume from a checkpoint:

```bash
python -m rl.train --resume models/checkpoints/cr-mppo/last.zip
```

## Video Links

- **Demo (3–5 min, non-technical):** [videos/demo.mp4](videos/demo.mp4)
- **Technical walkthrough (5–10 min):** [videos/technical.mp4](videos/technical.mp4)

## Evaluation

### YOLO troop detector (fine-tuned)

Validation metrics from `models/best.pt` against the held-out split of the Roboflow dataset (reproduce with `yolo val model=models/best.pt data=<your_data.yaml>`):

| Metric         | Value |
|----------------|-------|
| mAP@0.5        | 0.759 |
| mAP@0.5-0.95   | 0.567 |
| Precision      | 0.859 |
| Recall         | 0.669 |

### MaskablePPO training

Snapshot from `models/runs/cr-mppo/`:

| Metric               | Value     |
|----------------------|-----------|
| `ep_rew_mean`        | -0.49     |
| `ep_len_mean`        | 334 steps (≈3 min CR matches) |
| `entropy_loss`       | -2.25     |
| `value_loss`         | 1.28      |
| `clip_fraction`      | 0.00      |
| `approx_kl`          | 2.7e-04   |
| `explained_variance` | 0.087     |
| `fps`                | 1.98 (matches the 2 Hz nominal step rate) |

Reading the metrics: bounded `approx_kl` and zero `clip_fraction` confirm PPO updates are stable; high entropy means exploration is still active; positive `explained_variance` means the critic is starting to fit returns. `ep_rew_mean` near zero is consistent with near-random play — the run is far short of strategic convergence (typical RL on similar live games needs ≥100 K env steps), but every gradient signal is healthy.

Still, I ultimately didn't have as much time to train the model as I wanted, resulting in performance that was sub-par.
