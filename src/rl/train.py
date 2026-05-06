"""Train MaskablePPO on ClashRoyaleEnv.

The env runs at 2Hz wall-clock so every sample is expensive — keep n_steps,
batch sizes, and rollouts modest. Tensorboard logs go to runs/, checkpoints
to checkpoints/.

Usage:
    python -m rl.train --total-steps 50000
    python -m rl.train --resume models/checkpoints/cr-mppo/last.zip
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from rl.env import ClashRoyaleEnv, _kill

class KillSwitchCallback(BaseCallback):
    """Stop model.learn() cleanly when the global Q-kill flag fires."""
    def _on_step(self) -> bool:
        if _kill.is_set():
            if self.verbose:
                print("[train] Q pressed — stopping training")
            return False
        return True


def main():
    p = argparse.ArgumentParser()
    # Default is effectively unbounded — training runs until Q is pressed.
    # Override with --total-steps if you want a hard cap.
    p.add_argument("--total-steps", type=int, default=10**9)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run-name", type=str, default="cr-mppo")
    args = p.parse_args()

    # Resolve to repo-root models/ regardless of cwd, so artifacts always
    # land in the same place per the project layout (models/ is a
    # rubric-required top-level dir).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    runs_dir = os.path.join(repo_root, "models", "runs", args.run_name)
    ckpt_dir = os.path.join(repo_root, "models", "checkpoints", args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    env = ClashRoyaleEnv()

    # Auto-resume from last.zip if present, so re-running the command always
    # continues progressively. Pass --resume <path> to override, or delete
    # last.zip to start fresh.
    resume_path = args.resume
    if resume_path is None:
        candidate = os.path.join(ckpt_dir, "last.zip")
        if os.path.exists(candidate):
            resume_path = candidate

    if resume_path:
        print(f"[train] resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env, tensorboard_log=runs_dir)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            tensorboard_log=runs_dir,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(save_freq=args.save_every, save_path=ckpt_dir,
                           name_prefix="cr"),
        KillSwitchCallback(verbose=1),
    ]

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=(resume_path is None))
    finally:
        last_path = os.path.join(ckpt_dir, "last.zip")
        model.save(last_path)
        print(f"[train] saved final model to {last_path}")
        env.close()


if __name__ == "__main__":
    main()
