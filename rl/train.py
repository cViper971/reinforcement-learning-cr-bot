"""Train MaskablePPO on ClashRoyaleEnv.

The env runs at 2Hz wall-clock so every sample is expensive — keep n_steps,
batch sizes, and rollouts modest. Tensorboard logs go to runs/, checkpoints
to checkpoints/.

Usage:
    python -m rl.train --total-steps 50000
    python -m rl.train --resume checkpoints/last.zip
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
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
    p.add_argument("--total-steps", type=int, default=2500, help="default sized for a ~20min wall-clock session at 2Hz")
    p.add_argument("--n-steps", type=int, default=128, help="rollout length per update (~half a match at 2Hz)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995, help="high gamma since reward is sparse and matches are long")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=4, help="few epochs — sample efficiency over reuse, data is real-time")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run-name", type=str, default="cr-mppo")
    args = p.parse_args()

    runs_dir = os.path.join("runs", args.run_name)
    ckpt_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    env = ClashRoyaleEnv()

    if args.resume:
        print(f"[train] resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, tensorboard_log=runs_dir)
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
                    reset_num_timesteps=(args.resume is None))
    finally:
        last_path = os.path.join(ckpt_dir, "last.zip")
        model.save(last_path)
        print(f"[train] saved final model to {last_path}")
        env.close()


if __name__ == "__main__":
    main()
