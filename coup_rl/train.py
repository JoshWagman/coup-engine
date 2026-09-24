"""Shared-policy MaskablePPO self-play for :class:`CoupAECEnv`.

SuperSuit's AEC→vec conversion is a poor fit: only one seat acts at a time,
and elimination rewards often belong to a non-acting agent. This trainer
collects complete per-seat trajectories, then runs a PPO update on the
concatenated batch (all seats share one :class:`MaskableActorCriticPolicy`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter

from coup.constants import ACTION_SPACE_SIZE, Character

from .analytics import claim_truth_table
from .env import CoupAECEnv, observation_dim
from .recorder import EventRecorder, RunWriter


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    mask: np.ndarray
    reward: float = 0.0
    done: bool = False


@dataclass
class TrainConfig:
    num_players: int = 4
    total_timesteps: int = 200_000
    n_steps: int = 2048
    n_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    run_root: Path = field(default_factory=lambda: Path("data/runs"))
    checkpoint_every: int = 10


def _masked_policy(device: torch.device, lr: float) -> MaskableActorCriticPolicy:
    from gymnasium.spaces import Box, Discrete

    obs_space = Box(low=-np.inf, high=np.inf, shape=(observation_dim(),), dtype=np.float32)
    act_space = Discrete(ACTION_SPACE_SIZE)
    policy = MaskableActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: lr,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    return policy.to(device)


def _as_tensor(policy: MaskableActorCriticPolicy, obs: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=policy.device).unsqueeze(0)
    return obs_t, mask_t


def select_action(
    policy: MaskableActorCriticPolicy,
    obs: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, float, float]:
    """Sample a masked action; return ``(action, log_prob, value)``."""

    policy.set_training_mode(False)
    with torch.no_grad():
        obs_t, mask_t = _as_tensor(policy, obs, mask)
        actions, values, log_probs = policy(obs_t, action_masks=mask_t, deterministic=False)
    return int(actions.item()), float(log_probs.item()), float(values.item())


def compute_gae(transitions: list[Transition], gamma: float, gae_lambda: float) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    gae = 0.0
    next_value = 0.0
    for trans in reversed(transitions):
        nonterminal = 0.0 if trans.done else 1.0
        delta = trans.reward + gamma * next_value * nonterminal - trans.value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.append(gae)
        next_value = trans.value
    advantages.reverse()
    returns = [adv + t.value for adv, t in zip(advantages, transitions)]
    return advantages, returns


def _flatten(completed: list[list[Transition]], gamma: float, gae_lambda: float) -> dict[str, np.ndarray]:
    obs: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    masks: list[np.ndarray] = []
    advantages: list[float] = []
    returns: list[float] = []
    for traj in completed:
        if not traj:
            continue
        adv, ret = compute_gae(traj, gamma, gae_lambda)
        for trans, a, r in zip(traj, adv, ret):
            obs.append(trans.obs)
            actions.append(trans.action)
            log_probs.append(trans.log_prob)
            values.append(trans.value)
            masks.append(trans.mask)
            advantages.append(a)
            returns.append(r)
    adv_arr = np.asarray(advantages, dtype=np.float32)
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
    return {
        "obs": np.stack(obs, axis=0),
        "actions": np.asarray(actions, dtype=np.int64),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "masks": np.stack(masks, axis=0).astype(bool),
        "advantages": adv_arr,
        "returns": np.asarray(returns, dtype=np.float32),
    }


def ppo_update(policy: MaskableActorCriticPolicy, batch: dict[str, np.ndarray], config: TrainConfig) -> dict[str, float]:
    policy.set_training_mode(True)
    n = batch["obs"].shape[0]
    if n == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    mb = min(config.minibatch_size, n)
    idx_all = np.arange(n)
    last: dict[str, float] = {}
    for _ in range(config.n_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, n, mb):
            idx = idx_all[start : start + mb]
            obs = torch.as_tensor(batch["obs"][idx], dtype=torch.float32, device=policy.device)
            actions = torch.as_tensor(batch["actions"][idx], dtype=torch.int64, device=policy.device)
            old_logp = torch.as_tensor(batch["log_probs"][idx], dtype=torch.float32, device=policy.device)
            advantages = torch.as_tensor(batch["advantages"][idx], dtype=torch.float32, device=policy.device)
            returns = torch.as_tensor(batch["returns"][idx], dtype=torch.float32, device=policy.device)
            masks = torch.as_tensor(batch["masks"][idx], dtype=torch.bool, device=policy.device)
            values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
            values = values.flatten()
            ratio = torch.exp(log_prob - old_logp)
            clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            policy.optimizer.step()
            last = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.mean().item()),
            }
    return last


def _play_episode(
    env: CoupAECEnv,
    policy: MaskableActorCriticPolicy,
    seed: Optional[int],
) -> tuple[list[list[Transition]], dict[str, Any], list[dict[str, Any]]]:
    env.reset(seed=seed)
    buffers: dict[str, list[Transition]] = {a: [] for a in env.agents}
    collected_events: list[dict[str, Any]] = []
    recorder = env.recorder

    while env.agents:
        agent = env.agent_selection
        obs, _reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        mask = np.asarray(obs["action_mask"], dtype=np.int8)
        if not mask.any():
            mask = np.asarray(info.get("action_mask", mask), dtype=np.int8)
        action, logp, value = select_action(policy, obs["observation"], mask.astype(bool))
        env.step(action)
        trans = Transition(
            obs=np.asarray(obs["observation"], dtype=np.float32),
            action=action,
            log_prob=logp,
            value=value,
            mask=mask.astype(bool),
            reward=float(env.rewards.get(agent, 0.0)),
            done=bool(env.terminations.get(agent, False)),
        )
        buffers[agent].append(trans)
        for other, other_buf in buffers.items():
            if other == agent or not other_buf:
                continue
            extra = float(env.rewards.get(other, 0.0))
            if extra != 0.0:
                other_buf[-1].reward += extra
            if env.terminations.get(other, False):
                other_buf[-1].done = True

    if recorder is not None:
        collected_events = list(recorder._episode_events)
        winner = None
        for ev in reversed(collected_events):
            if ev.get("event_type") == "game_end":
                winner = ev.get("winner")
                break
        summary = {"winner": winner, "steps": recorder.step, "seed": seed}
    else:
        summary = {"winner": None, "steps": sum(len(b) for b in buffers.values()), "seed": seed}
    return list(buffers.values()), summary, collected_events


def _claim_metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    overall = claim_truth_table(events)
    if overall["overall"]["truth_rate"] is not None:
        metrics["claims/truth_rate"] = float(overall["overall"]["truth_rate"])
        metrics["claims/bluff_rate"] = float(overall["overall"]["bluff_rate"])
    for char in Character:
        table = claim_truth_table(events, char.name)
        rate = table["overall"]["truth_rate"]
        if rate is not None:
            metrics[f"claims/{char.name.lower()}_truth_rate"] = float(rate)
            metrics[f"claims/{char.name.lower()}_bluff_rate"] = float(table["overall"]["bluff_rate"])
            metrics[f"claims/{char.name.lower()}_n"] = float(table["overall"]["claims"])
    return metrics


def train(config: TrainConfig) -> Path:
    """Run self-play training; return the run directory."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    policy = _masked_policy(device, config.learning_rate)

    run_config = {
        "num_players": config.num_players,
        "total_timesteps": config.total_timesteps,
        "n_steps": config.n_steps,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "reward_spec": "elimination=-1, winner=+1",
        "algorithm": "maskable_ppo_shared_selfplay",
    }
    writer = RunWriter(config.run_root, run_config)
    recorder = EventRecorder(writer)
    env = CoupAECEnv(num_players=config.num_players, seed=config.seed, recorder=recorder)
    tb = SummaryWriter(log_dir=str(writer.dir / "metrics"))

    timesteps = 0
    updates = 0
    games = 0
    episode_seed = config.seed
    try:
        while timesteps < config.total_timesteps:
            completed: list[list[Transition]] = []
            update_events: list[dict[str, Any]] = []
            winners: list[Optional[int]] = []
            lengths: list[int] = []
            collected = 0
            while collected < config.n_steps and timesteps + collected < config.total_timesteps:
                trajs, summary, events = _play_episode(env, policy, episode_seed)
                episode_seed += 1
                games += 1
                update_events.extend(events)
                winners.append(summary.get("winner"))
                lengths.append(int(summary.get("steps") or 0))
                for traj in trajs:
                    if traj:
                        completed.append(traj)
                        collected += len(traj)
            if not completed:
                break
            batch = _flatten(completed, config.gamma, config.gae_lambda)
            losses = ppo_update(policy, batch, config)
            timesteps += collected
            updates += 1

            tb.add_scalar("time/timesteps", timesteps, timesteps)
            tb.add_scalar("episode/length_mean", float(np.mean(lengths) if lengths else 0.0), timesteps)
            tb.add_scalar("episode/games", games, timesteps)
            n_players = config.num_players
            for seat in range(n_players):
                if winners:
                    rate = sum(1 for w in winners if w == seat) / len(winners)
                    tb.add_scalar(f"episode/win_rate_seat_{seat}", rate, timesteps)
            for key, value in _claim_metrics(update_events).items():
                tb.add_scalar(key, value, timesteps)
            for key, value in losses.items():
                tb.add_scalar(f"train/{key}", value, timesteps)
            tb.flush()

            if updates % config.checkpoint_every == 0:
                ckpt = writer.dir / "checkpoints" / f"update_{updates}.pt"
                torch.save({"policy": policy.state_dict(), "updates": updates, "timesteps": timesteps}, ckpt)
    finally:
        ckpt = writer.dir / "checkpoints" / "final.pt"
        torch.save({"policy": policy.state_dict(), "updates": updates, "timesteps": timesteps}, ckpt)
        tb.close()
        env.close()
        writer.close()
    return writer.dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shared-policy MaskablePPO self-play on Coup")
    parser.add_argument("--num-players", type=int, default=4, help="Seats for this training run (2-6)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-steps", type=int, default=2048, help="Decisions collected per PPO update")
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args(argv)
    config = TrainConfig(
        num_players=args.num_players,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        seed=args.seed,
        device=args.device,
        run_root=args.run_root,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
    )
    run_dir = train(config)
    print(f"Run written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
