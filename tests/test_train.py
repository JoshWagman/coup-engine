"""Smoke test: a few PPO updates write a run directory."""

from __future__ import annotations

from pathlib import Path

from coup_rl.train import TrainConfig, train


def test_training_smoke(tmp_path: Path) -> None:
    config = TrainConfig(
        num_players=2,
        total_timesteps=64,
        n_steps=32,
        n_epochs=1,
        minibatch_size=16,
        seed=0,
        device="cpu",
        run_root=tmp_path,
        checkpoint_every=1,
        learning_rate=3e-4,
    )
    run_dir = train(config)
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "episodes.jsonl").exists()
    assert (run_dir / "checkpoints" / "final.pt").exists()
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "claim" in events or "decision" in events
    episodes = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").strip()
    assert episodes
