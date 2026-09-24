"""Pre-step snapshots, structured JSONL events, and per-run directory writers."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from coup.engine import BoardGameEngine

from .events import events_from_step, snapshot_engine


def git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return HEAD sha, or ``None`` if git is unavailable."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_run_id() -> str:
    """Timestamp plus a short random suffix, safe as a directory name."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class RunWriter:
    """Append-only ``events.jsonl`` / ``episodes.jsonl`` under ``data/runs/<id>/``."""

    def __init__(self, root: Path | str, config: dict[str, Any], run_id: Optional[str] = None) -> None:
        self.run_id = run_id or make_run_id()
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        (self.dir / "metrics").mkdir(exist_ok=True)
        payload = dict(config)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("git_sha", git_sha())
        with (self.dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self._events: TextIO = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._episodes: TextIO = (self.dir / "episodes.jsonl").open("a", encoding="utf-8")

    def write_event(self, event: dict[str, Any]) -> None:
        self._events.write(json.dumps(event) + "\n")
        self._events.flush()

    def write_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.write_event(event)

    def write_episode(self, row: dict[str, Any]) -> None:
        self._episodes.write(json.dumps(row) + "\n")
        self._episodes.flush()

    def close(self) -> None:
        self._events.close()
        self._episodes.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class EventRecorder:
    """Snapshot living hands *before* each action, then emit structured events."""

    def __init__(self, writer: Optional[RunWriter] = None) -> None:
        self.writer = writer
        self.episode_id: str = "0"
        self.step: int = 0
        self.episode_index: int = 0
        self._episode_events: list[dict[str, Any]] = []

    def begin_episode(self, *, num_players: int, seed: Optional[int], episode_id: Optional[str] = None) -> None:
        self.episode_index += 1
        self.episode_id = episode_id or str(self.episode_index)
        self.step = 0
        self._episode_events = []
        meta = {
            "event_type": "episode_start",
            "episode_id": self.episode_id,
            "num_players": num_players,
            "seed": seed,
        }
        self._episode_events.append(meta)
        if self.writer is not None:
            self.writer.write_event(meta)

    def record_step(
        self,
        engine: BoardGameEngine,
        actor: int,
        action: int,
        info: dict[str, Any],
        pre: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Emit events for a step. ``pre`` must be taken *before* ``apply_action``."""

        if pre is None:
            raise ValueError("record_step requires a pre-step snapshot (hands change on apply)")
        events = events_from_step(pre, actor, action, engine, info, self.episode_id, self.step)
        self.step += 1
        self._episode_events.extend(events)
        if self.writer is not None:
            self.writer.write_events(events)
        return events

    def end_episode(self, engine: BoardGameEngine, *, seed: Optional[int] = None) -> dict[str, Any]:
        over, winner = engine.is_game_over()
        row = {
            "episode_id": self.episode_id,
            "num_players": engine.num_players,
            "seed": seed,
            "winner": winner,
            "game_over": over,
            "steps": self.step,
            "turn_count": engine.turn_count,
        }
        if self.writer is not None:
            self.writer.write_episode(row)
        return row

    def snapshot(self, engine: BoardGameEngine) -> dict[str, Any]:
        return snapshot_engine(engine)
