"""PettingZoo AEC environment wrapping :class:`coup.BoardGameEngine`."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from coup import BoardGameEngine
from coup import constants as C

from .events import snapshot_engine
from .recorder import EventRecorder
from .rewards import agent_name, player_id, rewards_from_transition

_OBS_DIM: Optional[int] = None


def observation_dim() -> int:
    """Fixed length of :meth:`BoardGameEngine.get_observation_vector` (padded to 6)."""

    global _OBS_DIM
    if _OBS_DIM is None:
        engine = BoardGameEngine(num_players=2, seed=0)
        _OBS_DIM = int(engine.get_observation_vector(0).shape[0])
    return _OBS_DIM


def _obs_space() -> spaces.Dict:
    dim = observation_dim()
    return spaces.Dict(
        {
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(C.ACTION_SPACE_SIZE,), dtype=np.int8),
        }
    )


def _action_space() -> spaces.Discrete:
    return spaces.Discrete(C.ACTION_SPACE_SIZE)


class CoupAECEnv(AECEnv):
    """Sequential Coup environment (2-6 players) with a fixed 36-way action space.

    Observations are a Dict of the engine's numeric vector plus an action mask.
    God-view hidden cards are never included in agent observations; attach an
    :class:`EventRecorder` to log claim-vs-held truth for analysis.

    ``num_players`` is set on construction and may be overridden per episode via
    ``reset(options={"num_players": n})``. Spaces stay padded to ``MAX_PLAYERS``.
    """

    metadata = {
        "name": "coup_v0",
        "render_modes": ["human"],
        "is_parallelizable": False,
        "has_manual_policy": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        *,
        seed: Optional[int] = None,
        recorder: Optional[EventRecorder] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not 2 <= num_players <= C.MAX_PLAYERS:
            raise ValueError(f"num_players must be in [2, {C.MAX_PLAYERS}], got {num_players}")
        self.num_players = num_players
        self._seed: Optional[int] = seed
        self.recorder = recorder
        self.render_mode = render_mode
        self.possible_agents = [agent_name(i) for i in range(C.MAX_PLAYERS)]
        self._obs_space = _obs_space()
        self._act_space = _action_space()
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}
        self.engine = BoardGameEngine(num_players=num_players, seed=seed)
        self._episode_seed: Optional[int] = seed
        self.agents: list[str] = []
        self.agent_selection: str = agent_name(0)

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> None:
        opts = options or {}
        if "num_players" in opts:
            n = int(opts["num_players"])
            if not 2 <= n <= C.MAX_PLAYERS:
                raise ValueError(f"num_players must be in [2, {C.MAX_PLAYERS}], got {n}")
            self.num_players = n
        if seed is not None:
            self._seed = seed
        self._episode_seed = self._seed
        self.engine.num_players = self.num_players
        self.engine._seed = self._seed
        self.engine.reset()

        self.agents = [agent_name(i) for i in range(self.num_players)]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        current = self.engine.current_player
        self.agent_selection = agent_name(current if current is not None else 0)
        self._fill_infos()
        if self.recorder is not None:
            self.recorder.begin_episode(num_players=self.num_players, seed=self._episode_seed)

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        pid = player_id(agent)
        if pid >= self.engine.num_players:
            return self._empty_obs()
        vec = self.engine.get_observation_vector(pid).astype(np.float32, copy=False)
        mask = np.zeros(C.ACTION_SPACE_SIZE, dtype=np.int8)
        if not self.terminations.get(agent, False) and self.engine.current_player == pid:
            for move in self.engine.get_legal_moves(pid):
                mask[move] = 1
        return {"observation": vec, "action_mask": mask}

    def step(self, action: Any) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            self._sync_selection_after_dead_step()
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        pid = player_id(agent)
        pre = snapshot_engine(self.engine)
        pre_alive = dict(pre["alive"])
        info = self.engine.apply_action(pid, int(action))
        if self.recorder is not None:
            self.recorder.record_step(self.engine, pid, int(action), info, pre=pre)

        step_rewards = rewards_from_transition(pre_alive, self.engine)
        for name, value in step_rewards.items():
            if name in self.rewards:
                self.rewards[name] = value
            if value < 0:
                self.terminations[name] = True

        over, winner = self.engine.is_game_over()
        if over:
            for name in self.agents:
                self.terminations[name] = True
            if self.recorder is not None:
                self.recorder.end_episode(self.engine, seed=self._episode_seed)

        self._accumulate_rewards()
        self._fill_infos()
        self._advance_selection(previous=agent)

        if self.render_mode == "human":
            self.render()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        print(f"phase={self.engine.phase.name} agent={self.agent_selection} turn={self.engine.turn_count}")
        for line in self.engine.last_events:
            print(f"  {line}")

    def close(self) -> None:
        return

    def _empty_obs(self) -> dict[str, np.ndarray]:
        return {
            "observation": np.zeros(observation_dim(), dtype=np.float32),
            "action_mask": np.zeros(C.ACTION_SPACE_SIZE, dtype=np.int8),
        }

    def _fill_infos(self) -> None:
        for agent in self.agents:
            obs = self.observe(agent)
            self.infos[agent] = {
                "action_mask": obs["action_mask"],
                "phase": self.engine.phase.name,
                "legal_moves": np.flatnonzero(obs["action_mask"]).tolist(),
            }

    def _advance_selection(self, previous: str) -> None:
        """Pick the next living decision-maker, or drain terminated agents."""

        current = self.engine.current_player
        if current is not None:
            nxt = agent_name(current)
            if nxt in self.agents and not self.terminations[nxt]:
                self.agent_selection = nxt
                return
        # Game over (or no living decision): keep a terminated agent selected
        # so AEC ``step(None)`` / ``_was_dead_step`` can drain ``self.agents``.
        if previous in self.agents:
            self.agent_selection = previous
        elif self.agents:
            self.agent_selection = self.agents[0]

    def _sync_selection_after_dead_step(self) -> None:
        if not self.agents:
            return
        if self.agent_selection not in self.agents:
            self.agent_selection = self.agents[0]
