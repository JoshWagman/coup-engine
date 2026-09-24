"""PettingZoo API compliance and variable player-count tests for CoupAECEnv."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import api_test

from coup_rl.env import CoupAECEnv


@pytest.mark.parametrize("num_players", [2, 6])
def test_pettingzoo_api(num_players: int) -> None:
    env = CoupAECEnv(num_players=num_players, seed=0)
    api_test(env, num_cycles=1000, verbose_progress=False)


def test_reset_options_changes_agent_count() -> None:
    env = CoupAECEnv(num_players=2, seed=1)
    env.reset()
    assert len(env.agents) == 2
    env.reset(options={"num_players": 5})
    assert len(env.agents) == 5
    assert env.num_players == 5
    assert env.observation_space("player_0") == env.observation_space("player_4")
    env.reset(options={"num_players": 3}, seed=2)
    assert len(env.agents) == 3


def test_spaces_fixed_across_player_counts() -> None:
    e2 = CoupAECEnv(num_players=2, seed=0)
    e6 = CoupAECEnv(num_players=6, seed=0)
    e2.reset()
    e6.reset()
    obs2 = e2.observe("player_0")
    obs6 = e6.observe("player_0")
    assert obs2["observation"].shape == obs6["observation"].shape
    assert obs2["action_mask"].shape == obs6["action_mask"].shape == (36,)


def test_random_masked_playout_terminates() -> None:
    env = CoupAECEnv(num_players=4, seed=7)
    env.reset(seed=7)
    rng = np.random.RandomState(7)
    steps = 0
    while env.agents and steps < 20_000:
        obs, _r, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        mask = np.asarray(obs["action_mask"])
        legal = np.flatnonzero(mask)
        assert len(legal) > 0, info
        env.step(int(rng.choice(legal)))
        steps += 1
    assert not env.agents
    assert steps > 0
