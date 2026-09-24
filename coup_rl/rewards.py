"""Agent name helpers and sparse win/loss rewards for the PettingZoo wrapper."""

from __future__ import annotations

from typing import Mapping

from coup.engine import BoardGameEngine


def agent_name(player_id: int) -> str:
    """Return the PettingZoo agent id for a seat (``player_0``, ...)."""

    return f"player_{player_id}"


def player_id(agent: str) -> int:
    """Parse a seat index from a ``player_N`` agent id."""

    return int(agent.split("_")[1])


def rewards_from_transition(
    pre_alive: Mapping[int, bool],
    engine: BoardGameEngine,
) -> dict[str, float]:
    """Sparse rewards: ``-1`` on elimination, ``+1`` for the winner at game over.

    Credit can land on a different agent than the one who just acted. The sum
    is ``2 - n`` when a game ends with one winner and ``n - 1`` losers, so it
    is not zero-sum for ``n > 2``.
    """

    rewards = {agent_name(i): 0.0 for i in range(engine.num_players)}
    for pid, was_alive in pre_alive.items():
        if was_alive and not engine.players[pid].alive:
            rewards[agent_name(pid)] = -1.0
    over, winner = engine.is_game_over()
    if over and winner is not None:
        rewards[agent_name(winner)] = 1.0
    return rewards
