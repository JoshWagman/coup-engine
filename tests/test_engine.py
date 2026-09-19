"""Tests for the Coup engine: masking invariants, rules edge cases, and that
many seeded random playouts always terminate with exactly one winner."""

from __future__ import annotations

import random

import pytest

from coup import BoardGameEngine, Character, Phase
from coup import constants as C
from coup.engine import IllegalActionError


def play_random(engine: BoardGameEngine, seed: int, max_decisions: int = 20_000) -> int:
    """Drive ``engine`` to completion with random legal moves; return decisions used."""

    rng = random.Random(seed)
    decisions = 0
    while not engine.is_game_over()[0]:
        assert decisions < max_decisions, "game failed to terminate"
        actor = engine.current_player
        assert actor is not None
        legal = engine.get_legal_moves(actor)
        assert legal, "non-terminal state offered no legal moves"
        engine.apply_action(actor, rng.choice(legal))
        decisions += 1
    return decisions


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_setup_deals_correctly(num_players: int) -> None:
    engine = BoardGameEngine(num_players=num_players, seed=1)
    assert len(engine.players) == num_players
    for p in engine.players:
        assert p.coins == C.STARTING_COINS
        assert p.influence_count == C.STARTING_INFLUENCE
        assert p.revealed == []
    # 15 total cards - 2 per player still in deck.
    assert len(engine.deck) == 15 - C.STARTING_INFLUENCE * num_players
    assert engine.phase == Phase.AWAIT_ACTION
    assert engine.current_player == 0


def test_invalid_num_players() -> None:
    with pytest.raises(ValueError):
        BoardGameEngine(num_players=1)
    with pytest.raises(ValueError):
        BoardGameEngine(num_players=7)


def test_seed_reproducibility() -> None:
    a = BoardGameEngine(num_players=4, seed=123)
    b = BoardGameEngine(num_players=4, seed=123)

    rng_a, rng_b = random.Random(9), random.Random(9)
    for _ in range(200):
        if a.is_game_over()[0] or b.is_game_over()[0]:
            break
        pa, pb = a.current_player, b.current_player
        assert pa == pb
        la, lb = a.get_legal_moves(pa), b.get_legal_moves(pb)
        assert la == lb
        act_a = rng_a.choice(la)
        act_b = rng_b.choice(lb)
        assert act_a == act_b
        a.apply_action(pa, act_a)
        b.apply_action(pb, act_b)

    assert a.is_game_over() == b.is_game_over()


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_random_playouts_terminate_with_one_winner(num_players: int) -> None:
    for seed in range(30):
        engine = BoardGameEngine(num_players=num_players, seed=seed)
        play_random(engine, seed=seed)
        over, winner = engine.is_game_over()
        assert over
        alive = [p.player_id for p in engine.players if p.alive]
        assert alive == [winner]
        assert len(alive) == 1


def test_legal_moves_only_for_current_player() -> None:
    engine = BoardGameEngine(num_players=3, seed=2)
    current = engine.current_player
    for pid in range(3):
        if pid != current:
            assert engine.get_legal_moves(pid) == []


def test_masking_never_targets_self_or_dead() -> None:
    for seed in range(20):
        engine = BoardGameEngine(num_players=4, seed=seed)
        rng = random.Random(seed)
        while not engine.is_game_over()[0]:
            pid = engine.current_player
            assert pid is not None
            for action in engine.get_legal_moves(pid):
                for base in (C.COUP_BASE, C.ASSASSINATE_BASE, C.STEAL_BASE):
                    if C.is_in_range(action, base, C.MAX_PLAYERS):
                        target = C.decode_target(action, base)
                        assert target != pid, "cannot target self"
                        assert target < engine.num_players
                        assert engine.players[target].alive, "cannot target dead player"
            engine.apply_action(pid, rng.choice(engine.get_legal_moves(pid)))


def test_illegal_action_rejected() -> None:
    engine = BoardGameEngine(num_players=3, seed=4)
    pid = engine.current_player
    assert pid is not None
    # Wrong player.
    with pytest.raises(IllegalActionError):
        engine.apply_action((pid + 1) % 3, C.INCOME)
    # An action never legal during AWAIT_ACTION (a challenge).
    with pytest.raises(IllegalActionError):
        engine.apply_action(pid, C.CHALLENGE)


def test_income_increments_and_passes_turn() -> None:
    engine = BoardGameEngine(num_players=3, seed=5)
    p0_coins = engine.players[0].coins
    engine.apply_action(0, C.INCOME)
    assert engine.players[0].coins == p0_coins + 1
    assert engine.current_turn_player == 1
    assert engine.phase == Phase.AWAIT_ACTION


def test_forced_coup_at_ten_coins() -> None:
    engine = BoardGameEngine(num_players=3, seed=6)
    engine.players[0].coins = 10
    engine._decision_player = 0
    engine.current_turn_player = 0
    legal = engine.get_legal_moves(0)
    assert legal, "must have moves"
    assert all(C.is_in_range(a, C.COUP_BASE, C.MAX_PLAYERS) for a in legal)


def test_coup_costs_seven_and_removes_influence() -> None:
    engine = BoardGameEngine(num_players=2, seed=8)
    engine.players[0].coins = 7
    engine._decision_player = 0
    engine.current_turn_player = 0
    engine.apply_action(0, C.COUP_BASE + 1)
    assert engine.players[0].coins == 0
    # P1 has two cards, so a Coup pauses for them to choose which to reveal.
    assert engine.phase == Phase.AWAIT_LOSE_INFLUENCE
    assert engine.current_player == 1
    engine.apply_action(1, C.LOSE_SLOT_BASE + 0)
    assert engine.players[1].influence_count == 1
    # Turn passes to the next living player (P1) in the 2-player game.
    assert engine.current_turn_player == 1
    assert engine.phase == Phase.AWAIT_ACTION


def test_exchange_keeps_correct_count() -> None:
    engine = BoardGameEngine(num_players=3, seed=11)
    engine._decision_player = 0
    engine.current_turn_player = 0
    before = engine.players[0].influence_count
    engine.apply_action(0, C.EXCHANGE)
    # Everyone passes the challenge.
    while engine.phase == Phase.AWAIT_CHALLENGE_ACTION:
        engine.apply_action(engine.current_player, C.PASS)
    assert engine.phase == Phase.AWAIT_EXCHANGE
    legal = engine.get_legal_moves(0)
    assert all(C.is_in_range(a, C.EXCHANGE_KEEP_BASE, C.MAX_EXCHANGE_COMBOS) for a in legal)
    engine.apply_action(0, legal[0])
    assert engine.players[0].influence_count == before


def test_tax_gives_three_coins_when_unchallenged() -> None:
    engine = BoardGameEngine(num_players=3, seed=13)
    engine._decision_player = 0
    engine.current_turn_player = 0
    before = engine.players[0].coins
    engine.apply_action(0, C.TAX)
    while engine.phase == Phase.AWAIT_CHALLENGE_ACTION:
        engine.apply_action(engine.current_player, C.PASS)
    assert engine.players[0].coins == before + 3


def test_conservation_of_cards() -> None:
    """All 15 cards are always accounted for, and the number of cards held by
    players (living + revealed) stays constant at 2 * num_players."""

    for seed in range(15):
        engine = BoardGameEngine(num_players=4, seed=seed)
        rng = random.Random(seed)
        expected_out = 2 * engine.num_players
        while not engine.is_game_over()[0]:
            held = sum(p.influence_count + len(p.revealed) for p in engine.players)
            assert held == expected_out
            # Mid-Exchange, 2 freshly drawn cards live in the exchange hand.
            in_exchange = len(engine._exchange_hand) - engine.players[engine.pending.actor].influence_count if engine._exchange_hand is not None else 0
            assert held + len(engine.deck) + in_exchange == 15
            pid = engine.current_player
            engine.apply_action(pid, rng.choice(engine.get_legal_moves(pid)))


def test_observation_partial_hides_opponent_cards() -> None:
    engine = BoardGameEngine(num_players=3, seed=15)
    obs = engine.get_observation(player_id=0)
    assert obs["players"][0]["hidden_cards"] is not None
    assert obs["players"][1]["hidden_cards"] is None
    assert obs["players"][2]["hidden_cards"] is None
    # Full view reveals everyone.
    full = engine.get_observation()
    assert all(p["hidden_cards"] is not None for p in full["players"])


def test_observation_vector_is_fixed_length() -> None:
    e2 = BoardGameEngine(num_players=2, seed=1)
    e6 = BoardGameEngine(num_players=6, seed=1)
    v2 = e2.get_observation_vector(0)
    v6 = e6.get_observation_vector(0)
    assert v2.shape == v6.shape
    assert v2.dtype.name == "float32"


def test_observation_is_json_serializable() -> None:
    import json

    engine = BoardGameEngine(num_players=4, seed=3)
    json.dumps(engine.get_observation())
    json.dumps(engine.get_observation(player_id=1))
