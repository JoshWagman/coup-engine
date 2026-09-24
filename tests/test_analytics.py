"""Analytics over synthetic JSONL: Duke claimed vs actually held."""

from __future__ import annotations

import json

import pytest

from coup_rl.analytics import (
    bluff_vs_win_table,
    challenge_table,
    claim_truth_table,
    filter_actor,
    format_claim_table,
    load_events,
    main,
    phase_bluff_table,
    plot_bluff_vs_win,
    pooled_win_by_bluff,
    starting_hand_table,
    top_winner,
    win_counts,
)


def _write_run(tmp_path, events):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "events.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return run


def test_claim_truth_table_duke(tmp_path) -> None:
    events = [
        {"event_type": "claim", "character": "DUKE", "had_character": True, "actor": 0, "claim_kind": "action"},
        {"event_type": "claim", "character": "DUKE", "had_character": False, "actor": 0, "claim_kind": "action"},
        {"event_type": "block_claim", "character": "DUKE", "had_character": True, "actor": 1, "claim_kind": "block"},
        {"event_type": "claim", "character": "CAPTAIN", "had_character": False, "actor": 1, "claim_kind": "action"},
        {"event_type": "decision", "actor": 0},
    ]
    run = _write_run(tmp_path, events)
    loaded = load_events(run)
    table = claim_truth_table(loaded, "DUKE")
    assert table["overall"]["claims"] == 3
    assert table["overall"]["had_card"] == 2
    assert table["overall"]["truth_rate"] == pytest.approx(2 / 3)
    assert table["overall"]["bluff_rate"] == pytest.approx(1 / 3)
    assert table["by_claim_kind"]["action"]["claims"] == 2
    assert table["by_claim_kind"]["block"]["truth_rate"] == 1.0
    assert table["by_player"]["0"]["claims"] == 2
    text = format_claim_table(table)
    assert "DUKE" in text
    assert "claims=3" in text


def test_analytics_cli(tmp_path, capsys) -> None:
    events = [
        {"event_type": "claim", "character": "DUKE", "had_character": True, "actor": 0, "claim_kind": "action"},
        {"event_type": "claim", "character": "DUKE", "had_character": False, "actor": 1, "claim_kind": "action"},
    ]
    run = _write_run(tmp_path, events)
    assert main([str(run), "--character", "DUKE"]) == 0
    out = capsys.readouterr().out
    assert "DUKE" in out
    assert "claims=2" in out
    assert "50.0%" in out


def _write_run_with_episodes(tmp_path, events, episodes):
    run = _write_run(tmp_path, events)
    with (run / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes:
            fh.write(json.dumps(row) + "\n")
    return run


def test_top_winner_and_player_filter(tmp_path, capsys) -> None:
    events = [
        {"event_type": "claim", "character": "DUKE", "had_character": False, "actor": 0, "claim_kind": "action"},
        {"event_type": "claim", "character": "ASSASSIN", "had_character": True, "actor": 1, "claim_kind": "action"},
        {"event_type": "claim", "character": "CONTESSA", "had_character": False, "actor": 1, "claim_kind": "block"},
    ]
    episodes = [{"winner": 1}, {"winner": 1}, {"winner": 0}]
    run = _write_run_with_episodes(tmp_path, events, episodes)
    assert win_counts(episodes) == {0: 1, 1: 2}
    assert top_winner(episodes) == 1
    assert all(e["actor"] == 1 for e in filter_actor(events, 1))

    assert main([str(run), "--top"]) == 0
    out = capsys.readouterr().out
    assert "player_1  wins=2" in out
    assert "<-- top" in out
    assert "player_1 only" in out
    assert "ASSASSIN" in out
    assert "CONTESSA" in out
    duke_section = out.split("Claim truth: DUKE", 1)[1].split("Claim truth:", 1)[0]
    assert "claims=0" in duke_section


def test_phase_bluff_table_early_mid_late() -> None:
    events = [
        {"event_type": "episode_start", "num_players": 2, "episode_id": "1"},
        {"event_type": "decision", "phase": "AWAIT_ACTION", "action_id": 2, "actor": 0},
        {
            "event_type": "claim",
            "character": "DUKE",
            "had_character": False,
            "actor": 0,
            "claim_kind": "action",
        },
        {"event_type": "decision", "phase": "AWAIT_ACTION", "action_id": 5, "actor": 0},  # Coup
        {
            "event_type": "claim",
            "character": "CAPTAIN",
            "had_character": False,
            "actor": 1,
            "claim_kind": "action",
        },
        {"event_type": "lose_influence", "actor": 0, "step": 10},
        {"event_type": "lose_influence", "actor": 1, "step": 11},
        {
            "event_type": "claim",
            "character": "ASSASSIN",
            "had_character": False,
            "actor": 0,
            "claim_kind": "action",
        },
    ]
    table = phase_bluff_table(events)
    assert table["early"]["bluffs"] == 1
    assert table["early"]["by_character"][0]["character"] == "DUKE"
    assert table["mid"]["by_character"][0]["character"] == "CAPTAIN"
    assert table["late"]["by_character"][0]["character"] == "ASSASSIN"


def test_starting_hand_table_replays_seed() -> None:
    from coup import BoardGameEngine

    engine = BoardGameEngine(num_players=2, seed=0)
    expected = " + ".join(sorted(c.name for c in engine.players[0].cards))
    table = starting_hand_table(
        [{"winner": 0, "seed": 0, "num_players": 2, "game_over": True}]
    )
    assert table["games"] == 1
    assert table["hands"][0]["hand"] == expected
    assert table["hands"][0]["wins"] == 1


def test_phases_and_hands_cli(tmp_path, capsys) -> None:
    events = [
        {"event_type": "episode_start", "num_players": 2, "episode_id": "1"},
        {"event_type": "decision", "phase": "AWAIT_ACTION", "action_id": 2, "actor": 0},
        {
            "event_type": "claim",
            "character": "DUKE",
            "had_character": False,
            "actor": 0,
            "claim_kind": "action",
        },
    ]
    episodes = [{"winner": 0, "seed": 0, "num_players": 2, "game_over": True}]
    run = _write_run_with_episodes(tmp_path, events, episodes)
    assert main([str(run), "--phases"]) == 0
    phase_out = capsys.readouterr().out
    assert "False claims by game phase" in phase_out
    assert "DUKE" in phase_out
    assert main([str(run), "--hands"]) == 0
    hand_out = capsys.readouterr().out
    assert "Starting hands vs wins" in hand_out


def test_bluff_vs_win_table_and_cli(tmp_path, capsys) -> None:
    events = [
        {"event_type": "claim", "episode_id": "1", "actor": 0, "had_character": False, "character": "DUKE"},
        {"event_type": "claim", "episode_id": "1", "actor": 0, "had_character": False, "character": "DUKE"},
        {"event_type": "claim", "episode_id": "2", "actor": 0, "had_character": True, "character": "DUKE"},
        {"event_type": "claim", "episode_id": "2", "actor": 1, "had_character": True, "character": "ASSASSIN"},
    ]
    episodes = [
        {"episode_id": "1", "winner": 0, "num_players": 2, "game_over": True},
        {"episode_id": "2", "winner": 1, "num_players": 2, "game_over": True},
    ]
    table = bluff_vs_win_table(events, episodes)
    p0 = table["players"][0]
    p1 = table["players"][1]
    assert p0["wins"] == 1 and p0["games"] == 2
    assert p0["bluffs"] == 2 and p0["claims"] == 3
    assert p0["bluff_rate"] == pytest.approx(2 / 3)
    assert p1["win_rate"] == pytest.approx(0.5)
    high = next(b for b in p0["by_bluff_bin"] if b["bluff_bin"] == "67-100%")
    assert high["games"] == 1 and high["wins"] == 1

    run = _write_run_with_episodes(tmp_path, events, episodes)
    assert main([str(run), "--bluff-wins"]) == 0
    out = capsys.readouterr().out
    assert "Win rate vs bluff rate" in out
    assert "player_0" in out


def test_pooled_curve_and_plot(tmp_path) -> None:
    events = [
        {"event_type": "claim", "episode_id": "1", "actor": 0, "had_character": False},
        {"event_type": "claim", "episode_id": "1", "actor": 1, "had_character": True},
        {"event_type": "claim", "episode_id": "2", "actor": 0, "had_character": True},
        {"event_type": "claim", "episode_id": "2", "actor": 1, "had_character": True},
    ]
    episodes = [
        {"episode_id": "1", "winner": 0, "num_players": 2},
        {"episode_id": "2", "winner": 1, "num_players": 2},
    ]
    curve = pooled_win_by_bluff(events, episodes)
    zero = next(b for b in curve["bins"] if b["bluff_bin"] == "0%")
    assert zero["games"] == 3
    full = next(b for b in curve["bins"] if b["bluff_bin"] == "90-100%")
    assert full["games"] == 1 and full["wins"] == 1
    dest = tmp_path / "win_vs_bluff.png"
    plot_bluff_vs_win(curve, dest)
    assert dest.exists() and dest.stat().st_size > 0

    run = _write_run_with_episodes(tmp_path, events, episodes)
    cli_png = tmp_path / "cli.png"
    assert main([str(run), "--plot", str(cli_png)]) == 0
    assert cli_png.exists()


def test_challenge_table_certain_vs_uncertain(tmp_path, capsys) -> None:
    events = [
        {"event_type": "episode_start", "num_players": 2, "episode_id": "1"},
        {
            "event_type": "decision",
            "phase": "AWAIT_CHALLENGE_ACTION",
            "action_id": 23,
            "actor": 1,
        },
        {
            "event_type": "challenge_resolved",
            "actor": 1,
            "character": "DUKE",
            "was_bluff": True,
            "target": 0,
        },
        {"event_type": "lose_influence", "actor": 0, "card": "DUKE"},
        {"event_type": "lose_influence", "actor": 0, "card": "DUKE"},
        {"event_type": "lose_influence", "actor": 1, "card": "DUKE"},
        {
            "event_type": "decision",
            "phase": "AWAIT_CHALLENGE_ACTION",
            "action_id": 22,
            "actor": 0,
        },
        {
            "event_type": "decision",
            "phase": "AWAIT_CHALLENGE_ACTION",
            "action_id": 23,
            "actor": 1,
        },
        {
            "event_type": "challenge_resolved",
            "actor": 1,
            "character": "DUKE",
            "was_bluff": True,
            "target": 0,
        },
    ]
    table = challenge_table(events)
    assert table["opportunities"] == 3
    assert table["challenges"] == 2
    assert table["correct"] == 2
    assert table["uncertain"] == 1
    assert table["certain"] == 1

    run = _write_run(tmp_path, events)
    assert main([str(run), "--challenges"]) == 0
    out = capsys.readouterr().out
    assert "Challenge rate" in out
    assert "uncertain" in out
