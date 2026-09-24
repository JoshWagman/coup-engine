"""Recorder must read hidden hands *before* apply_action (challenge reshuffles)."""

from __future__ import annotations

from coup import BoardGameEngine, Character
from coup import constants as C
from coup_rl.events import events_from_step, snapshot_engine
from coup_rl.recorder import EventRecorder, RunWriter


def test_tax_claim_uses_pre_step_hand() -> None:
    engine = BoardGameEngine(num_players=3, seed=0)
    engine.players[0].cards = [Character.DUKE, Character.ASSASSIN]
    pre = snapshot_engine(engine)
    info = engine.apply_action(0, C.TAX)
    events = events_from_step(pre, 0, C.TAX, engine, info, "ep", 0)
    claims = [e for e in events if e["event_type"] == "claim"]
    assert len(claims) == 1
    assert claims[0]["character"] == "DUKE"
    assert claims[0]["had_character"] is True
    assert claims[0]["copies_held"] == 1
    assert claims[0]["claim_kind"] == "action"


def test_challenge_truth_matches_pre_reshuffle_hand() -> None:
    """A proven Duke is swapped out of the hand; truth still comes from the snapshot."""

    engine = BoardGameEngine(num_players=3, seed=0)
    engine.players[0].cards = [Character.DUKE, Character.ASSASSIN]
    engine.deck = [Character.CONTESSA, Character.CAPTAIN, Character.AMBASSADOR]
    engine.rng.shuffle = lambda seq: seq.sort(key=lambda c: 0 if c != Character.CONTESSA else 1)

    pre_tax = snapshot_engine(engine)
    info_tax = engine.apply_action(0, C.TAX)
    tax_events = events_from_step(pre_tax, 0, C.TAX, engine, info_tax, "ep", 0)
    assert any(e["event_type"] == "claim" and e["had_character"] for e in tax_events)

    assert engine.phase.name == "AWAIT_CHALLENGE_ACTION"
    challenger = engine.current_player
    assert challenger is not None and challenger != 0

    pre_ch = snapshot_engine(engine)
    assert "DUKE" in pre_ch["cards"][0]
    info_ch = engine.apply_action(challenger, C.CHALLENGE)
    ch_events = events_from_step(pre_ch, challenger, C.CHALLENGE, engine, info_ch, "ep", 1)

    resolved = [e for e in ch_events if e["event_type"] == "challenge_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["character"] == "DUKE"
    assert resolved[0]["was_bluff"] is False
    assert resolved[0]["had_character"] is True
    assert resolved[0]["copies_held"] == 1
    # Post-step the proven Duke has been reshuffled out (deck was Duke-free).
    assert Character.DUKE not in engine.players[0].cards


def test_bluff_tax_recorded(tmp_path) -> None:
    engine = BoardGameEngine(num_players=3, seed=0)
    engine.players[0].cards = [Character.CONTESSA, Character.ASSASSIN]
    writer = RunWriter(tmp_path, {"num_players": 3}, run_id="bluff")
    rec = EventRecorder(writer)
    rec.begin_episode(num_players=3, seed=0, episode_id="1")
    pre = rec.snapshot(engine)
    info = engine.apply_action(0, C.TAX)
    events = rec.record_step(engine, 0, C.TAX, info, pre=pre)
    writer.close()
    claim = next(e for e in events if e["event_type"] == "claim")
    assert claim["had_character"] is False
    assert claim["copies_held"] == 0
    on_disk = (tmp_path / "bluff" / "events.jsonl").read_text(encoding="utf-8")
    assert "claim" in on_disk
    assert '"had_character": false' in on_disk


def test_block_claim_kind() -> None:
    engine = BoardGameEngine(num_players=2, seed=1)
    engine.players[0].cards = [Character.ASSASSIN, Character.CONTESSA]
    engine.players[1].cards = [Character.DUKE, Character.CAPTAIN]
    engine.apply_action(0, C.FOREIGN_AID)
    # P1 may block with Duke.
    assert engine.phase.name == "AWAIT_BLOCK"
    assert engine.current_player == 1
    pre = snapshot_engine(engine)
    info = engine.apply_action(1, C.BLOCK_DUKE)
    events = events_from_step(pre, 1, C.BLOCK_DUKE, engine, info, "ep", 0)
    blocks = [e for e in events if e["event_type"] == "block_claim"]
    assert len(blocks) == 1
    assert blocks[0]["character"] == "DUKE"
    assert blocks[0]["claim_kind"] == "block"
    assert blocks[0]["had_character"] is True
