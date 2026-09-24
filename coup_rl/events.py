"""Structured game-event schema and builders.

Events are plain JSON-serialisable dicts. Claim/block truth (``had_character``,
``copies_held``) must be derived from a **pre-step** hand snapshot: a proven
claim is reshuffled immediately, so a post-step hand read is wrong.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from coup import constants as C
from coup.engine import BoardGameEngine, PendingAction


def snapshot_engine(engine: BoardGameEngine) -> dict[str, Any]:
    """God-view snapshot of live hands and public state, taken *before* a step."""

    pending = None
    if engine.pending is not None:
        pa: PendingAction = engine.pending
        pending = {
            "action_type": pa.action_type.name,
            "actor": pa.actor,
            "target": pa.target,
            "claimed_character": None if pa.claimed_character is None else pa.claimed_character.name,
            "blocker": pa.blocker,
            "block_character": None if pa.block_character is None else pa.block_character.name,
        }
    return {
        "phase": engine.phase.name,
        "turn_count": engine.turn_count,
        "current_player": engine.current_player,
        "cards": {p.player_id: [c.name for c in p.cards] for p in engine.players},
        "revealed": {p.player_id: [c.name for c in p.revealed] for p in engine.players},
        "alive": {p.player_id: p.alive for p in engine.players},
        "coins": {p.player_id: p.coins for p in engine.players},
        "pending": pending,
    }


def copies_held(cards: Iterable[str], character: str) -> int:
    """Count how many copies of ``character`` are in a name list."""

    return sum(1 for c in cards if c == character)


def claim_from_action(action: int, phase: str) -> Optional[tuple[str, str]]:
    """Return ``(character_name, claim_kind)`` if ``action`` is a character claim.

    ``claim_kind`` is ``"action"`` (Tax / Assassinate / Steal / Exchange) or
    ``"block"`` (block with Duke / Contessa / Captain / Ambassador).
    """

    if phase == "AWAIT_ACTION":
        if action == C.TAX:
            return (C.Character.DUKE.name, "action")
        if action == C.EXCHANGE:
            return (C.Character.AMBASSADOR.name, "action")
        if C.is_in_range(action, C.ASSASSINATE_BASE, C.MAX_PLAYERS):
            return (C.Character.ASSASSIN.name, "action")
        if C.is_in_range(action, C.STEAL_BASE, C.MAX_PLAYERS):
            return (C.Character.CAPTAIN.name, "action")
        return None
    if phase == "AWAIT_BLOCK" and action in C.BLOCK_ACTION_TO_CHARACTER:
        return (C.BLOCK_ACTION_TO_CHARACTER[action].name, "block")
    return None


def _base_event(
    *,
    event_type: str,
    episode_id: str,
    step: int,
    pre: dict[str, Any],
    actor: int,
    action: int,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "episode_id": episode_id,
        "step": step,
        "turn_count": pre["turn_count"],
        "phase": pre["phase"],
        "actor": actor,
        "action_id": action,
        "action_name": C.action_name(action),
    }


def events_from_step(
    pre: dict[str, Any],
    actor: int,
    action: int,
    engine: BoardGameEngine,
    info: dict[str, Any],
    episode_id: str,
    step: int,
) -> list[dict[str, Any]]:
    """Build structured events for one ``apply_action`` using the pre-step snapshot."""

    events: list[dict[str, Any]] = []
    events.append(_base_event(event_type="decision", episode_id=episode_id, step=step, pre=pre, actor=actor, action=action))

    claimed = claim_from_action(action, pre["phase"])
    if claimed is not None:
        character, kind = claimed
        held = copies_held(pre["cards"].get(actor, []), character)
        event_type = "block_claim" if kind == "block" else "claim"
        ev = _base_event(event_type=event_type, episode_id=episode_id, step=step, pre=pre, actor=actor, action=action)
        ev.update(
            {
                "character": character,
                "had_character": held > 0,
                "copies_held": held,
                "claim_kind": kind,
            }
        )
        events.append(ev)

    if action == C.CHALLENGE:
        pending = pre.get("pending") or {}
        if pre["phase"] == "AWAIT_CHALLENGE_ACTION":
            target = pending.get("actor")
            character = pending.get("claimed_character")
        elif pre["phase"] == "AWAIT_CHALLENGE_BLOCK":
            target = pending.get("blocker")
            character = pending.get("block_character")
        else:
            target = None
            character = None
        if target is not None and character is not None:
            held = copies_held(pre["cards"].get(target, []), character)
            was_bluff = held == 0
            ch = _base_event(event_type="challenge", episode_id=episode_id, step=step, pre=pre, actor=actor, action=action)
            ch.update({"target": target, "character": character})
            events.append(ch)
            resolved = dict(ch)
            resolved["event_type"] = "challenge_resolved"
            resolved["was_bluff"] = was_bluff
            resolved["had_character"] = not was_bluff
            resolved["copies_held"] = held
            events.append(resolved)

    for pid, pre_rev in pre["revealed"].items():
        post_rev = [c.name for c in engine.players[pid].revealed]
        if len(post_rev) > len(pre_rev):
            for card in post_rev[len(pre_rev) :]:
                ev = _base_event(
                    event_type="lose_influence",
                    episode_id=episode_id,
                    step=step,
                    pre=pre,
                    actor=pid,
                    action=action,
                )
                ev["card"] = card
                events.append(ev)
        if pre["alive"].get(pid) and not engine.players[pid].alive:
            ev = _base_event(
                event_type="elimination",
                episode_id=episode_id,
                step=step,
                pre=pre,
                actor=pid,
                action=action,
            )
            events.append(ev)

    if info.get("done"):
        ev = _base_event(event_type="game_end", episode_id=episode_id, step=step, pre=pre, actor=actor, action=action)
        ev["winner"] = info.get("winner")
        events.append(ev)

    return events
