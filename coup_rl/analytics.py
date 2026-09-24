"""Query helpers over recorded Coup runs (claim-vs-held tables, etc.)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from collections import Counter, defaultdict

from coup import BoardGameEngine
from coup import constants as C
from coup.constants import Character

CLAIM_EVENTS = frozenset({"claim", "block_claim"})
PHASES = ("early", "mid", "late")


def load_events(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load ``events.jsonl`` from a run directory."""

    path = Path(run_dir) / "events.jsonl"
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_episodes(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load ``episodes.jsonl`` from a run directory."""

    path = Path(run_dir) / "episodes.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(run_dir: Path | str) -> dict[str, Any]:
    """Load config, events, and episode summaries for a run directory."""

    root = Path(run_dir)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {
        "dir": str(root),
        "config": config,
        "events": load_events(root),
        "episodes": load_episodes(root),
    }


def win_counts(episodes: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Return games won per seat index from ``episodes.jsonl`` rows."""

    counts: dict[int, int] = {}
    for row in episodes:
        winner = row.get("winner")
        if winner is None:
            continue
        seat = int(winner)
        counts[seat] = counts.get(seat, 0) + 1
    return dict(sorted(counts.items()))


def top_winner(episodes: Sequence[dict[str, Any]]) -> Optional[int]:
    """Seat with the most wins. Ties go to the lowest seat index."""

    counts = win_counts(episodes)
    if not counts:
        return None
    best = max(counts.values())
    return min(seat for seat, n in counts.items() if n == best)


def filter_actor(events: Iterable[dict[str, Any]], actor: int) -> list[dict[str, Any]]:
    """Keep events whose ``actor`` is ``actor`` (claims, blocks, decisions by that seat)."""

    return [e for e in events if e.get("actor") == actor]


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    had = sum(1 for r in rows if r.get("had_character"))
    return {
        "claims": n,
        "had_card": had,
        "truth_rate": (had / n) if n else None,
        "bluff_rate": ((n - had) / n) if n else None,
    }


def _group(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key))
        buckets.setdefault(label, []).append(row)
    return {label: _summarize(items) for label, items in sorted(buckets.items())}


def claim_truth_table(
    events: Iterable[dict[str, Any]],
    character: Optional[str] = None,
) -> dict[str, Any]:
    """Compare how often ``character`` was claimed vs actually held.

    ``character`` is a name like ``"DUKE"``. ``None`` includes every character.
    Rows are ``claim`` and ``block_claim`` events only.
    """

    rows = [e for e in events if e.get("event_type") in CLAIM_EVENTS]
    if character is not None:
        character = character.upper()
        rows = [e for e in rows if e.get("character") == character]
    return {
        "character": character,
        "overall": _summarize(rows),
        "by_player": _group(rows, "actor"),
        "by_claim_kind": _group(rows, "claim_kind"),
        "by_character": _group(rows, "character"),
    }


def _fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def format_win_table(counts: dict[int, int], highlight: Optional[int] = None) -> str:
    """Pretty-print games won per seat."""

    total = sum(counts.values()) or 1
    lines = ["Wins by seat"]
    for seat, n in counts.items():
        mark = "  <-- top" if highlight is not None and seat == highlight else ""
        lines.append(f"  player_{seat}  wins={n}  win_rate={100.0 * n / total:.1f}%{mark}")
    return "\n".join(lines)


class _PhaseState:
    """Live influence plus whether a Coup/Assassinate action has been taken."""

    def __init__(self, num_players: int) -> None:
        self.n = num_players
        self.cards = {i: 2 for i in range(num_players)}
        self.alive = {i: True for i in range(num_players)}
        self.lethal = False

    def living(self) -> list[int]:
        return [i for i in range(self.n) if self.alive[i] and self.cards[i] > 0]

    def phase(self) -> Optional[str]:
        living = self.living()
        if len(living) <= 1:
            return None
        if len(living) in (2, 3) and all(self.cards[i] == 1 for i in living):
            return "late"
        if not self.lethal:
            return "early"
        return "mid"


def _apply_phase_event(state: _PhaseState, event: dict[str, Any]) -> Optional[str]:
    """Update ``state``; return the phase to tag a claim event with, else None."""

    et = event.get("event_type")
    if et == "decision" and event.get("phase") == "AWAIT_ACTION":
        aid = event.get("action_id")
        if isinstance(aid, int) and (
            C.is_in_range(aid, C.COUP_BASE, C.MAX_PLAYERS)
            or C.is_in_range(aid, C.ASSASSINATE_BASE, C.MAX_PLAYERS)
        ):
            state.lethal = True
    tagged = None
    if et in CLAIM_EVENTS:
        tagged = state.phase()
    if et == "lose_influence":
        pid = int(event["actor"])
        if state.cards.get(pid, 0) > 0:
            state.cards[pid] -= 1
        if state.cards.get(pid, 0) == 0:
            state.alive[pid] = False
    if et == "elimination":
        pid = int(event["actor"])
        state.alive[pid] = False
        state.cards[pid] = 0
    return tagged


def phase_bluff_table(
    events: Iterable[dict[str, Any]],
    actor: Optional[int] = None,
) -> dict[str, Any]:
    """False-claim counts by early / mid / late, using the phase rules below.

    * early — no Coup or Assassinate action taken yet, and not already late
    * late — 2 or 3 players left, each with exactly 1 influence
    * mid — Coup/Assassinate has happened, but the position is not late
    """

    claims: dict[str, Counter[str]] = {p: Counter() for p in PHASES}
    bluffs: dict[str, Counter[str]] = {p: Counter() for p in PHASES}
    bluffs_kind: dict[str, dict[str, Counter[str]]] = {
        p: defaultdict(Counter) for p in PHASES
    }
    state: Optional[_PhaseState] = None
    for event in events:
        if event.get("event_type") == "episode_start":
            state = _PhaseState(int(event.get("num_players") or 4))
            continue
        if state is None:
            continue
        tagged = _apply_phase_event(state, event)
        if tagged is None or event.get("event_type") not in CLAIM_EVENTS:
            continue
        if actor is not None and event.get("actor") != actor:
            continue
        character = str(event.get("character"))
        claims[tagged][character] += 1
        if not event.get("had_character"):
            bluffs[tagged][character] += 1
            bluffs_kind[tagged][str(event.get("claim_kind"))][character] += 1
    out: dict[str, Any] = {}
    for phase in PHASES:
        total_c = sum(claims[phase].values())
        total_b = sum(bluffs[phase].values())
        by_char = []
        for character, n_bluff in bluffs[phase].most_common():
            n_claim = claims[phase][character]
            by_char.append(
                {
                    "character": character,
                    "bluffs": n_bluff,
                    "claims": n_claim,
                    "share_of_bluffs": (n_bluff / total_b) if total_b else None,
                    "bluff_rate": (n_bluff / n_claim) if n_claim else None,
                }
            )
        out[phase] = {
            "claims": total_c,
            "bluffs": total_b,
            "bluff_rate": (total_b / total_c) if total_c else None,
            "by_character": by_char,
            "by_kind": {
                kind: dict(counter.most_common())
                for kind, counter in bluffs_kind[phase].items()
            },
        }
    return out


def format_phase_table(table: dict[str, Any]) -> str:
    """Pretty-print :func:`phase_bluff_table`."""

    lines = [
        "False claims by game phase",
        "  early = no Coup/Assassinate taken yet (and not already late)",
        "  late  = 2 or 3 players left, each with 1 influence",
        "  mid   = after first Coup/Assassinate, until late",
        "",
    ]
    for phase in PHASES:
        block = table[phase]
        lines.append(
            f"=== {phase.upper()}  claims={block['claims']}  bluffs={block['bluffs']}  "
            f"bluff_rate={_fmt_rate(block['bluff_rate'])} ==="
        )
        if not block["bluffs"]:
            lines.append("  (no false claims)")
            lines.append("")
            continue
        lines.append("  Most falsely claimed (share of this phase's bluffs):")
        for row in block["by_character"]:
            share = _fmt_rate(row["share_of_bluffs"])
            rate = _fmt_rate(row["bluff_rate"])
            lines.append(
                f"    {row['character']:11}  bluffs={row['bluffs']:<5}  ({share:>6} of bluffs)  "
                f"claims={row['claims']:<5}  bluff_rate={rate}"
            )
        if block["by_kind"]:
            lines.append("  Bluffs by claim kind:")
            for kind, chars in block["by_kind"].items():
                lines.append(f"    {kind}:")
                for character, n in chars.items():
                    lines.append(f"      {character:11} {n}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _hand_key(cards: Sequence[Any]) -> tuple[str, ...]:
    return tuple(sorted(c.name for c in cards))


def starting_hand_table(episodes: Sequence[dict[str, Any]], winner: Optional[int] = None) -> dict[str, Any]:
    """Replay each episode seed and count opening hands vs wins.

    ``win_rate`` is wins / times that pair was dealt (to ``winner``'s seat if
    set, otherwise to any seat). Opening hands are not stored in events.jsonl.
    """

    wins: Counter[tuple[str, ...]] = Counter()
    dealt: Counter[tuple[str, ...]] = Counter()
    games = 0
    for row in episodes:
        if not row.get("game_over", True) or row.get("seed") is None:
            continue
        n = int(row.get("num_players") or 4)
        engine = BoardGameEngine(num_players=n, seed=int(row["seed"]))
        w = row.get("winner")
        w_id = int(w) if w is not None else None
        if winner is not None:
            if winner >= n:
                continue
            dealt[_hand_key(engine.players[winner].cards)] += 1
            if w_id == winner:
                wins[_hand_key(engine.players[w_id].cards)] += 1
                games += 1
            continue
        for player in engine.players:
            dealt[_hand_key(player.cards)] += 1
        if w_id is not None and 0 <= w_id < n:
            wins[_hand_key(engine.players[w_id].cards)] += 1
            games += 1
    rows = []
    for key, n_wins in wins.most_common():
        n_dealt = dealt[key]
        rows.append(
            {
                "hand": " + ".join(key),
                "wins": n_wins,
                "dealt": n_dealt,
                "win_rate": (n_wins / n_dealt) if n_dealt else None,
            }
        )
    return {"games": games, "winner_filter": winner, "hands": rows}


def format_hand_table(table: dict[str, Any]) -> str:
    """Pretty-print :func:`starting_hand_table`."""

    who = (
        f"player_{table['winner_filter']} wins only"
        if table.get("winner_filter") is not None
        else "all seats"
    )
    lines = [
        f"Starting hands vs wins ({who}, games={table['games']})",
        "  win_rate = wins / times this pair was dealt "
        + ("to that seat" if table.get("winner_filter") is not None else "to anyone"),
        "  (4-player random baseline is 25%)",
        "",
        f"  {'hand':28}  {'wins':>5}  {'dealt':>5}  win_rate",
    ]
    for row in table["hands"]:
        lines.append(
            f"  {row['hand']:28}  {row['wins']:5}  {row['dealt']:5}  {_fmt_rate(row['win_rate'])}"
        )
    return "\n".join(lines)


def format_claim_table(table: dict[str, Any]) -> str:
    """Pretty-print a :func:`claim_truth_table` result."""

    lines: list[str] = []
    title = table.get("character") or "ALL"
    overall = table["overall"]
    lines.append(f"Claim truth: {title}")
    lines.append(
        f"  overall  claims={overall['claims']}  had={overall['had_card']}  "
        f"truth={_fmt_rate(overall['truth_rate'])}  bluff={_fmt_rate(overall['bluff_rate'])}"
    )
    for label, summary in table["by_claim_kind"].items():
        lines.append(
            f"  kind={label:6} claims={summary['claims']}  "
            f"truth={_fmt_rate(summary['truth_rate'])}  bluff={_fmt_rate(summary['bluff_rate'])}"
        )
    for label, summary in table["by_player"].items():
        lines.append(
            f"  player={label}  claims={summary['claims']}  "
            f"truth={_fmt_rate(summary['truth_rate'])}  bluff={_fmt_rate(summary['bluff_rate'])}"
        )
    if table.get("character") is None:
        for label, summary in table["by_character"].items():
            lines.append(
                f"  char={label:11} claims={summary['claims']}  "
                f"truth={_fmt_rate(summary['truth_rate'])}  bluff={_fmt_rate(summary['bluff_rate'])}"
            )
    return "\n".join(lines)


BLUFF_WIN_BINS: tuple[tuple[str, Optional[float], Optional[float]], ...] = (
    ("0%", 0.0, 0.0),
    ("1-33%", 0.0, 1.0 / 3.0),
    ("34-66%", 1.0 / 3.0, 2.0 / 3.0),
    ("67-100%", 2.0 / 3.0, 1.0),
)


def _bin_bluff_rate(rate: float) -> str:
    if rate == 0.0:
        return "0%"
    if rate <= 1.0 / 3.0:
        return "1-33%"
    if rate <= 2.0 / 3.0:
        return "34-66%"
    return "67-100%"


def bluff_vs_win_table(
    events: Iterable[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    actor: Optional[int] = None,
) -> dict[str, Any]:
    """Win rate vs bluff rate for each seat, plus win rate by per-game bluff bin.

    A game's bluff rate for a seat is that seat's false claims / claims in that
    episode. Seats with no claims in a game go in ``no_claims``.
    """

    winners: dict[str, int] = {}
    seats_in_game: dict[str, int] = {}
    for row in episodes:
        eid = str(row.get("episode_id", ""))
        if not eid or row.get("winner") is None:
            continue
        winners[eid] = int(row["winner"])
        seats_in_game[eid] = int(row.get("num_players") or 0)

    per_game: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for event in events:
        if event.get("event_type") not in CLAIM_EVENTS:
            continue
        if event.get("actor") is None or event.get("episode_id") is None:
            continue
        pid = int(event["actor"])
        if actor is not None and pid != actor:
            continue
        per_game[(str(event["episode_id"]), pid)].append(bool(event.get("had_character")))

    players: set[int] = set()
    if actor is not None:
        players.add(actor)
    else:
        players.update(winners.values())
        for n in seats_in_game.values():
            players.update(range(n))
        players.update(pid for _, pid in per_game)

    rows = []
    for pid in sorted(players):
        games = 0
        wins = 0
        claims = 0
        bluffs = 0
        bins: dict[str, list[int]] = {label: [0, 0] for label, *_ in BLUFF_WIN_BINS}
        bins["no_claims"] = [0, 0]
        for eid, winner in winners.items():
            n = seats_in_game.get(eid, 0)
            if n and pid >= n:
                continue
            games += 1
            won = winner == pid
            if won:
                wins += 1
            held_flags = per_game.get((eid, pid))
            if not held_flags:
                bins["no_claims"][0] += 1
                bins["no_claims"][1] += int(won)
                continue
            n_claim = len(held_flags)
            n_bluff = sum(1 for held in held_flags if not held)
            claims += n_claim
            bluffs += n_bluff
            label = _bin_bluff_rate(n_bluff / n_claim)
            bins[label][0] += 1
            bins[label][1] += int(won)
        by_bin = []
        for label, *_ in (*BLUFF_WIN_BINS, ("no_claims", None, None)):
            n_games, n_wins = bins[label]
            by_bin.append(
                {
                    "bluff_bin": label,
                    "games": n_games,
                    "wins": n_wins,
                    "win_rate": (n_wins / n_games) if n_games else None,
                }
            )
        rows.append(
            {
                "player": pid,
                "games": games,
                "wins": wins,
                "win_rate": (wins / games) if games else None,
                "claims": claims,
                "bluffs": bluffs,
                "bluff_rate": (bluffs / claims) if claims else None,
                "by_bluff_bin": by_bin,
            }
        )
    return {"players": rows}


def format_bluff_vs_win_table(table: dict[str, Any]) -> str:
    """Pretty-print :func:`bluff_vs_win_table`."""

    lines = [
        "Win rate vs bluff rate by seat",
        "  bluff_rate = false claims / claims over the whole run",
        "  bins = that seat's bluff rate inside a single game, then win% in those games",
        "",
    ]
    for row in table["players"]:
        lines.append(
            f"player_{row['player']}  wins={row['wins']}/{row['games']}  "
            f"win_rate={_fmt_rate(row['win_rate'])}  "
            f"claims={row['claims']}  bluffs={row['bluffs']}  "
            f"bluff_rate={_fmt_rate(row['bluff_rate'])}"
        )
        for bucket in row["by_bluff_bin"]:
            if bucket["games"] == 0:
                continue
            lines.append(
                f"  bluff {bucket['bluff_bin']:9}  games={bucket['games']:<5}  "
                f"wins={bucket['wins']:<5}  win_rate={_fmt_rate(bucket['win_rate'])}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _seat_game_bluff_outcomes(
    events: Iterable[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    actor: Optional[int] = None,
) -> list[tuple[float, bool]]:
    """One ``(bluff_rate, won)`` per seat per game that had at least one claim."""

    winners: dict[str, int] = {}
    seats_in_game: dict[str, int] = {}
    for row in episodes:
        eid = str(row.get("episode_id", ""))
        if not eid or row.get("winner") is None:
            continue
        winners[eid] = int(row["winner"])
        seats_in_game[eid] = int(row.get("num_players") or 0)

    per_game: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for event in events:
        if event.get("event_type") not in CLAIM_EVENTS:
            continue
        if event.get("actor") is None or event.get("episode_id") is None:
            continue
        pid = int(event["actor"])
        if actor is not None and pid != actor:
            continue
        per_game[(str(event["episode_id"]), pid)].append(bool(event.get("had_character")))

    points: list[tuple[float, bool]] = []
    for (eid, pid), held_flags in per_game.items():
        if eid not in winners:
            continue
        n = seats_in_game.get(eid, 0)
        if n and pid >= n:
            continue
        n_claim = len(held_flags)
        if n_claim == 0:
            continue
        n_bluff = sum(1 for held in held_flags if not held)
        points.append((n_bluff / n_claim, winners[eid] == pid))
    return points


def pooled_win_by_bluff(
    events: Iterable[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    *,
    bin_width: float = 0.1,
    actor: Optional[int] = None,
) -> dict[str, Any]:
    """Pool every seat and bin win rate by that seat-game's bluff rate.

    ``0%`` is its own bin; later bins are ``(lo, hi]`` of width ``bin_width``.
    """

    points = _seat_game_bluff_outcomes(events, episodes, actor)
    n_bins = int(round(1.0 / bin_width))
    labels = ["0%"] + [f"{int(i * bin_width * 100)}-{int((i + 1) * bin_width * 100)}%" for i in range(n_bins)]
    counts = [[0, 0] for _ in labels]  # games, wins

    def index_for(rate: float) -> int:
        if rate == 0.0:
            return 0
        idx = min(n_bins, int((rate - 1e-12) / bin_width) + 1)
        return idx

    for rate, won in points:
        idx = index_for(rate)
        counts[idx][0] += 1
        counts[idx][1] += int(won)

    rows = []
    for label, (games, wins) in zip(labels, counts):
        rows.append(
            {
                "bluff_bin": label,
                "games": games,
                "wins": wins,
                "win_rate": (wins / games) if games else None,
            }
        )
    return {
        "games_with_claims": len(points),
        "bin_width": bin_width,
        "bins": rows,
    }


def plot_bluff_vs_win(
    curve: dict[str, Any],
    dest: Path | str,
    *,
    title: str = "Win rate vs bluff rate (all seats)",
) -> Path:
    """Write a PNG of pooled win rate (y) against bluff-rate bins (x)."""

    import matplotlib.pyplot as plt

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = [row for row in curve["bins"] if row["games"]]
    xs = list(range(len(plotted)))
    ys = [100.0 * float(row["win_rate"]) for row in plotted]
    labels = [row["bluff_bin"] for row in plotted]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", color="#2563eb", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Bluff rate in that game (false claims / claims)")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, max(50.0, max(ys) + 5 if ys else 50.0))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest_path, dpi=140)
    plt.close(fig)
    return dest_path


CHALLENGE_PHASES = frozenset({"AWAIT_CHALLENGE_ACTION", "AWAIT_CHALLENGE_BLOCK"})


def _visible_copies(character: str, revealed: Sequence[str], hand: Optional[Sequence[str]]) -> int:
    seen = sum(1 for c in revealed if c == character)
    if hand is not None:
        seen += sum(1 for c in hand if c == character)
    return seen


def challenge_table(
    events: Iterable[dict[str, Any]],
    actor: Optional[int] = None,
) -> dict[str, Any]:
    """Challenge frequency, accuracy, and certainty from the challenger's view.

    A challenge is **certain** if the challenger can already account for all 3
    copies of the claimed character from public revealed cards plus their own
    known hidden cards. After an Exchange (or a proven-claim reshuffle) that
    player's hidden hand is treated as unknown, so only the 3 revealed copies
    count as certain.
    """

    opportunities = 0
    challenges = 0
    correct = 0
    wrong = 0
    certain = 0
    certain_correct = 0
    uncertain = 0
    uncertain_correct = 0
    hand_known = 0
    by_character: dict[str, dict[str, int]] = defaultdict(
        lambda: {"challenges": 0, "correct": 0, "certain": 0, "uncertain": 0}
    )

    hands: dict[int, Optional[list[str]]] = {}
    revealed: list[str] = []

    for event in events:
        et = event.get("event_type")
        if et == "episode_start":
            n = int(event.get("num_players") or 4)
            seed = event.get("seed")
            revealed = []
            hands = {}
            if seed is not None:
                engine = BoardGameEngine(num_players=n, seed=int(seed))
                hands = {p.player_id: [c.name for c in p.cards] for p in engine.players}
            continue

        pid = event.get("actor")
        if et == "decision" and event.get("phase") in CHALLENGE_PHASES:
            if actor is None or pid == actor:
                opportunities += 1
                if event.get("action_id") == C.CHALLENGE:
                    challenges += 1

        if et == "challenge_resolved":
            if actor is None or pid == actor:
                character = str(event.get("character"))
                was_bluff = bool(event.get("was_bluff"))
                hand = hands.get(int(pid)) if pid is not None else None
                if hand is not None:
                    hand_known += 1
                visible = _visible_copies(character, revealed, hand)
                is_certain = visible >= C.COPIES_PER_CHARACTER
                if was_bluff:
                    correct += 1
                else:
                    wrong += 1
                row = by_character[character]
                row["challenges"] += 1
                row["correct"] += int(was_bluff)
                if is_certain:
                    certain += 1
                    certain_correct += int(was_bluff)
                    row["certain"] += 1
                else:
                    uncertain += 1
                    uncertain_correct += int(was_bluff)
                    row["uncertain"] += 1
            # Proven claim: target's card is reshuffled; their new card is hidden.
            if event.get("was_bluff") is False and event.get("target") is not None:
                hands[int(event["target"])] = None

        if et == "lose_influence" and event.get("card"):
            loser = int(event["actor"])
            card = str(event["card"])
            revealed.append(card)
            owned = hands.get(loser)
            if owned is not None and card in owned:
                owned.remove(card)

        if et == "decision" and isinstance(event.get("action_id"), int):
            aid = int(event["action_id"])
            if C.is_in_range(aid, C.EXCHANGE_KEEP_BASE, C.MAX_EXCHANGE_COMBOS) and pid is not None:
                hands[int(pid)] = None

    resolved = correct + wrong
    return {
        "opportunities": opportunities,
        "challenges": challenges,
        "passes": opportunities - challenges,
        "challenge_rate": (challenges / opportunities) if opportunities else None,
        "resolved": resolved,
        "correct": correct,
        "wrong": wrong,
        "correct_rate": (correct / resolved) if resolved else None,
        "certain": certain,
        "certain_correct": certain_correct,
        "certain_correct_rate": (certain_correct / certain) if certain else None,
        "uncertain": uncertain,
        "uncertain_correct": uncertain_correct,
        "uncertain_correct_rate": (uncertain_correct / uncertain) if uncertain else None,
        "uncertain_share": (uncertain / resolved) if resolved else None,
        "hand_known": hand_known,
        "by_character": dict(sorted(by_character.items())),
    }


def format_challenge_table(table: dict[str, Any]) -> str:
    """Pretty-print :func:`challenge_table`."""

    lines = [
        "Challenge rate and accuracy",
        "  opportunity = Pass or Challenge during a challenge window",
        "  correct = the claim was a bluff (challenger was right)",
        "  certain = challenger could already see all 3 copies (own known hand + revealed)",
        "  uncertain = fewer than 3 copies visible, so not 100% sure it was a bluff",
        "",
        f"  opportunities={table['opportunities']}  challenges={table['challenges']}  "
        f"passes={table['passes']}  challenge_rate={_fmt_rate(table['challenge_rate'])}",
        f"  resolved={table['resolved']}  correct={table['correct']}  wrong={table['wrong']}  "
        f"correct_rate={_fmt_rate(table['correct_rate'])}",
        f"  certain={table['certain']}  certain_correct={table['certain_correct']}  "
        f"certain_correct_rate={_fmt_rate(table['certain_correct_rate'])}",
        f"  uncertain={table['uncertain']}  ({_fmt_rate(table['uncertain_share'])} of challenges)  "
        f"uncertain_correct={table['uncertain_correct']}  "
        f"uncertain_correct_rate={_fmt_rate(table['uncertain_correct_rate'])}",
        f"  challenger hand still known at challenge time: {table['hand_known']}/{table['resolved']}",
        "",
        "  By character:",
    ]
    for character, row in table["by_character"].items():
        n = row["challenges"]
        corr = (row["correct"] / n) if n else None
        lines.append(
            f"    {character:11}  challenges={n:<5}  correct={row['correct']:<5}  "
            f"correct_rate={_fmt_rate(corr)}  certain={row['certain']}  uncertain={row['uncertain']}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a Coup RL run directory (claims, phases, starting hands)."
    )
    parser.add_argument("run_dir", type=Path, help="Path to a data/runs folder")
    parser.add_argument(
        "--claims",
        action="store_true",
        help="Print claim-vs-held tables (default if --phases/--hands are omitted).",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Print false-claim counts by early / mid / late game.",
    )
    parser.add_argument(
        "--hands",
        action="store_true",
        help="Replay episode seeds and print starting-hand vs win counts.",
    )
    parser.add_argument(
        "--bluff-wins",
        action="store_true",
        help="Print each seat's win rate vs bluff rate (overall and by per-game bluff bin).",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="AUTO",
        default=None,
        metavar="PNG",
        help=(
            "Write a pooled (all seats) win-rate vs bluff-rate PNG. "
            "Optional path; default is RUN/win_vs_bluff.png. Implies --bluff-wins."
        ),
    )
    parser.add_argument(
        "--challenges",
        action="store_true",
        help="Print challenge rate, accuracy, and how often challenges were made without seeing all 3 copies.",
    )
    parser.add_argument(
        "--character",
        default=None,
        help="With --claims: only this character (DUKE, ASSASSIN, ...).",
    )
    player_group = parser.add_mutually_exclusive_group()
    player_group.add_argument(
        "--player",
        type=int,
        default=None,
        help="Restrict to one seat (0-based player_N).",
    )
    player_group.add_argument(
        "--top",
        action="store_true",
        help="Restrict to the seat with the most wins (lowest index on a tie).",
    )
    args = parser.parse_args(argv)
    want_phases = args.phases
    want_hands = args.hands
    want_bluff_wins = args.bluff_wins or args.plot is not None
    want_challenges = args.challenges
    want_claims = args.claims or not (
        want_phases or want_hands or want_bluff_wins or want_challenges
    )

    events = load_events(args.run_dir)
    episodes = load_episodes(args.run_dir)
    counts = win_counts(episodes)
    seat: Optional[int] = args.player
    if args.top:
        seat = top_winner(episodes)
        if seat is None:
            print("No finished episodes with a winner; cannot use --top.", file=sys.stderr)
            return 1
    if counts:
        print(format_win_table(counts, highlight=seat))
        print()

    printed = False
    if want_phases:
        if seat is not None:
            print(f"Phase bluffs for player_{seat} only")
            print()
        print(format_phase_table(phase_bluff_table(events, actor=seat)))
        printed = True
    if want_hands:
        if printed:
            print()
        print(format_hand_table(starting_hand_table(episodes, winner=seat)))
        printed = True
    if want_challenges:
        if printed:
            print()
        if seat is not None:
            print(f"Challenges for player_{seat} only")
            print()
        print(format_challenge_table(challenge_table(events, actor=seat)))
        printed = True
    if want_bluff_wins:
        if printed:
            print()
        print(format_bluff_vs_win_table(bluff_vs_win_table(events, episodes, actor=seat)))
        printed = True
        if args.plot is not None:
            dest = args.run_dir / "win_vs_bluff.png" if args.plot == "AUTO" else Path(args.plot)
            curve = pooled_win_by_bluff(events, episodes, actor=seat)
            title = (
                f"Win rate vs bluff rate (player_{seat})"
                if seat is not None
                else "Win rate vs bluff rate (all seats)"
            )
            out = plot_bluff_vs_win(curve, dest, title=title)
            print()
            print(f"Wrote {out}")
    if want_claims:
        if printed:
            print()
        claim_events = filter_actor(events, seat) if seat is not None else list(events)
        if seat is not None:
            print(f"Claim/bluff stats for player_{seat} only")
            print()
        if args.character:
            print(format_claim_table(claim_truth_table(claim_events, args.character)))
        else:
            print(format_claim_table(claim_truth_table(claim_events)))
            print()
            for char in Character:
                print(format_claim_table(claim_truth_table(claim_events, char.name)))
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
