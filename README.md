# Coup Headless Game Engine

A headless, deterministic, reinforcement-learning-ready game engine for the base
[Coup](https://boardgamegeek.com/boardgame/131357/coup) card game, written in
pure Python.

The engine owns **game logic only** - there is no UI, network, or AI code. It is
designed so it can later be wrapped in a
[PettingZoo](https://pettingzoo.farama.org/) /
[Gymnasium](https://gymnasium.farama.org/) environment for RL training.

## Highlights

- **Strict separation of concerns** - only rules and state live here.
- **Seedable RNG** - pass `seed=...` for fully reproducible games (deck shuffles
  and card draws are deterministic).
- **Fixed discrete action space** with **action masking** via
  `get_legal_moves`, so RL agents never attempt illegal moves.
- **Sequential decision points** - every reactive choice in Coup (challenges,
  blocks, which influence to lose, which cards to keep in an Exchange) is its own
  agent action, mapping cleanly onto `step(action)`.
- **Structured + numeric observations** - a JSON-serialisable dict (full or
  per-agent partial view) and a fixed-length NumPy vector for policy networks.

## Rules implemented (base Coup)

- 2-6 players. Deck of 15 cards: 5 characters (Duke, Assassin, Captain,
  Ambassador, Contessa) x 3 copies. Each player starts with 2 face-down
  influence and 2 coins. Last player with influence wins (no draws).
- **General actions:** Income (+1), Foreign Aid (+2, blockable by Duke),
  Coup (pay 7, target loses an influence; unblockable and unchallengeable;
  forced when you start your turn with 10+ coins).
- **Character actions:** Tax / Duke (+3), Assassinate / Assassin (pay 3, target
  loses an influence, blockable by Contessa), Steal / Captain (take up to 2 from
  a target, blockable by Captain or Ambassador), Exchange / Ambassador (draw 2,
  keep your influence count, return the rest).
- **Challenges:** any character claim - by an actor or a blocker - can be
  challenged. The loser of a challenge loses an influence; a proven claim is
  reshuffled into the deck and replaced. Assassinate coins are spent up front and
  are **not** refunded, so the famous double-kill (a target challenging a real
  Assassin and then being assassinated) is possible.

## Installation

```bash
pip install -r requirements.txt
```

Only NumPy is required at runtime; `pytest` is for the test suite.

## Quick start

```python
from coup import BoardGameEngine

engine = BoardGameEngine(num_players=4, seed=42)
obs = engine.reset()

while not engine.is_game_over()[0]:
    player = engine.current_player            # whose decision we await
    legal = engine.get_legal_moves(player)    # action mask
    action = legal[0]                         # your policy picks here
    info = engine.apply_action(player, action)

over, winner = engine.is_game_over()
print("winner:", winner)
```

## Run the demo

A random-legal-move mock game loop (two players, then a four-player game):

```bash
python -m coup.engine
```

## Run the tests

```bash
pytest
```

## API overview

`BoardGameEngine` exposes exactly the surface an RL wrapper needs:

| Method | Purpose | Gym/PettingZoo analog |
| --- | --- | --- |
| `__init__(num_players=4, *, seed=None)` | Configure the game and RNG. | env construction |
| `reset()` | Deal a fresh game; return the initial observation. | `reset()` |
| `current_player` (property) | Player id whose decision is awaited (`None` if over). | `agent_selection` |
| `get_legal_moves(player_id)` | Sorted list of legal action IDs (the action mask). | action mask |
| `apply_action(player_id, action)` | Validate, mutate, advance the state machine; returns an info dict. | `step(action)` |
| `get_observation(player_id=None)` | JSON-serialisable dict; full god-view or per-agent partial view. | `observe(agent)` |
| `get_observation_vector(player_id)` | Fixed-length `float32` NumPy encoding (incl. action mask). | network input |
| `is_game_over()` | `(is_over, winner_id)`. | termination |

The engine is intentionally **reward-free**; a wrapper derives rewards from
eliminations / the winner reported by `is_game_over()`.

## Action-ID layout

Actions are integers in `[0, 36)`. The layout is fixed and independent of the
current phase; `get_legal_moves` returns the subset legal right now.

| IDs | Meaning |
| --- | --- |
| `0` | Income |
| `1` | Foreign Aid |
| `2` | Tax (claims Duke) |
| `3` | Exchange (claims Ambassador) |
| `4-9` | Coup (target = id - 4) |
| `10-15` | Assassinate (target = id - 10) |
| `16-21` | Steal (target = id - 16) |
| `22` | Pass / Decline |
| `23` | Challenge |
| `24` | Block with Duke (vs Foreign Aid) |
| `25` | Block with Contessa (vs Assassinate) |
| `26` | Block with Captain (vs Steal) |
| `27` | Block with Ambassador (vs Steal) |
| `28-29` | Lose influence: reveal card in slot 0 / 1 |
| `30-35` | Exchange keep-combination index |

Player-target ranges are padded to 6 (`MAX_PLAYERS`) so the space is identical
regardless of the actual player count.

## Turn state machine

Each phase has exactly one current decision-maker (`current_player`).
`apply_action` advances the machine one decision at a time.

- `AWAIT_ACTION` - the acting player picks a main action (forced Coup at 10+
  coins).
- `AWAIT_CHALLENGE_ACTION` - opponents, in turn order, each Challenge or Pass a
  character claim.
- `AWAIT_BLOCK` - the eligible blocker(s) declare a block or Pass (Foreign Aid:
  any opponent; Steal / Assassinate: the target).
- `AWAIT_CHALLENGE_BLOCK` - non-blockers Challenge or Pass the block claim.
- `AWAIT_LOSE_INFLUENCE` - a player with a genuine choice picks which card slot
  to reveal (a sole remaining card is auto-revealed).
- `AWAIT_EXCHANGE` - the acting player chooses which cards to keep.
- `GAME_OVER`.

Reaction windows are linearised in turn order: the first player to
challenge/block resolves the window; if everyone passes, the window closes. A
pending-losses queue handles multi-loss sequences (e.g. the double-kill).

## Project layout

```
coup/
  __init__.py     # public exports
  constants.py    # enums + fixed action-ID layout + helpers
  engine.py       # BoardGameEngine, state dataclasses, and the demo loop
tests/
  test_engine.py  # masking invariants, rules edge cases, seeded playouts
requirements.txt
README.md
```
