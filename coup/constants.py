"""Static definitions for the Coup engine.

This module holds every value that does not depend on live game state:

* the :class:`Character`, :class:`ActionType` and :class:`Phase` enums,
* the flat integer action-ID layout used for reinforcement-learning action
  masking, and
* small pure helper functions for encoding / decoding those action IDs.

Keeping these out of :mod:`coup.engine` makes the action space easy to reason
about in isolation (e.g. when building a PettingZoo / Gymnasium wrapper) and
guarantees the layout is identical everywhere it is referenced.

Action-ID layout (``MAX_PLAYERS == 6`` -> total space of ``ACTION_SPACE_SIZE``)::

    0            Income
    1            Foreign Aid
    2            Tax                (claims Duke)
    3            Exchange           (claims Ambassador)
    4  .. 9      Coup(target=0..5)
    10 .. 15     Assassinate(target=0..5)   (claims Assassin)
    16 .. 21     Steal(target=0..5)         (claims Captain)
    22           Pass / Decline     (decline to challenge or block)
    23           Challenge
    24           Block with Duke        (vs Foreign Aid)
    25           Block with Contessa    (vs Assassinate)
    26           Block with Captain     (vs Steal)
    27           Block with Ambassador  (vs Steal)
    28           Lose influence: reveal card in slot 0
    29           Lose influence: reveal card in slot 1
    30 .. 35     Exchange keep-combination index (see EXCHANGE_COMBOS)

The layout is *fixed* and independent of the current phase. ``get_legal_moves``
returns the subset that is legal right now, which is exactly the mask an RL
agent needs.
"""

from __future__ import annotations

from enum import IntEnum
from itertools import combinations

MAX_PLAYERS: int = 6
"""Largest player count the fixed action layout supports."""


class Character(IntEnum):
    """The five influence characters in base Coup (3 copies of each)."""

    DUKE = 0
    ASSASSIN = 1
    CAPTAIN = 2
    AMBASSADOR = 3
    CONTESSA = 4


COPIES_PER_CHARACTER: int = 3
"""Number of copies of each character in the deck."""


class ActionType(IntEnum):
    """Logical action / reaction kinds, independent of concrete action IDs."""

    INCOME = 0
    FOREIGN_AID = 1
    TAX = 2
    EXCHANGE = 3
    COUP = 4
    ASSASSINATE = 5
    STEAL = 6
    PASS = 7
    CHALLENGE = 8
    BLOCK = 9
    LOSE_INFLUENCE = 10
    EXCHANGE_KEEP = 11


class Phase(IntEnum):
    """Decision points of the turn state machine.

    Each phase has exactly one *current decision-maker* (see
    :attr:`coup.engine.BoardGameEngine.current_player`). ``apply_action`` moves
    the machine forward one decision at a time, which maps cleanly onto a
    PettingZoo ``agent_selection`` / ``step`` loop.
    """

    AWAIT_ACTION = 0
    AWAIT_CHALLENGE_ACTION = 1
    AWAIT_BLOCK = 2
    AWAIT_CHALLENGE_BLOCK = 3
    AWAIT_LOSE_INFLUENCE = 4
    AWAIT_EXCHANGE = 5
    GAME_OVER = 6


# --- Flat action-ID layout ---------------------------------------------------

INCOME: int = 0
FOREIGN_AID: int = 1
TAX: int = 2
EXCHANGE: int = 3

COUP_BASE: int = 4
ASSASSINATE_BASE: int = 10
STEAL_BASE: int = 16

PASS: int = 22
CHALLENGE: int = 23

BLOCK_DUKE: int = 24
BLOCK_CONTESSA: int = 25
BLOCK_CAPTAIN: int = 26
BLOCK_AMBASSADOR: int = 27

LOSE_SLOT_BASE: int = 28  # slots 28, 29 -> influence position 0, 1

EXCHANGE_KEEP_BASE: int = 30
# Temporary Exchange hand is at most (2 influence + 2 drawn) = 4 cards, from
# which the player keeps up to 2. The largest number of keep-combinations is
# C(4, 2) = 6, so six IDs are reserved.
MAX_EXCHANGE_COMBOS: int = 6

ACTION_SPACE_SIZE: int = EXCHANGE_KEEP_BASE + MAX_EXCHANGE_COMBOS  # 36

# Blocking-action ID -> the character being claimed by that block.
BLOCK_ACTION_TO_CHARACTER: dict[int, Character] = {
    BLOCK_DUKE: Character.DUKE,
    BLOCK_CONTESSA: Character.CONTESSA,
    BLOCK_CAPTAIN: Character.CAPTAIN,
    BLOCK_AMBASSADOR: Character.AMBASSADOR,
}

# The character a player must claim to legitimately perform each action.
ACTION_REQUIRED_CHARACTER: dict[ActionType, Character] = {
    ActionType.TAX: Character.DUKE,
    ActionType.ASSASSINATE: Character.ASSASSIN,
    ActionType.STEAL: Character.CAPTAIN,
    ActionType.EXCHANGE: Character.AMBASSADOR,
}

# Which characters may legally block each action type.
ACTION_BLOCKERS: dict[ActionType, tuple[Character, ...]] = {
    ActionType.FOREIGN_AID: (Character.DUKE,),
    ActionType.ASSASSINATE: (Character.CONTESSA,),
    ActionType.STEAL: (Character.CAPTAIN, Character.AMBASSADOR),
}

COUP_COST: int = 7
ASSASSINATE_COST: int = 3
FORCE_COUP_THRESHOLD: int = 10
STARTING_COINS: int = 2
STARTING_INFLUENCE: int = 2
STEAL_AMOUNT: int = 2


def encode_targeted(base: int, target: int) -> int:
    """Return the action ID for a targeted action against ``target``."""

    return base + target


def decode_target(action: int, base: int) -> int:
    """Return the target player index encoded in ``action`` relative to ``base``."""

    return action - base


def is_in_range(action: int, base: int, count: int) -> bool:
    """Return ``True`` if ``action`` falls in ``[base, base + count)``."""

    return base <= action < base + count


def exchange_combos(hand_size: int, keep: int) -> list[tuple[int, ...]]:
    """Enumerate keep-combinations for an Exchange as index tuples.

    The temporary Exchange hand (existing influence + 2 freshly drawn cards) is
    sorted deterministically by the caller; this returns the combinations of
    *positions* to keep, ordered lexicographically so combo index ``k`` always
    maps to the same selection given the same hand. ``keep`` equals the player's
    current living-influence count.
    """

    return list(combinations(range(hand_size), keep))


def action_name(action: int) -> str:
    """Human-readable label for an action ID (used by the demo / logging)."""

    fixed = {
        INCOME: "Income",
        FOREIGN_AID: "ForeignAid",
        TAX: "Tax(Duke)",
        EXCHANGE: "Exchange(Ambassador)",
        PASS: "Pass",
        CHALLENGE: "Challenge",
        BLOCK_DUKE: "Block(Duke)",
        BLOCK_CONTESSA: "Block(Contessa)",
        BLOCK_CAPTAIN: "Block(Captain)",
        BLOCK_AMBASSADOR: "Block(Ambassador)",
    }
    if action in fixed:
        return fixed[action]
    if is_in_range(action, COUP_BASE, MAX_PLAYERS):
        return f"Coup(->P{decode_target(action, COUP_BASE)})"
    if is_in_range(action, ASSASSINATE_BASE, MAX_PLAYERS):
        return f"Assassinate(->P{decode_target(action, ASSASSINATE_BASE)})"
    if is_in_range(action, STEAL_BASE, MAX_PLAYERS):
        return f"Steal(->P{decode_target(action, STEAL_BASE)})"
    if is_in_range(action, LOSE_SLOT_BASE, 2):
        return f"LoseInfluence(slot{decode_target(action, LOSE_SLOT_BASE)})"
    if is_in_range(action, EXCHANGE_KEEP_BASE, MAX_EXCHANGE_COMBOS):
        return f"ExchangeKeep(combo{decode_target(action, EXCHANGE_KEEP_BASE)})"
    return f"Action({action})"
