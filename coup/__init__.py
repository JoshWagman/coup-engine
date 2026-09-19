"""Headless, RL-ready game engine for base Coup.

Public API:
    ``BoardGameEngine`` - the main engine class.
    ``PlayerState`` / ``PendingAction`` - state dataclasses.
    ``constants`` - enums and the fixed action-ID layout.
"""

from __future__ import annotations

from . import constants
from .constants import ActionType, Character, Phase
from .engine import BoardGameEngine, IllegalActionError, PendingAction, PlayerState

__all__ = [
    "BoardGameEngine",
    "PlayerState",
    "PendingAction",
    "IllegalActionError",
    "Character",
    "ActionType",
    "Phase",
    "constants",
]
