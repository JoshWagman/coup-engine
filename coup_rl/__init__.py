"""PettingZoo RL layer for the headless Coup engine.

The engine stays reward-free and AI-free. This package wraps it with:

* :class:`CoupAECEnv` - sequential PettingZoo AEC environment
* structured run recording (claim-vs-held truth, challenges, eliminations)
* :func:`claim_truth_table` analytics over a run directory
* a shared-policy MaskablePPO self-play trainer (``python -m coup_rl.train``)
"""

from __future__ import annotations

from .analytics import (
    bluff_vs_win_table,
    claim_truth_table,
    load_events,
    load_run,
    phase_bluff_table,
    starting_hand_table,
)
from .env import CoupAECEnv
from .recorder import EventRecorder, RunWriter

__all__ = [
    "CoupAECEnv",
    "EventRecorder",
    "RunWriter",
    "bluff_vs_win_table",
    "claim_truth_table",
    "load_events",
    "load_run",
    "phase_bluff_table",
    "starting_hand_table",
]
