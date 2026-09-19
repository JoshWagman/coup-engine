"""Headless game engine for base Coup.

The engine is intentionally decoupled from any UI, network or AI code. It only
owns game logic and exposes a small, RL-friendly surface:

* a **fixed discrete action space** (see :mod:`coup.constants`),
* **action masking** via :meth:`BoardGameEngine.get_legal_moves`, and
* a **sequential decision-point** state machine so that every reactive choice
  in Coup (challenges, blocks, choosing which influence to lose, choosing which
  cards to keep during an Exchange) is its own agent action.

State layout
------------
The authoritative state consists of:

* ``players`` - a list of :class:`PlayerState`. Each holds ``coins``, the list
  of face-down ``cards`` (living influence; index = slot) and the list of
  face-up ``revealed`` cards (public, dead influence).
* ``deck`` - the face-down court deck (:class:`~coup.constants.Character`).
* ``phase`` - the current :class:`~coup.constants.Phase`.
* ``current_turn_player`` - whose turn it is (the acting player).
* ``current_player`` - whose *decision* the engine is currently waiting on
  (equal to the acting player during ``AWAIT_ACTION``, otherwise a reacting
  player). This is the id that must be passed to ``apply_action`` next.
* ``pending`` - a :class:`PendingAction` describing the action being resolved.

Action-ID format
----------------
Actions are plain integers in ``[0, ACTION_SPACE_SIZE)``; the full mapping is
documented in :mod:`coup.constants`. ``get_legal_moves`` returns exactly the
integers that are legal for the current decision-maker in the current phase,
which doubles as the RL action mask.

Mapping to Gymnasium / PettingZoo
---------------------------------
* :meth:`reset` -> ``env.reset()``
* :meth:`current_player` -> ``env.agent_selection``
* :meth:`get_legal_moves` -> action mask
* :meth:`apply_action` -> ``env.step(action)``
* :meth:`get_observation` -> ``env.observe(agent)``

The engine is deliberately reward-free (separation of concerns); a thin wrapper
can derive rewards from eliminations / the winner reported by
:meth:`is_game_over`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import constants as C
from .constants import ActionType, Character, Phase


@dataclass
class PlayerState:
    """Mutable per-player state.

    Attributes:
        player_id: Stable index of the player.
        coins: Number of coins currently held.
        cards: Living (face-down) influence characters; list index is the slot
            referenced by the ``Lose influence slot N`` actions.
        revealed: Dead (face-up) influence characters; public information.
    """

    player_id: int
    coins: int = 0
    cards: list[Character] = field(default_factory=list)
    revealed: list[Character] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        """Whether the player still has at least one living influence."""

        return len(self.cards) > 0

    @property
    def influence_count(self) -> int:
        """Number of living influence cards."""

        return len(self.cards)


@dataclass
class PendingAction:
    """Describes the action currently being resolved by the state machine.

    Attributes:
        action_type: The main action kind being attempted.
        actor: Player id who initiated the action.
        target: Target player id (``None`` for untargeted actions).
        claimed_character: Character the actor claims for the main action
            (``None`` for actions that make no claim, e.g. Foreign Aid).
        blocker: Player id who declared a block (``None`` if unblocked so far).
        block_character: Character claimed by the blocker (``None`` if no block).
    """

    action_type: ActionType
    actor: int
    target: Optional[int] = None
    claimed_character: Optional[Character] = None
    blocker: Optional[int] = None
    block_character: Optional[Character] = None


class IllegalActionError(ValueError):
    """Raised when ``apply_action`` receives an action that is not legal."""


class BoardGameEngine:
    """Headless engine for base Coup (2-6 players).

    See the module docstring for the state layout and the action-ID format.
    """

    def __init__(self, num_players: int = 4, *, seed: Optional[int] = None, **kwargs: Any) -> None:
        """Initialise engine configuration.

        Args:
            num_players: Number of players (2-6 inclusive).
            seed: Optional RNG seed for fully reproducible games (deck shuffles,
                card draws). ``None`` seeds from system entropy.
            **kwargs: Accepted and ignored for forward compatibility.

        Raises:
            ValueError: If ``num_players`` is outside ``[2, 6]``.
        """

        if not 2 <= num_players <= C.MAX_PLAYERS:
            raise ValueError(f"num_players must be in [2, {C.MAX_PLAYERS}], got {num_players}")

        self.num_players: int = num_players
        self._seed: Optional[int] = seed
        self.rng: random.Random = random.Random(seed)

        # State populated by reset().
        self.players: list[PlayerState] = []
        self.deck: list[Character] = []
        self.phase: Phase = Phase.AWAIT_ACTION
        self.current_turn_player: int = 0
        self._decision_player: int = 0
        self.pending: Optional[PendingAction] = None
        self.turn_count: int = 0
        self.winner: Optional[int] = None

        # Internal machinery for the sequential decision-point state machine.
        self._reaction_queue: list[int] = []
        self._loss_queue: list[int] = []
        self._continuation: Optional[str] = None
        self._exchange_hand: Optional[list[Character]] = None
        self._exchange_combos: Optional[list[tuple[int, ...]]] = None
        self.last_events: list[str] = []

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Reset to a fresh starting position and return the initial observation.

        Builds a shuffled 15-card deck (5 characters x 3 copies), deals two
        influence cards and two coins to each player, and sets the phase to
        ``AWAIT_ACTION`` with player 0 to act.
        """

        self.rng = random.Random(self._seed)
        self.deck = [Character(c) for c in range(len(Character)) for _ in range(C.COPIES_PER_CHARACTER)]
        self.rng.shuffle(self.deck)

        self.players = [PlayerState(player_id=i, coins=C.STARTING_COINS) for i in range(self.num_players)]
        for _ in range(C.STARTING_INFLUENCE):
            for p in self.players:
                p.cards.append(self.deck.pop())

        self.phase = Phase.AWAIT_ACTION
        self.current_turn_player = 0
        self._decision_player = 0
        self.pending = None
        self.turn_count = 0
        self.winner = None
        self._reaction_queue = []
        self._loss_queue = []
        self._continuation = None
        self._exchange_hand = None
        self._exchange_combos = None
        self.last_events = ["game reset"]

        return self.get_observation()

    @property
    def current_player(self) -> Optional[int]:
        """Player id whose decision the engine is awaiting (``None`` if over)."""

        if self.phase == Phase.GAME_OVER:
            return None
        return self._decision_player

    def get_legal_moves(self, player_id: int) -> list[int]:
        """Return the sorted list of legal action IDs for ``player_id``.

        Returns an empty list if the game is over or if it is not
        ``player_id``'s turn to decide. The returned list is exactly the action
        mask an RL agent should apply.
        """

        if self.phase == Phase.GAME_OVER or player_id != self._decision_player:
            return []

        phase = self.phase
        if phase == Phase.AWAIT_ACTION:
            return self._legal_main_actions(player_id)
        if phase in (Phase.AWAIT_CHALLENGE_ACTION, Phase.AWAIT_CHALLENGE_BLOCK):
            return [C.PASS, C.CHALLENGE]
        if phase == Phase.AWAIT_BLOCK:
            return self._legal_block_actions()
        if phase == Phase.AWAIT_LOSE_INFLUENCE:
            return [C.LOSE_SLOT_BASE + i for i in range(len(self.players[player_id].cards))]
        if phase == Phase.AWAIT_EXCHANGE:
            assert self._exchange_combos is not None
            return [C.EXCHANGE_KEEP_BASE + i for i in range(len(self._exchange_combos))]
        return []

    def apply_action(self, player_id: int, action: int) -> dict[str, Any]:
        """Validate and apply ``action`` for ``player_id``; advance the machine.

        The action is validated against :meth:`get_legal_moves`, the state is
        mutated, phase transitions / win checks run, and an info dict describing
        the outcome is returned (analogous to the ``info`` from a Gym ``step``).

        Raises:
            IllegalActionError: If the game is over, it is not the player's
                decision, or the action is not currently legal.
        """

        if self.phase == Phase.GAME_OVER:
            raise IllegalActionError("game is over")
        if player_id != self._decision_player:
            raise IllegalActionError(
                f"it is player {self._decision_player}'s decision, not player {player_id}"
            )
        if action not in self.get_legal_moves(player_id):
            raise IllegalActionError(f"action {action} ({C.action_name(action)}) is not legal now")

        self.last_events = []
        phase = self.phase
        if phase == Phase.AWAIT_ACTION:
            self._apply_main_action(player_id, action)
        elif phase == Phase.AWAIT_CHALLENGE_ACTION:
            self._apply_challenge_action(player_id, action)
        elif phase == Phase.AWAIT_BLOCK:
            self._apply_block(player_id, action)
        elif phase == Phase.AWAIT_CHALLENGE_BLOCK:
            self._apply_challenge_block(player_id, action)
        elif phase == Phase.AWAIT_LOSE_INFLUENCE:
            self._apply_lose_influence(player_id, action)
        elif phase == Phase.AWAIT_EXCHANGE:
            self._apply_exchange(player_id, action)

        over, winner = self.is_game_over()
        return {
            "events": list(self.last_events),
            "phase": self.phase.name,
            "current_player": self.current_player,
            "done": over,
            "winner": winner,
        }

    def is_game_over(self) -> tuple[bool, Optional[int]]:
        """Return ``(is_over, winner_id)``. Coup has no draws, so on game over
        exactly one player id is the winner; otherwise ``(False, None)``."""

        return (self.phase == Phase.GAME_OVER, self.winner)

    # ------------------------------------------------------------------
    # Legal-move helpers
    # ------------------------------------------------------------------
    def _alive_opponents(self, player_id: int) -> list[int]:
        return [i for i in range(self.num_players) if i != player_id and self.players[i].alive]

    def _legal_main_actions(self, player_id: int) -> list[int]:
        player = self.players[player_id]
        coins = player.coins
        targets = self._alive_opponents(player_id)

        # 10+ coins forces a Coup.
        if coins >= C.FORCE_COUP_THRESHOLD:
            return sorted(C.COUP_BASE + t for t in targets)

        moves: list[int] = [C.INCOME, C.FOREIGN_AID, C.TAX, C.EXCHANGE]
        if coins >= C.COUP_COST:
            moves += [C.COUP_BASE + t for t in targets]
        if coins >= C.ASSASSINATE_COST:
            moves += [C.ASSASSINATE_BASE + t for t in targets]
        moves += [C.STEAL_BASE + t for t in targets]
        return sorted(moves)

    def _legal_block_actions(self) -> list[int]:
        assert self.pending is not None
        blocks = {
            ActionType.FOREIGN_AID: [C.BLOCK_DUKE],
            ActionType.ASSASSINATE: [C.BLOCK_CONTESSA],
            ActionType.STEAL: [C.BLOCK_CAPTAIN, C.BLOCK_AMBASSADOR],
        }[self.pending.action_type]
        return [C.PASS, *blocks]

    # ------------------------------------------------------------------
    # Turn / ordering helpers
    # ------------------------------------------------------------------
    def _alive_in_order(self, start: int, exclude: set[int] = frozenset()) -> list[int]:  # type: ignore[assignment]
        order: list[int] = []
        for i in range(self.num_players):
            pid = (start + i) % self.num_players
            if pid in exclude or not self.players[pid].alive:
                continue
            order.append(pid)
        return order

    def _next_alive(self, start: int) -> int:
        for i in range(1, self.num_players + 1):
            pid = (start + i) % self.num_players
            if self.players[pid].alive:
                return pid
        return start

    # ------------------------------------------------------------------
    # AWAIT_ACTION
    # ------------------------------------------------------------------
    def _apply_main_action(self, player_id: int, action: int) -> None:
        player = self.players[player_id]

        if action == C.INCOME:
            player.coins += 1
            self._log(f"P{player_id} takes Income (+1 -> {player.coins})")
            self._end_turn()
            return

        if action == C.FOREIGN_AID:
            self.pending = PendingAction(ActionType.FOREIGN_AID, actor=player_id)
            self._log(f"P{player_id} attempts Foreign Aid")
            self._open_block_window()
            return

        if action == C.TAX:
            self.pending = PendingAction(ActionType.TAX, actor=player_id, claimed_character=Character.DUKE)
            self._log(f"P{player_id} claims Duke for Tax")
            self._open_action_challenge()
            return

        if action == C.EXCHANGE:
            self.pending = PendingAction(
                ActionType.EXCHANGE, actor=player_id, claimed_character=Character.AMBASSADOR
            )
            self._log(f"P{player_id} claims Ambassador to Exchange")
            self._open_action_challenge()
            return

        if C.is_in_range(action, C.COUP_BASE, C.MAX_PLAYERS):
            target = C.decode_target(action, C.COUP_BASE)
            player.coins -= C.COUP_COST
            self.pending = PendingAction(ActionType.COUP, actor=player_id, target=target)
            self._log(f"P{player_id} launches a Coup on P{target} (-7 -> {player.coins})")
            if self.players[target].alive:
                self._queue_loss(target)
            self._begin_losses("end_turn")
            return

        if C.is_in_range(action, C.ASSASSINATE_BASE, C.MAX_PLAYERS):
            target = C.decode_target(action, C.ASSASSINATE_BASE)
            player.coins -= C.ASSASSINATE_COST
            self.pending = PendingAction(
                ActionType.ASSASSINATE, actor=player_id, target=target, claimed_character=Character.ASSASSIN
            )
            self._log(f"P{player_id} claims Assassin to assassinate P{target} (-3 -> {player.coins})")
            self._open_action_challenge()
            return

        if C.is_in_range(action, C.STEAL_BASE, C.MAX_PLAYERS):
            target = C.decode_target(action, C.STEAL_BASE)
            self.pending = PendingAction(
                ActionType.STEAL, actor=player_id, target=target, claimed_character=Character.CAPTAIN
            )
            self._log(f"P{player_id} claims Captain to steal from P{target}")
            self._open_action_challenge()
            return

    # ------------------------------------------------------------------
    # Challenge on the main action
    # ------------------------------------------------------------------
    def _open_action_challenge(self) -> None:
        assert self.pending is not None
        eligible = self._alive_in_order(
            (self.pending.actor + 1) % self.num_players, exclude={self.pending.actor}
        )
        if not eligible:
            self._post_action_claim_verified()
            return
        self._reaction_queue = eligible
        self.phase = Phase.AWAIT_CHALLENGE_ACTION
        self._decision_player = eligible[0]

    def _apply_challenge_action(self, player_id: int, action: int) -> None:
        if action == C.PASS:
            self._reaction_queue.pop(0)
            if not self._reaction_queue:
                self._post_action_claim_verified()
            else:
                self._decision_player = self._reaction_queue[0]
            return
        # CHALLENGE
        self._resolve_action_challenge(player_id)

    def _resolve_action_challenge(self, challenger: int) -> None:
        assert self.pending is not None and self.pending.claimed_character is not None
        actor = self.pending.actor
        claimed = self.pending.claimed_character
        if self._has_card(actor, claimed):
            self._log(
                f"P{challenger} challenges P{actor}'s {claimed.name} - it was real; P{challenger} loses influence"
            )
            self._swap_card(actor, claimed)
            self._queue_loss(challenger)
            self._begin_losses("post_claim")
        else:
            self._log(
                f"P{challenger} challenges P{actor}'s {claimed.name} - it was a bluff; the action fails"
            )
            self._queue_loss(actor)
            self._begin_losses("cancel")

    def _post_action_claim_verified(self) -> None:
        """Called once the main-action claim stands (unchallenged or verified)."""

        assert self.pending is not None
        self._reaction_queue = []
        if self.pending.action_type in (
            ActionType.FOREIGN_AID,
            ActionType.ASSASSINATE,
            ActionType.STEAL,
        ):
            self._open_block_window()
        else:  # TAX, EXCHANGE are unblockable
            self._apply_effect()

    # ------------------------------------------------------------------
    # Block window
    # ------------------------------------------------------------------
    def _open_block_window(self) -> None:
        assert self.pending is not None
        pa = self.pending
        if pa.action_type == ActionType.FOREIGN_AID:
            eligible = self._alive_in_order((pa.actor + 1) % self.num_players, exclude={pa.actor})
        else:  # steal / assassinate: only the target may block
            eligible = [pa.target] if pa.target is not None and self.players[pa.target].alive else []

        if not eligible:
            self._apply_effect()
            return
        self._reaction_queue = eligible  # type: ignore[assignment]
        self.phase = Phase.AWAIT_BLOCK
        self._decision_player = eligible[0]

    def _apply_block(self, player_id: int, action: int) -> None:
        assert self.pending is not None
        if action == C.PASS:
            self._reaction_queue.pop(0)
            if not self._reaction_queue:
                self._apply_effect()
            else:
                self._decision_player = self._reaction_queue[0]
            return
        # A block declaration (possibly a bluff).
        block_char = C.BLOCK_ACTION_TO_CHARACTER[action]
        self.pending.blocker = player_id
        self.pending.block_character = block_char
        self._log(f"P{player_id} claims {block_char.name} to block P{self.pending.actor}")
        self._open_block_challenge()

    # ------------------------------------------------------------------
    # Challenge on the block
    # ------------------------------------------------------------------
    def _open_block_challenge(self) -> None:
        assert self.pending is not None and self.pending.blocker is not None
        eligible = self._alive_in_order(self.pending.actor, exclude={self.pending.blocker})
        if not eligible:
            self._log("block goes unchallenged")
            self._end_turn()
            return
        self._reaction_queue = eligible
        self.phase = Phase.AWAIT_CHALLENGE_BLOCK
        self._decision_player = eligible[0]

    def _apply_challenge_block(self, player_id: int, action: int) -> None:
        if action == C.PASS:
            self._reaction_queue.pop(0)
            if not self._reaction_queue:
                self._log("block stands; action is blocked")
                self._end_turn()
            else:
                self._decision_player = self._reaction_queue[0]
            return
        # CHALLENGE
        self._resolve_block_challenge(player_id)

    def _resolve_block_challenge(self, challenger: int) -> None:
        assert (
            self.pending is not None
            and self.pending.blocker is not None
            and self.pending.block_character is not None
        )
        blocker = self.pending.blocker
        block_char = self.pending.block_character
        if self._has_card(blocker, block_char):
            self._log(
                f"P{challenger} challenges the {block_char.name} block - it was real; "
                f"P{challenger} loses influence and the action stays blocked"
            )
            self._swap_card(blocker, block_char)
            self._queue_loss(challenger)
            self._begin_losses("end_turn")
        else:
            self._log(
                f"P{challenger} challenges the {block_char.name} block - it was a bluff; "
                f"P{blocker} loses influence and the action proceeds"
            )
            self._queue_loss(blocker)
            self._begin_losses("apply_effect")

    # ------------------------------------------------------------------
    # Effect application
    # ------------------------------------------------------------------
    def _apply_effect(self) -> None:
        assert self.pending is not None
        pa = self.pending
        actor = self.players[pa.actor]

        if pa.action_type == ActionType.FOREIGN_AID:
            actor.coins += 2
            self._log(f"Foreign Aid succeeds (P{pa.actor} +2 -> {actor.coins})")
            self._end_turn()
        elif pa.action_type == ActionType.TAX:
            actor.coins += 3
            self._log(f"Tax succeeds (P{pa.actor} +3 -> {actor.coins})")
            self._end_turn()
        elif pa.action_type == ActionType.STEAL:
            assert pa.target is not None
            target = self.players[pa.target]
            amount = min(C.STEAL_AMOUNT, target.coins)
            target.coins -= amount
            actor.coins += amount
            self._log(f"Steal succeeds (P{pa.actor} takes {amount} from P{pa.target})")
            self._end_turn()
        elif pa.action_type == ActionType.ASSASSINATE:
            assert pa.target is not None
            if self.players[pa.target].alive:
                self._log(f"Assassination on P{pa.target} succeeds")
                self._queue_loss(pa.target)
            self._begin_losses("end_turn")
        elif pa.action_type == ActionType.EXCHANGE:
            self._open_exchange()
        else:
            self._end_turn()

    def _open_exchange(self) -> None:
        assert self.pending is not None
        actor = self.players[self.pending.actor]
        keep = len(actor.cards)
        drawn = [self.deck.pop(), self.deck.pop()]
        self._exchange_hand = sorted(actor.cards + drawn, key=int)
        self._exchange_combos = C.exchange_combos(len(self._exchange_hand), keep)
        self.phase = Phase.AWAIT_EXCHANGE
        self._decision_player = self.pending.actor
        self._log(f"P{self.pending.actor} draws 2 and must keep {keep} for the Exchange")

    def _apply_exchange(self, player_id: int, action: int) -> None:
        assert self._exchange_hand is not None and self._exchange_combos is not None
        combo = self._exchange_combos[C.decode_target(action, C.EXCHANGE_KEEP_BASE)]
        kept = [self._exchange_hand[i] for i in combo]
        returned = [c for i, c in enumerate(self._exchange_hand) if i not in combo]
        self.players[player_id].cards = list(kept)
        self.deck.extend(returned)
        self.rng.shuffle(self.deck)
        self._exchange_hand = None
        self._exchange_combos = None
        self._log(f"P{player_id} completes the Exchange")
        self._end_turn()

    # ------------------------------------------------------------------
    # Influence-loss resolution (queued, sequential)
    # ------------------------------------------------------------------
    def _queue_loss(self, player_id: int) -> None:
        self._loss_queue.append(player_id)

    def _begin_losses(self, continuation: str) -> None:
        """Start resolving queued influence losses, then run ``continuation``."""

        self._continuation = continuation
        self._advance_losses()

    def _advance_losses(self) -> None:
        while self._loss_queue:
            if self._check_game_over():
                return
            pid = self._loss_queue[0]
            player = self.players[pid]
            if not player.alive:
                self._loss_queue.pop(0)
                continue
            if len(player.cards) == 1:
                # No choice to make: auto-reveal the only card.
                self._loss_queue.pop(0)
                self._reveal_card(pid, 0)
                continue
            # A genuine choice: hand control to the player.
            self.phase = Phase.AWAIT_LOSE_INFLUENCE
            self._decision_player = pid
            return

        if self._check_game_over():
            return
        self._resume()

    def _apply_lose_influence(self, player_id: int, action: int) -> None:
        slot = C.decode_target(action, C.LOSE_SLOT_BASE)
        self._loss_queue.pop(0)
        self._reveal_card(player_id, slot)
        self._advance_losses()

    def _resume(self) -> None:
        cont = self._continuation
        self._continuation = None
        if cont == "post_claim":
            self._post_action_claim_verified()
        elif cont == "apply_effect":
            self._apply_effect()
        else:  # "end_turn" / "cancel" / None
            self._end_turn()

    # ------------------------------------------------------------------
    # Card / deck primitives
    # ------------------------------------------------------------------
    def _has_card(self, player_id: int, character: Character) -> bool:
        return character in self.players[player_id].cards

    def _swap_card(self, player_id: int, character: Character) -> None:
        """Return a proven ``character`` to the deck and draw a replacement."""

        player = self.players[player_id]
        player.cards.remove(character)
        self.deck.append(character)
        self.rng.shuffle(self.deck)
        player.cards.append(self.deck.pop())

    def _reveal_card(self, player_id: int, slot: int) -> None:
        player = self.players[player_id]
        card = player.cards.pop(slot)
        player.revealed.append(card)
        self._log(f"P{player_id} reveals and loses {card.name}")
        if not player.alive:
            self._log(f"P{player_id} is eliminated")

    # ------------------------------------------------------------------
    # Turn control / termination
    # ------------------------------------------------------------------
    def _end_turn(self) -> None:
        self.pending = None
        self._reaction_queue = []
        if self._check_game_over():
            return
        self.turn_count += 1
        nxt = self._next_alive(self.current_turn_player)
        self.current_turn_player = nxt
        self._decision_player = nxt
        self.phase = Phase.AWAIT_ACTION

    def _check_game_over(self) -> bool:
        alive = [p.player_id for p in self.players if p.alive]
        if len(alive) <= 1:
            self.winner = alive[0] if alive else None
            self.phase = Phase.GAME_OVER
            self.pending = None
            self._reaction_queue = []
            self._loss_queue = []
            return True
        return False

    def _log(self, message: str) -> None:
        self.last_events.append(message)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_observation(self, player_id: Optional[int] = None) -> dict[str, Any]:
        """Return a JSON-serialisable observation.

        Args:
            player_id: If ``None``, returns the full god-view including every
                player's hidden cards (handy for a frontend / debugging). If a
                player id is given, returns that agent's partial view: only that
                player's hidden cards are revealed; opponents expose counts only.

        Returns:
            A nested dict describing players, the pending action, the current
            phase / decision-maker, and the legal moves for the requested
            perspective.
        """

        players_obs = []
        for p in self.players:
            reveal_hidden = player_id is None or p.player_id == player_id
            players_obs.append(
                {
                    "player_id": p.player_id,
                    "coins": p.coins,
                    "alive": p.alive,
                    "influence_count": p.influence_count,
                    "hidden_cards": [c.name for c in p.cards] if reveal_hidden else None,
                    "revealed_cards": [c.name for c in p.revealed],
                }
            )

        pending_obs = None
        if self.pending is not None:
            pa = self.pending
            pending_obs = {
                "action_type": pa.action_type.name,
                "actor": pa.actor,
                "target": pa.target,
                "claimed_character": pa.claimed_character.name if pa.claimed_character is not None else None,
                "blocker": pa.blocker,
                "block_character": pa.block_character.name if pa.block_character is not None else None,
            }

        over, winner = self.is_game_over()
        legal_for = player_id if player_id is not None else self._decision_player
        return {
            "num_players": self.num_players,
            "phase": self.phase.name,
            "current_turn_player": self.current_turn_player,
            "current_player": self.current_player,
            "turn_count": self.turn_count,
            "deck_size": len(self.deck),
            "players": players_obs,
            "pending": pending_obs,
            "legal_moves": self.get_legal_moves(legal_for),
            "game_over": over,
            "winner": winner,
        }

    def get_observation_vector(self, player_id: int) -> np.ndarray:
        """Return a fixed-length float32 encoding of ``player_id``'s view.

        The layout is stable across games and player counts (padded to
        ``MAX_PLAYERS``) so it can feed directly into an RL policy network. It
        concatenates, in order: turn counter and deck size; one-hot phase,
        acting player and deciding player; the pending-action context; per-player
        public stats (coins, alive flag, influence count, per-character revealed
        counts); the observer's own per-character hand counts; and the current
        action mask.
        """

        n_char = len(Character)
        n_phase = len(Phase)
        mp = C.MAX_PLAYERS

        parts: list[np.ndarray] = []
        parts.append(np.array([self.turn_count, len(self.deck)], dtype=np.float32))
        parts.append(self._one_hot(int(self.phase), n_phase))
        parts.append(self._one_hot(self.current_turn_player, mp))
        parts.append(self._one_hot(self._decision_player, mp))

        # Pending-action context.
        pend = np.zeros(1 + len(ActionType) + mp + n_char + mp + n_char, dtype=np.float32)
        if self.pending is not None:
            pa = self.pending
            off = 0
            pend[off] = 1.0
            off += 1
            pend[off + int(pa.action_type)] = 1.0
            off += len(ActionType)
            if pa.target is not None:
                pend[off + pa.target] = 1.0
            off += mp
            if pa.claimed_character is not None:
                pend[off + int(pa.claimed_character)] = 1.0
            off += n_char
            if pa.blocker is not None:
                pend[off + pa.blocker] = 1.0
            off += mp
            if pa.block_character is not None:
                pend[off + int(pa.block_character)] = 1.0
        parts.append(pend)

        # Per-player public stats.
        for i in range(mp):
            block = np.zeros(3 + n_char, dtype=np.float32)
            if i < self.num_players:
                p = self.players[i]
                block[0] = p.coins
                block[1] = 1.0 if p.alive else 0.0
                block[2] = p.influence_count
                for c in p.revealed:
                    block[3 + int(c)] += 1.0
            parts.append(block)

        # Observer's own hidden hand (per-character counts).
        own = np.zeros(n_char, dtype=np.float32)
        for c in self.players[player_id].cards:
            own[int(c)] += 1.0
        parts.append(own)

        # Action mask.
        mask = np.zeros(C.ACTION_SPACE_SIZE, dtype=np.float32)
        for a in self.get_legal_moves(player_id):
            mask[a] = 1.0
        parts.append(mask)

        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _one_hot(index: int, size: int) -> np.ndarray:
        vec = np.zeros(size, dtype=np.float32)
        if 0 <= index < size:
            vec[index] = 1.0
        return vec


def _random_playout(num_players: int = 4, seed: int = 0, max_decisions: int = 10_000) -> None:
    """Run one game where every decision-maker picks a uniformly random legal move.

    This doubles as a smoke test that the engine always terminates with exactly
    one winner and never offers an illegal move.
    """

    engine = BoardGameEngine(num_players=num_players, seed=seed)
    rng = random.Random(seed)

    print(f"=== Coup random playout (players={num_players}, seed={seed}) ===")
    decisions = 0
    while not engine.is_game_over()[0] and decisions < max_decisions:
        actor = engine.current_player
        assert actor is not None
        legal = engine.get_legal_moves(actor)
        assert legal, "a non-terminal state must offer at least one legal move"
        phase_before = engine.phase.name  # phase in which the action was taken
        action = rng.choice(legal)
        info = engine.apply_action(actor, action)
        decisions += 1
        print(f"[{decisions:03d}] ({phase_before}) P{actor} -> {C.action_name(action)}")
        for event in info["events"]:
            print(f"        - {event}")

    over, winner = engine.is_game_over()
    print("=" * 48)
    if over and winner is not None:
        print(f"Game over after {decisions} decisions. Winner: Player {winner}")
    else:
        print(f"Stopped after {decisions} decisions without a winner (unexpected)")


if __name__ == "__main__":
    # Mock game loop: two or more players making random legal moves until the end.
    for demo_seed in (0, 1, 2):
        _random_playout(num_players=2, seed=demo_seed)
        print()
    _random_playout(num_players=4, seed=7)
