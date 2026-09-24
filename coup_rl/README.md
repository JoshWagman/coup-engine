# Coup RL (`coup_rl`)

PettingZoo env, run recording, analytics CLI, and shared-policy MaskablePPO
self-play. Game rules stay in `coup/`; this package sits beside it.

Run every command from the **repo root** (`coup-engine/`) with the venv on.
zsh treats `<` and `>` as redirects — do not type placeholders with angle brackets.
Use the folder name `train` prints, for example `data/runs/20260920-201059-f67a7b9a`.

```bash
cd coup-engine
source .venv/bin/activate
python -m coup_rl.train --help
python -m coup_rl.analytics --help
```

---

## Install

```bash
pip install -r requirements.txt      # engine: numpy, pytest
pip install -r requirements-rl.txt   # + PettingZoo, Gymnasium, PyTorch, sb3-contrib, TensorBoard
```

---

## Train

```bash
python -m coup_rl.train [flags]
```

Self-play with **one shared MaskablePPO**. All seats are the same network.
`--num-players` is set per run (2–6). Each run creates a new folder under
`--run-root` (default `data/runs/`). Rewards are wrapper-only: `-1` on
elimination, `+1` on a win (not zero-sum for `n > 2`).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-players` | `4` | Seats (2–6) |
| `--timesteps` | `200000` | Decisions to collect |
| `--n-steps` | `2048` | Decisions per PPO update |
| `--n-epochs` | `4` | PPO epochs per update |
| `--seed` | `0` | Policy + first game seed (later games `seed+1`, …) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--run-root` | `data/runs` | Parent of run folders |
| `--learning-rate` | `0.0003` | Adam LR |
| `--checkpoint-every` | `10` | Write `checkpoints/update_N.pt` (always writes `final.pt`) |

Not on the CLI (`TrainConfig` defaults): minibatch 64, gamma 0.99, GAE λ 0.95,
clip 0.2, entropy 0.01, value coef 0.5, grad clip 0.5.

### Run commands

```bash
# default 4-player, 200k timesteps
python -m coup_rl.train

# same as the defaults, written out
python -m coup_rl.train --num-players 4 --timesteps 200000 --n-steps 2048 --n-epochs 4 --seed 0 --device cpu --run-root data/runs --learning-rate 0.0003 --checkpoint-every 10

# player counts
python -m coup_rl.train --num-players 2 --seed 0
python -m coup_rl.train --num-players 3 --seed 0
python -m coup_rl.train --num-players 4 --seed 0
python -m coup_rl.train --num-players 5 --seed 0
python -m coup_rl.train --num-players 6 --seed 0

# smoke
python -m coup_rl.train --num-players 2 --timesteps 64 --n-steps 32 --n-epochs 1 --seed 0

# longer / GPU / custom folder
python -m coup_rl.train --num-players 4 --timesteps 1000000 --seed 0
python -m coup_rl.train --num-players 4 --timesteps 200000 --device cuda
python -m coup_rl.train --num-players 4 --run-root /tmp/coup-runs --seed 1
```

Prints `Run written to data/runs/TIMESTAMP-HASH`. That path is `RUN` below.

### Run folder

```
data/runs/TIMESTAMP-HASH/
  config.json       flags, git sha, reward spec
  events.jsonl      structured events (append-only)
  episodes.jsonl    one row per finished game (winner, length, seed)
  metrics/          TensorBoard
  checkpoints/      update_N.pt and final.pt
```

`data/` is gitignored.

---

## Analyze

```bash
python -m coup_rl.analytics RUN [flags]
```

`RUN` is the folder `train` printed. All of these print a **wins-by-seat** table
first. `--player` and `--top` cannot be combined. `--top` is the seat with the
most wins (lowest index on a tie) — same shared policy, so this is positional,
not a different agent.

| Flag | Meaning |
| --- | --- |
| `--claims` | Claim-vs-held tables. **Default** if you omit the other report flags. |
| `--phases` | False claims by early / mid / late |
| `--hands` | Starting two-card hands vs wins (replays episode seeds) |
| `--bluff-wins` | Each seat's win rate vs bluff rate (overall and by per-game bluff bin) |
| `--challenges` | Challenge rate, accuracy, and challenges made without seeing all 3 copies |
| `--character NAME` | With `--claims`: only `DUKE`, `ASSASSIN`, `CAPTAIN`, `AMBASSADOR`, or `CONTESSA` |
| `--player N` | Only seat `player_N` |
| `--top` | Only the highest-win seat |

You can combine `--phases`, `--hands`, `--bluff-wins`, and `--challenges`.
Adding `--claims` prints the claim tables as well.

### Claim vs held (`--claims`, default)

Whether the actor **held** the claimed character at claim time (pre-step
snapshot). Split by action vs block and by character.

```bash
python -m coup_rl.analytics RUN
python -m coup_rl.analytics RUN --claims
python -m coup_rl.analytics RUN --top
python -m coup_rl.analytics RUN --top --character DUKE
python -m coup_rl.analytics RUN --player 2
python -m coup_rl.analytics RUN --player 2 --character CONTESSA
python -m coup_rl.analytics RUN --character ASSASSIN
python -m coup_rl.analytics RUN --character CAPTAIN
python -m coup_rl.analytics RUN --character AMBASSADOR
python -m coup_rl.analytics RUN --character CONTESSA
python -m coup_rl.analytics RUN --character DUKE
```

### False claims by phase (`--phases`)

Which characters were faked most in early / mid / late. Phases are reconstructed
from the event log:

- **early** — no Coup or Assassinate action taken yet (and not already late)
- **late** — 2 or 3 players left, each with 1 influence
- **mid** — after the first Coup/Assassinate, until late

Prints bluff counts, share of that phase’s bluffs, and bluff rate per character,
plus action vs block.

```bash
python -m coup_rl.analytics RUN --phases
python -m coup_rl.analytics RUN --phases --top
python -m coup_rl.analytics RUN --phases --player 2
```

### Starting hands vs wins (`--hands`)

Opening hands are not in `events.jsonl`. This replays
`BoardGameEngine(..., seed=episode_seed)` from `episodes.jsonl`.
`win_rate` is wins / times that pair was dealt (to anyone, or to `--player` /
`--top` only). 4-player random baseline is 25%.

```bash
python -m coup_rl.analytics RUN --hands
python -m coup_rl.analytics RUN --hands --top
python -m coup_rl.analytics RUN --hands --player 2
```

### Win rate vs bluff rate (`--bluff-wins`)

For each seat: overall `win_rate` and `bluff_rate` (false claims / claims), then
win rate in games where that seat's **in-game** bluff rate fell in `0%`,
`1-33%`, `34-66%`, `67-100%`, or `no_claims`.

```bash
python -m coup_rl.analytics RUN --bluff-wins
python -m coup_rl.analytics RUN --bluff-wins --top
python -m coup_rl.analytics RUN --bluff-wins --player 2
```

Pooled plot of **all seats** (bluff rate on x, win rate on y). Writes
`RUN/win_vs_bluff.png` unless you pass a path:

```bash
python -m coup_rl.analytics RUN --plot
python -m coup_rl.analytics RUN --plot /tmp/win_vs_bluff.png
```

### Challenges (`--challenges`)

How often a seat Challenges vs Passes in a challenge window, how often they
were right (`was_bluff`), and how often they challenged **without being able
to account for all 3 copies** of that character (own known hidden cards +
public revealed cards). After Exchange or a proven-claim reshuffle the hidden
hand is treated as unknown.

```bash
python -m coup_rl.analytics RUN --challenges
python -m coup_rl.analytics RUN --challenges --top
python -m coup_rl.analytics RUN --challenges --player 2
```

### Combine

```bash
python -m coup_rl.analytics RUN --phases --hands
python -m coup_rl.analytics RUN --bluff-wins --phases
python -m coup_rl.analytics RUN --challenges --phases
python -m coup_rl.analytics RUN --phases --hands --claims --top
```

### TensorBoard

```bash
tensorboard --logdir data/runs
tensorboard --logdir RUN/metrics
```

Open the URL it prints (usually `http://localhost:6006`).

---

## Tests

```bash
pytest
pytest tests/test_env.py tests/test_recorder.py tests/test_analytics.py tests/test_train.py
pytest tests/test_engine.py
```

---

## Library notes

Python helpers used by the CLI: `claim_truth_table`, `phase_bluff_table`,
`starting_hand_table`, `bluff_vs_win_table`, `load_events`, `load_episodes`,
`win_counts`, `top_winner`.

```python
from coup_rl.analytics import load_events, phase_bluff_table, starting_hand_table, load_episodes

run = "data/runs/20260920-201059-f67a7b9a"
print(phase_bluff_table(load_events(run)))
print(starting_hand_table(load_episodes(run)))
```

`CoupAECEnv` is a PettingZoo `AECEnv` (no extra CLI): `player_0` … `player_{n-1}`,
Dict obs `{observation, action_mask}`, `Discrete(36)`. Pass `EventRecorder` to
log god-view hands; they are not in the agent observation.

```python
from coup_rl import CoupAECEnv
env = CoupAECEnv(num_players=4, seed=0)
env.reset()
obs, reward, term, trunc, info = env.last()
```

Engine demo (not RL):

```bash
python -m coup.engine
```

### Event types (`events.jsonl`)

| `event_type` | Fields of interest |
| --- | --- |
| `episode_start` | `episode_id`, `num_players`, `seed` |
| `decision` | `actor`, `action_id`, `action_name`, `phase`, `step` |
| `claim` | `character`, `had_character`, `copies_held`, `claim_kind=action` |
| `block_claim` | same, `claim_kind=block` |
| `challenge` / `challenge_resolved` | `target`, `character`, `was_bluff` |
| `lose_influence` | `actor` is who lost the card |
| `elimination` / `game_end` | `winner` on `game_end` |
