# Reward Design

## Overview

Two reward modes are supported, selected per-episode via the `sparse_reward` flag
passed to `/reset`.

---

## Shaped reward (default)

Used by: `nav_shaped`, `farming`, `multitask`, `nav_curriculum` experiments.

### Navigation

| Signal | Value | Condition |
|--------|-------|-----------|
| Progress shaping | `+0.15 × Δdist` | Each step — reward getting closer, penalise moving away |
| Time penalty | `−0.015` | Each step — incentivise efficiency |
| Jump bonus | `+0.5` | Agent successfully jumps over a 1-block wall |
| Success bonus | `+10.0` | Distance ≤ 1.5 blocks from target |
| Invalid action | `−0.10` | Action failed (e.g. `forward` blocked, `jump` preconditions not met) |
| Stuck penalty | `−0.25` | Agent has not moved meaningfully for ≥ 8 consecutive steps |

**Design rationale.** The progress coefficient (0.15) is large enough to provide
a dense gradient signal but small enough that the agent cannot earn a positive
return by oscillating in place (progress cancels every other step, time penalty
accumulates). The success bonus dominates all other signals over a typical episode
(10 >> 0.15 × ~10 typical progress), ensuring goal-directed rather than
exploration-by-shaping behaviour.

### Farming

All navigation signals apply, plus:

| Signal | Value | Condition |
|--------|-------|-----------|
| Near target | `+0.05` | Distance ≤ 1.10 blocks |
| Crop in front | `+0.10` | Mature wheat directly ahead at interact range |
| Pre-interact reward | `+2.0` | Interact action while mature crop is in front |
| Successful harvest | `+4.0` | `interact` destroys a mature crop (`lastInteractValid = true`) |
| Pointless interact | `−0.05` | Interact while no mature crop is in front |
| Drift penalty | `−0.10` | Agent moves away after getting within 1.15 blocks |
| Success bonus | `+10.0` | Crop harvested and agent within 1.05 blocks |

**Design rationale.** Farming requires a two-phase behaviour: (1) navigate to
the crop, (2) use the `interact` action at the correct moment. Without the
`crop_in_front` and pre-interact bonuses the agent rarely discovers that
`interact` is the goal action. The `pointless_interact` penalty prevents the
degenerate strategy of spamming interact from a distance.

---

## Sparse reward

Used by: `nav_sparse` experiment.

| Signal | Value | Condition |
|--------|-------|-----------|
| Success bonus | `+10.0` | Terminal — task completed |
| Time penalty | `−0.01` | Each step — tiny, just to break ties |
| Invalid action | `−0.10` | Action failed |
| Stuck penalty | `−0.25` | Agent stuck ≥ 8 steps |

No distance shaping, no progress signal. The agent must discover the goal
entirely through exploration.

**Comparison purpose.** Running the same PPO setup under both reward modes lets
us measure: (a) how many extra timesteps sparse training needs to converge, and
(b) whether the final policy generalises differently. Sparse training typically
needs `ent_coef=0.10` (vs 0.05 for shaped) because the reward signal is
near-zero for the first thousands of episodes.

---

## Curriculum effect on rewards

The curriculum scheduler only changes the *difficulty parameters* passed to
`/reset` (target distance, obstacle count). The per-step reward function is
identical across all curriculum levels — this is intentional so that the learning
signal does not change as the curriculum advances, keeping the comparison clean.

---

## Tuning history

| Parameter | Old value | New value | Reason |
|-----------|-----------|-----------|--------|
| Progress coeff | 0.10 | **0.15** | Too weak on longer-distance episodes |
| Time penalty | 0.01 | **0.015** | Agents were taking unnecessarily long paths |
| `cropInFront` bonus | 0.25/step | **0.10/step** | Old value caused "camping in front of crop" |
| Stuck threshold | 5 steps | **8 steps** | Turning in place triggered false positives |
| Stuck penalty | 0.10 | **0.25** | Insufficient deterrent against spinning |
