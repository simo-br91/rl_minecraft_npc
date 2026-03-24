# Reward Design

## Overview

Two reward modes are supported, selected per-episode via the `sparse_reward` flag
passed to `/reset`.

---

## Shaped reward (default)

Used by: `nav_shaped`, `farming`, `combat`, `multitask`, `nav_curriculum` experiments.

### Navigation

| Signal | Value | Condition |
|--------|-------|-----------|
| Progress shaping | `+0.15 × Δdist` | Each step — reward closing distance, penalise moving away |
| Time penalty | `−0.015` | Each step — incentivise efficiency |
| Jump bonus | `+0.30` | Agent successfully jumps over a 1-block wall |
| Success bonus | `+10.0` | Distance ≤ 1.5 blocks from target |
| Invalid action | `−0.10` | Action failed (e.g. `forward` blocked, `jump` preconditions not met) |
| Stuck penalty | `−0.25` | Agent has not moved meaningfully for ≥ 8 consecutive steps |
| Sprint food cost | `−0.005` | Per step the agent is sprinting |

**Design rationale.** The progress coefficient (0.15) is large enough to provide
a dense gradient signal but small enough that the agent cannot earn a positive
return by oscillating in place. The success bonus (10) dominates all other
signals over a typical episode, ensuring goal-directed behaviour.

### Farming

All navigation signals apply, plus:

| Signal | Value | Condition |
|--------|-------|-----------|
| Near crop | `+0.05` | Distance ≤ 1.10 blocks to active crop |
| Crop in front | `+0.10` | Mature wheat directly ahead at interact range |
| Pre-interact reward | `+2.0` | Interact action while mature crop is in front |
| Successful harvest | `+4.0` | `interact` destroys a mature crop |
| All crops done | `+10.0` | All crops harvested (episode success) |
| Pointless interact | `−0.05` | Interact while no mature crop is in front |
| Drift penalty | `−0.10` | Agent moves away after getting within 1.15 blocks of crop |

**Design rationale.** Farming requires two-phase behaviour: (1) navigate to the
crop, (2) use `interact` at the right moment. Without the `crop_in_front` and
pre-interact bonuses the agent rarely discovers that `interact` is the goal
action. The `pointless_interact` penalty prevents spamming interact from a
distance.

### Combat

| Signal | Value | Condition |
|--------|-------|-----------|
| Time penalty | `−0.015` | Each step |
| Attack connected | `+1.0` | Melee swing hit a mob |
| Kill bonus | `+8.0` | Per mob killed |
| All mobs killed | `+10.0` | Terminal success |
| Took damage | `−1.0` | Per hit received |
| Agent died | `−5.0` | Agent health ≤ 0 |
| Invalid action | `−0.10` | Action failed |
| Stuck penalty | `−0.25` | Stuck ≥ 8 steps |

**Design rationale.** The large kill bonus relative to attack-connected ensures
the agent prioritises finishing mobs over trading blows. The death penalty creates
incentive to eat when low on health. The attack-connected bonus helps discover the
`attack` action early.

### Multitask (combined episode)

All three tasks are active simultaneously. The reward blends contributions:

| Signal | Value | Condition |
|--------|-------|-----------|
| Time penalty | `−0.015` | Each step |
| Navigation progress | `+0.08 × Δdist` | Closing on current target |
| Crop in front | `+0.05` | Mature wheat ahead |
| Pre-harvest | `+1.0` | Interact while crop in front |
| Successful harvest | `+2.0` | Crop harvested |
| Attack connected | `+2.0` | Swing hit mob |
| Kill bonus | `+8.0` | Per mob killed |
| Took damage | `−1.0` | Per hit received |
| Agent died | `−5.0` | Death |
| Food management | `+1.0` | Eating while food < 10 |
| All objectives done | `+10.0` | Terminal success |
| Jump bonus | `+0.30` | Jumped over wall |
| Invalid action | `−0.10` | Failed action |
| Stuck penalty | `−0.25` | Stuck ≥ 8 steps |
| Sprint food cost | `−0.005` | Per sprint step |

**Design rationale.** Navigation progress is down-weighted (0.08 vs 0.15) because
the agent needs headroom to stop navigating and fight a nearby mob without being
penalised heavily. Farming harvest bonuses are halved compared to single-task to
prevent the agent from ignoring mobs in favour of crops.

---

## Sparse reward

Used by: `nav_sparse` experiment.

| Signal | Value | Condition |
|--------|-------|-----------|
| Success bonus | `+10.0` | Terminal — task completed |
| Time penalty | `−0.01` | Each step |
| Invalid action | `−0.10` | Failed action |
| Stuck penalty | `−0.25` | Stuck ≥ 8 steps |
| Auto-truncation | — | Episode ends if stuck ≥ 20 consecutive steps |

No distance shaping, no progress signal. The agent must discover the goal
entirely through exploration.

**Auto-truncation in sparse mode.** Without this, a stuck agent consumes the
entire episode budget (maxSteps) doing nothing useful. The 20-step auto-truncate
recycles the rollout budget 7–10× faster on stuck episodes, substantially
accelerating sparse training wall-clock time.

**Comparison purpose.** Running the same PPO setup under both reward modes measures:
(a) how many extra timesteps sparse training needs to converge, and (b) whether
the final policy generalises differently. Sparse training uses `ent_coef=0.10`
(vs 0.05 for shaped) because the reward signal is near-zero for the first
thousands of episodes.

---

## Curriculum effect on rewards

The curriculum scheduler only changes the *difficulty parameters* passed to
`/reset` (target distance, obstacle count, crop count). The per-step reward
function is identical across all curriculum levels — this is intentional so
that the learning signal does not change as the curriculum advances.

---

## Tuning history

| Parameter | Old value | New value | Reason |
|-----------|-----------|-----------|--------|
| Progress coeff | 0.10 | **0.15** | Too weak on longer-distance episodes |
| Time penalty | −0.01 | **−0.015** | Agents took unnecessarily long paths |
| `cropInFront` bonus | 0.25/step | **0.10/step** | Old value caused "camping in front of crop" |
| Stuck threshold | 5 steps | **8 steps** | Turning in place triggered false positives |
| Stuck penalty | −0.10 | **−0.25** | Insufficient deterrent against spinning |
| Jump bonus | +0.5 | **+0.30** | Was disproportionately large vs progress reward |
| `stuck_norm` divisor | 10.0 | **15.0** | Feature saturated at 10 steps; agent blind to prolonged stuckness |
| Sparse auto-truncate | none | **20 stuck steps** | Eliminates wasted rollout budget on stuck episodes |
