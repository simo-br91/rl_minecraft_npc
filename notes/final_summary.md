# Final Summary

## Project: Learning Multi-Skill NPC Behavior in Minecraft via RL

---

## 1. What was built

A complete end-to-end reinforcement learning system inside Minecraft 1.20.1,
consisting of:

**Minecraft side (Forge mod `rlnpc`)**
- A controllable `RLNpcEntity` (PathfinderMob with AI disabled)
- An HTTP bridge server (`BridgeServer`) exposing `/health`, `/reset`, `/step`
- `EnvironmentManager`: episode lifecycle, task configuration, reward computation
- Two tasks: **navigation** (reach a gold block target) and **farming** (harvest
  mature wheat)
- Sparse/shaped reward toggle and curriculum difficulty parameters passed per reset
- Obstacle placement (1-block stone walls) for navigation

**Python side**
- `MinecraftEnv`: Gymnasium-compatible wrapper over the HTTP bridge
- PPO training scripts for all experiments (navigation shaped, navigation sparse,
  navigation curriculum, farming, multi-task)
- `CurriculumScheduler`: 4-level difficulty progression with rolling success-rate
  advancement threshold
- Evaluation scripts with per-task statistics
- Comparison and generalization plotting pipeline

---

## 2. Experiments run

| # | Experiment | Script | Key result |
|---|-----------|--------|------------|
| 1 | Navigation, shaped rewards | `train_navigation.py` | Baseline success rate |
| 2 | Navigation, sparse rewards | `train_nav_sparse.py` | Slower convergence vs shaped |
| 3 | Navigation, curriculum (4 levels) | `train_nav_curriculum.py` | Faster early learning, comparable final SR |
| 4 | Farming single-task | `train_farming.py` | Harder; requires more timesteps |
| 5 | Multi-task shared policy | `train_multitask.py` | One policy solves both tasks |

**Comparison plots:** `python_rl/logs/plots/`
- `shaped_vs_sparse.png` — reward and success-rate curves side by side
- `curriculum_vs_nocurriculum.png` — with level-advancement markers
- `multitask_overview.png` — multi-task vs single-task per task
- `final_success_rates.png` — bar chart of final SR across all experiments

**Generalization test:** `generalization_test.py` evaluates any checkpoint
across 6 held-out difficulty configurations (A–F) varying distance and obstacle
count independently of the training distribution.

---

## 3. Architecture

```
Minecraft 1.20.1
└── Forge mod rlnpc
    ├── RLNpcEntity      — controllable entity (no vanilla AI)
    ├── BridgeServer     — HTTP on 127.0.0.1:8765
    │   ├── /health
    │   ├── /reset  ← task, sparse_reward, min_dist, max_dist, num_obstacles
    │   └── /step   ← action (0-5)
    └── EnvironmentManager
        ├── configureNavigationTask()
        ├── configureFarmingTask()
        ├── placeNavigationObstacles()
        ├── computeReward()       ← shaped or sparse branch
        └── ObservationBuilder   ← 11-dim feature vector

Python
└── MinecraftEnv (Gymnasium)
    ├── action_space  : Discrete(6)
    │   0 forward | 1 turn_left | 2 turn_right
    │   3 interact | 4 no_op   | 5 jump
    └── observation_space : Box(11,)
        dx, dz, dist, yaw_norm, blocked_front, on_ground,
        stuck_norm, task_id, crop_in_front, near_target,
        obstacle_1block_ahead

Training
└── PPO (Stable-Baselines3)
    ├── Single-task: navigation shaped / sparse / curriculum
    ├── Single-task: farming
    └── Multi-task:  shared policy, task sampled 2:1 farming:navigation
                     warm-start from single-task checkpoint
```

---

## 4. Observation space (11 dimensions)

| Index | Feature | Range | Notes |
|-------|---------|-------|-------|
| 0 | `dx` | ±1000 | Target X − Agent X |
| 1 | `dz` | ±1000 | Target Z − Agent Z |
| 2 | `distance` | 0–1000 | XZ Euclidean distance to target |
| 3 | `yaw_norm` | −1–1 | Agent yaw / 180 |
| 4 | `blocked_front` | 0/1 | Solid block directly ahead |
| 5 | `on_ground` | 0/1 | Solid block below agent |
| 6 | `stuck_norm` | 0–1 | stuck_steps / 10, clamped |
| 7 | `task_id` | 0/1 | 0 = navigation, 1 = farming |
| 8 | `crop_in_front` | 0/1 | Mature wheat directly ahead |
| 9 | `near_target` | 0/1 | distance ≤ 1.10 |
| 10 | `obstacle_1block_ahead` | 0/1 | Jumpable 1-block wall ahead |

---

## 5. Key findings

**Shaped vs sparse.** Shaped rewards converge significantly faster and to a
higher final success rate. Sparse training with the same timestep budget achieves
a noticeably lower success rate, consistent with the exploration bottleneck in
sparse RL.

**Curriculum.** The 4-level curriculum (distance 3–6 → 10–18 blocks, 0–3
obstacles) accelerates early learning (the agent achieves >70% success on easy
levels before attempting harder ones) and reaches comparable or higher final
performance on the hardest level compared to training directly at hard difficulty.

**Multi-task.** A single shared PPO policy warm-started from a single-task
checkpoint successfully learns both navigation and farming with only 200k shared
timesteps. Per-task success rates are slightly below the corresponding
single-task baselines (negative transfer), but both tasks are solved.

**Generalization.** The navigation policy generalises well to the training
distribution (config C) and nearby configurations, but success rates drop
noticeably on the hardest held-out configuration (F: 14–20 blocks, 3 obstacles),
suggesting the policy has memorised aspects of the training difficulty range.

---

## 6. Limitations

- Single Minecraft instance: no parallel environment collection.
- Farming only implements harvest; plant-then-harvest is a future extension.
- No pixel-based observation; the agent uses hand-engineered features only.
- Jump is a discrete teleport, not physics-based; visually unrealistic.
- No combat task (out of scope for this sprint).

---

## 7. Future work

- Replace hand-crafted features with a voxel grid encoder (CNN or PointNet).
- Implement full farming cycle: till → plant → wait/simulate → harvest.
- Add combat task and hierarchical controller (high-level skill selector).
- Parallel environments using a headless Minecraft server.
- Automatic hyperparameter tuning (Optuna) across experiments.
