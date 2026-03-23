# Minecraft RL Multi-Skill NPC

End-to-end reinforcement learning inside Minecraft 1.20.1 using a custom Forge
mod and Stable-Baselines3 PPO.  A controllable NPC learns navigation and farming
skills from scratch.

---

## Repository layout

```
rl_minecraft_npc/
├── minecraft_mod/              Forge 1.20.1 mod (Java 17)
│   └── src/main/java/com/aberrada/rlnpc/
│       ├── RLNpcMod.java           Mod entry point
│       ├── RLNpcEntity.java        Controllable NPC entity
│       ├── ModEntities.java        Entity type registration
│       ├── BridgeServer.java       HTTP bridge on localhost:8765
│       ├── EnvironmentManager.java Episode logic, reward, task config
│       ├── ActionExecutor.java     Discrete action execution + collision
│       ├── ObservationBuilder.java 11-dim feature vector
│       ├── EpisodeState.java       Per-episode state (task, flags, counters)
│       ├── ClientModEvents.java    Client-side entity renderer registration
│       └── RLNpcRenderer.java      NPC renderer (Steve skin)
│
├── python_rl/
│   ├── env/
│   │   └── minecraft_env.py        Gymnasium wrapper over HTTP bridge
│   ├── train/
│   │   ├── curriculum_scheduler.py 4-level curriculum with auto-advancement
│   │   ├── train_navigation.py     Navigation, shaped rewards  (baseline)
│   │   ├── train_nav_sparse.py     Navigation, sparse rewards  (comparison)
│   │   ├── train_nav_curriculum.py Navigation, 4-level curriculum
│   │   ├── train_farming.py        Farming single-task
│   │   └── train_multitask.py      Shared policy: navigation + farming
│   ├── eval/
│   │   ├── evaluate.py             Navigation deterministic rollout
│   │   ├── evaluate_farming.py     Farming deterministic rollout
│   │   ├── evaluate_multitask.py   Per-task breakdown for shared model
│   │   ├── plot_results.py         Reward + success-rate curve for one run
│   │   ├── compare_experiments.py  Side-by-side comparison plots
│   │   └── generalization_test.py  Held-out difficulty configs (A–F)
│   └── configs/
│       ├── nav_shaped.yaml
│       ├── nav_sparse.yaml
│       ├── nav_curriculum.yaml
│       ├── farming.yaml
│       └── multitask.yaml
│
└── notes/
    ├── reward_design.md    Full reward function documentation
    ├── bugs.md             Bug log (fixed + known issues)
    ├── final_summary.md    Architecture, findings, limitations
    └── experiment_log.md   Day-by-day experiment notes
```

---

## Quick start

### 1. Run Minecraft

Open the project in IntelliJ (or use the VSCode launch config) and run the
`runClient` configuration.  Create or open a singleplayer **flat world** and
wait until you see:

```
[rlnpc] RL environment ready. HTTP bridge listening on http://127.0.0.1:8765
```

### 2. Install Python dependencies

```bash
pip install stable-baselines3 gymnasium requests matplotlib pandas numpy
```

### 3. Run experiments (in order)

```bash
# Baseline — navigation with shaped rewards
python -m python_rl.train.train_navigation

# Comparison — navigation with sparse rewards
python -m python_rl.train.train_nav_sparse

# Curriculum — 4-level difficulty progression
python -m python_rl.train.train_nav_curriculum

# Single-task farming
python -m python_rl.train.train_farming

# Multi-task shared policy (warm-starts from farm_run1)
python -m python_rl.train.train_multitask
```

### 4. Evaluate

```bash
# Quick rollout (5 episodes, prints per-step info)
python -m python_rl.eval.evaluate --model nav_shaped_run1
python -m python_rl.eval.evaluate_farming --model farm_run1
python -m python_rl.eval.evaluate_multitask --model multitask_run1

# Generalization test across 6 held-out configs
python -m python_rl.eval.generalization_test --model nav_shaped_run1
python -m python_rl.eval.generalization_test --model nav_curriculum_run1
```

### 5. Generate plots

```bash
# Single-run reward + success-rate plot
python -m python_rl.eval.plot_results \
    --monitor nav_shaped_monitor.csv \
    --success nav_shaped_success.csv \
    --title "Navigation (shaped)"

# All comparison plots at once
python -m python_rl.eval.compare_experiments
```

Plots are saved to `python_rl/logs/plots/`.

---

## Action space

| ID | Action | Description |
|----|--------|-------------|
| 0 | `forward` | Move 0.35 blocks in the look direction |
| 1 | `turn_left` | Rotate −15° yaw |
| 2 | `turn_right` | Rotate +15° yaw |
| 3 | `interact` | Harvest crop directly ahead (farming only) |
| 4 | `no_op` | Do nothing |
| 5 | `jump` | Teleport up 1 block (requires 1-block wall ahead) |

## Observation space (11 dims)

`[dx, dz, distance, yaw_norm, blocked_front, on_ground, stuck_norm, task_id, crop_in_front, near_target, obstacle_1block_ahead]`

See `notes/reward_design.md` for full reward documentation.

---

## Curriculum levels

| Level | Distance | Obstacles | Description |
|-------|----------|-----------|-------------|
| 1 | 3–6 blocks | 0 | Short, flat |
| 2 | 5–9 blocks | 1 | Medium, 1 obstacle |
| 3 | 7–14 blocks | 2 | Long, 2 obstacles (= default training) |
| 4 | 10–18 blocks | 3 | Very long, 3 obstacles |

Advancement: rolling success rate ≥ 0.70 over the last 20 episodes.

---

## Notes

- Checkpoints are saved to `python_rl/checkpoints/`
- Logs (monitor CSV, success CSV) are saved to `python_rl/logs/`
- TensorBoard logs: `python_rl/logs/tb/`  →  `tensorboard --logdir python_rl/logs/tb`
- All Python modules use `python -m python_rl.train.<script>` style to ensure
  correct package resolution.
