# Minecraft RL Multi-Skill NPC

End-to-end reinforcement learning inside **Minecraft 1.20.1** using a custom
Forge mod and **Stable-Baselines3 PPO**.  A controllable NPC learns navigation,
farming, and combat from scratch through a shared policy.

See [`notes/README.md`](notes/README.md) for the full project documentation,
architecture diagram, action/observation tables, and experiment guide.

---

## Quick install

```bash
# 1. Clone
git clone <repo-url>
cd rl_minecraft_npc

# 2. Install Python package (editable)
pip install -e .

# 3. Or install deps only
pip install -r requirements.txt
```

## Quick start

### Run Minecraft
Open the project in IntelliJ and launch the `runClient` configuration.
Create or open a **flat world** and wait for:
```
[rlnpc] RL environment ready. HTTP bridge listening on http://127.0.0.1:8765
```

### Train
```bash
python -m python_rl.train.train_navigation          # shaped-reward baseline
python -m python_rl.train.train_nav_sparse          # sparse comparison
python -m python_rl.train.train_nav_curriculum      # 4-level curriculum
python -m python_rl.train.train_farming             # single-task farming
python -m python_rl.train.train_combat              # combat task
python -m python_rl.train.train_multitask           # shared policy (all tasks in one episode)
```

### Evaluate
```bash
python -m python_rl.eval.evaluate --model nav_shaped_run1
python -m python_rl.eval.evaluate_farming --model farm_run1
python -m python_rl.eval.evaluate_multitask --model multitask_run1
python -m python_rl.eval.generalization_test --model nav_curriculum_run1
python -m python_rl.eval.compare_experiments
```

## Repository layout

```
rl_minecraft_npc/
├── minecraft_mod/        Forge 1.20.1 mod (Java 17)
├── python_rl/
│   ├── env/              Gymnasium wrapper
│   ├── train/            PPO training scripts + curriculum
│   ├── eval/             Evaluation + plotting scripts
│   └── configs/          YAML experiment configs
└── notes/                Full documentation
```