from setuptools import setup, find_packages

setup(
    name="rl_minecraft_npc",
    version="1.0.0",
    description="Reinforcement learning for a multi-skill NPC inside Minecraft 1.20.1 via a custom Forge mod.",
    author="Simo",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "stable-baselines3>=2.3.0",
        "sb3-contrib>=2.3.0",
        "gymnasium>=0.29.0",
        "torch>=2.1.0",
        "tensorboard>=2.14.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
