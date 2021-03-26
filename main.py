#!/usr/bin/env python3

from cpr_reputation.games import LogisticReplenishment
from cpr_reputation.learning import OneResourceWLogisticReplenishmentEnv

import numpy as np
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from gym.spaces import Box

logistic_defaults_ini = {
    "name": "fish",
    "amount": 0.55,
    "rate": 0.24,
    "maximum": 100
}
defaults_ini = {
    "name": "One Resource w Logistic Replenishment",
    "resource": LogisticReplenishment(**logistic_defaults_ini),
    "num_appropriators": 7
}

if __name__ == "__main__":
    register_env(
        "1resource-logisticreplenishment",
        lambda config: OneResourceWLogisticReplenishmentEnv(config, **defaults_ini)
    )

    # tune.run("harvest", {"framework": "torch"})
    appropriator1 = (
        None,
        Box(
            - float("inf"),
            float("inf"),
            (1, 2),
            np.float32
        ),  # obs
        Box(0, logistic_defaults_ini["maximum"], (1,), np.float32),  # action
        dict(),
    )
    config = {
        "multiagent": {
            "policies": {"appropriator1": appropriator1},
            "policy_mapping_fn": lambda agent_id: "appropriator1",
        },
        "framework": "torch",
        "model": {
            "dim": 0,
            "conv_filters": [
                [16, [1, 1], 1],
                [32, [1, 2], 1],
            ]
        },
    }
    ray.init()
    trainer = ppo.PPOTrainer(env="1resource-logisticreplenishment", config=config)

    while True:
        print(trainer.train())
