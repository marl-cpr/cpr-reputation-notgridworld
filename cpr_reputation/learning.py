from typing import Dict, Tuple

from cpr_reputation.games import OneResourceWLogisticReplenishment
from cpr_reputation.typing import Obs, Stepped

import numpy as np
from ray.rllib.env import MultiAgentEnv


class OneResourceWLogisticReplenishmentEnv(MultiAgentEnv):
    def __init__(self, config, **kwargs):
        self.config = config
        self.time = 0
        self.game = OneResourceWLogisticReplenishment(**kwargs)

    def reset(self) -> Dict[str, Obs["2,1", float]]:
        self.game.resource.amount = self.game.resource.starting_amount
        self.game.reputation = {
            appropriator_id: 0. for appropriator_id in self.game.appropriator_names
        }
        self.time = 0
        return self._get_obs()

    def step(self, actions: Dict[str, float]) -> Stepped:
        rewards = {
            appropriator_id: 0.0
            for appropriator_id, _
            in self.game.appropriators.items()}
        for appropriator_id, action in actions.items():
            reward = self.game.process_action(appropriator_id, action)
            rewards[appropriator_id] += reward

        amount = self.game.resource.amount
        obs = self._get_obs()

        is_done = self.time > 1000
        done = {appropriator_id: is_done for appropriator_id, _ in self.game.appropriators.items()}
        done["__all__"] = is_done  # Required for rllib (I think)
        if is_done:
            for appropriator_id, _ in self.game.appropriators.items():
                rewards[appropriator_id] += self.game.reputation[appropriator_id]

        info = dict()
        self.time += 1

        return obs, rewards, done, info

    def _get_obs(self) -> Dict[str, Obs["2,1", float]]:
        return {
            appropriator_id: np.array([
                [self.game.resource.amount, self.game.reputation[appropriator_id]]
            ])
            for appropriator_id
            in self.game.appropriator_names
        }
