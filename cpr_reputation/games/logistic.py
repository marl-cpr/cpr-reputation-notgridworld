from typing import Sequence
from cpr_reputation.abstract import CPRProblem, Resource

class LogisticReplenishment(Resource):
    def __init__(
            self,
            name: str,
            amount: float,
            rate: float,
            maximum: float,
    ):
        """
        amount: in [0,1], fraction of maximum
        rate: in [0,1]
        maximum: any positive float
        """
        super().__init__(name, amount)
        self.maximum = maximum
        self.amount = amount * self.maximum
        self.starting_amount = amount  # for resets
        self.rate = rate / maximum  # rate of growth

    @property
    def increment(self) -> float:
        """dx/dt = r * x * (maximum - x)"""
        return self.rate * self.amount * (self.maximum - self.amount)

class OneResourceWLogisticReplenishment(CPRProblem):
    def __init__(
            self,
            name: str,
            resource: Resource,
            num_appropriators: int
    ):
        super().__init__(name, resource, num_appropriators)
        assert hasattr(resource, "maximum")
        assert not isinstance(resource, Sequence)

    def calculate_reputation(self, action: float) -> float:
        assert self.resource.maximum > self.resource.amount + action
        return self.resource.maximum - self.resource.amount - action

    def process_action(self, appropriator_id: str, action: float) -> float:
        """You MUST call self.game.resource.replenish_step in MultiAgentEnv.step implementation"""
        action = action[0]
        if action >= self.resource.amount:
            self.reputation[appropriator_id] -= 1e5
            return self.resource.amount

        self.reputation[appropriator_id] -= self.calculate_reputation(action)
        self.resource.amount -= action
        return action
