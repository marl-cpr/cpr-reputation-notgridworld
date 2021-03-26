from typing import Sequence

from cpr_reputation.abstract import CPRProblem, Resource


class LogisticReplenishment(Resource):
    def __init__(
        self, name: str, amount: float, rate: float, maximum: float,
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
    def __init__(self, name: str, resource: Sequence[Resource], num_appropriators: int):
        super().__init__(name, resource, num_appropriators)
        assert hasattr(resource[0], "maximum")

    def calculate_reputation(self, action: float) -> float:
        assert self.resource[0].maximum > self.resource[0].amount + action
        return self.resource[0].maximum - self.resource[0].amount - action

    def process_action(self, appropriator_id: str, action: float) -> float:
        """You MUST call self.game.resource.replenish_step in MultiAgentEnv.step implementation"""
        # action = action[0]
        if action >= self.resource[0].amount:
            amount = self.resource[0].amount
            self.reputation[appropriator_id] -= 2 * self.resource[0].maximum
            self.appropriators[appropriator_id].amount += amount
            return amount

        self.resource[0].amount -= action
        self.appropriators[appropriator_id].amount += action
        self.reputation[appropriator_id] -= self.calculate_reputation(action)
        return action
