from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, Union, List

@dataclass
class Resource(ABC):
    name: str
    amount: float  # starting amount

    @abstractmethod
    def increment(self) -> float:
        """You must decorate implementation @property."""
        pass

    def replenish_step(self):
        self.amount += self.increment


@dataclass
class Appropriator:
    name: str
    amount: float = 0



class CPRProblem(ABC):
    """Common Pool Resource Problem with Reputation"""
    def __init__(
            self,
            name: str,
            resource: Union[Resource, Sequence[Resource]],
            num_appropriators: int
    ):
        self.name = name
        self.resource = resource
        self.num_appropriators = num_appropriators
        self.appropriator_names = [f"Appropriator{k}" for k in range(num_appropriators)]
        self.appropriators = {
            appropriator_id: Appropriator(appropriator_id)
            for appropriator_id
            in self.appropriator_names
        }
        self.reputation = {appropriator_id: 0. for appropriator_id in self.appropriator_names}

    @abstractmethod
    def process_action(self, appropriator_id: str, action: Union[float, Sequence[float]]) -> float:
        """Processes action, returns reward"""
        pass
