from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Protocol

import numpy as np


@dataclass
class Resource(ABC):
    name: str
    amount: float  # starting amount

    @property
    @abstractmethod
    def increment(self) -> float:
        """You must decorate implementation @property."""
        pass

    def replenish_step(self):
        self.amount += self.increment


class ResourceWMax(Resource, Protocol):
    def maximum(self) -> int:
        return 0


@dataclass
class Appropriator:
    name: str
    amount: float = 0


class CPRProblem(ABC):
    """Common Pool Resource Problem with Reputation"""

    def __init__(
        self, name: str, resource: Sequence[Resource], num_appropriators: int,
    ):
        self.name = name
        self.resource = resource
        self.num_appropriators = num_appropriators
        self.appropriator_names = [f"Appropriator{k}" for k in range(num_appropriators)]
        self.appropriators = {
            appropriator_id: Appropriator(appropriator_id)
            for appropriator_id in self.appropriator_names
        }
        self.reputation = {
            appropriator_id: 0.0 for appropriator_id in self.appropriator_names
        }

    @abstractmethod
    def process_action(self, appropriator_id: str, action: np.ndarray) -> float:
        """Processes action, returns reward"""
        pass
