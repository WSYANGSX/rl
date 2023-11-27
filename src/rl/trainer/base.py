from abc import ABC, abstractmethod
from rl.policy import BasePolicy
from typing import Any
from rl.data.collector import Collector


class Trainer(ABC):
    def __init__(self, collector: Collector, eposide: int = 10000) -> None:
        self._eposide = eposide
        self._collector = collector

    @classmethod
    def train(self) -> None:
        """
        对策略进行训练
        """
        pass
