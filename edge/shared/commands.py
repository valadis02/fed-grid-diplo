from abc import ABC, abstractmethod
from typing import Dict, Any


class Command(ABC):
    @abstractmethod
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        pass

