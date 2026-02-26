from typing import Optional
from shared.node import FederatedNode
from shared.logging_config import logger


class FederatedNodeState:
    __current_node: Optional['FederatedNode'] = None

    @classmethod
    def initialize_node(cls, node: FederatedNode) -> None:
        if cls.__current_node is not None:
            raise logger.warning("Current working node is already initialized!")
        cls.__current_node = node

    @classmethod
    def get_current_node(cls) -> Optional['FederatedNode']:
        return cls.__current_node

    @classmethod
    def reset_node(cls) -> None:
        cls.__current_node = None
