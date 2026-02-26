from __future__ import annotations
import os
from typing import Dict, Any, Union
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from shared.node import FederatedNode, FederatedNodeType, ParentFederatedNode, ChildFederatedNode, parse_topology_for_port
from shared.node_state import FederatedNodeState
from shared.logging_config import logger
from shared.commands import Command

shared_router = APIRouter()


class InitializeNodeCommand(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return initialize_node(data)


class GetNodeInfo(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return get_node_info()


class SetParentNode(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return set_parent_node(data)


class SetChildNode(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return set_child_node(data)


class SetChildrenNodes(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return set_children_nodes(data)

class AutoInitialization(Command):
    def execute(self, data: Dict[str, any]) -> Dict[str, Any]:
        return auto_initialization()

command_map = {
    0: InitializeNodeCommand(),
    1: GetNodeInfo(),
    2: SetParentNode(),
    3: SetChildNode(),
    4: SetChildrenNodes(),
    5: AutoInitialization(),
}


@shared_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            operation: int = message.get('operation')
            data: Dict[str, any] = message.get('data', {})

            try:
                command = command_map.get(operation)
                if command:
                    response = command.execute(data)
                else:
                    response = {"Error": f"Invalid Operation {command}."}
            except Exception as e:
                response = {"error": str(e)}

            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.warning("Websocket disconnected...")


def initialize_node(data: Dict[str, any]) -> Dict[str, any]:
    if FederatedNodeState.get_current_node() is not None:
        raise ValueError("Current node is already initialized!")

    node = FederatedNode(data.get("name"), data.get("federated_node_type"), data.get("ip_address"), data.get("port"))
    FederatedNodeState.initialize_node(node)

    logger.info(f"Successfully created a new node with name {node.name}, type {node.federated_node_type}, "
                f"ip {node.ip_address}, port {node.port} and device mac {node.device_mac}.")

    return {
        "message": "Node initialized successfully.",
        "node": {
            "name": node.name,
            "federated_node_type": node.federated_node_type,
            "ip_address": node.ip_address,
            "port": node.port,
            "device_address": node.device_mac
        }
    }


def get_node_info() -> Dict[str, Union[str, Dict[str, Union[str, FederatedNodeType]]]]:
    node = FederatedNodeState.get_current_node()
    if node is None:
        raise ValueError("Current node is not initialized!")

    return {
        "message": "Node is present.",
        "node": {
            "name": node.name,
            "federated_node_type": node.federated_node_type,
            "ip_address": node.ip_address,
            "port": node.port,
            "device_address": node.device_mac
        }
    }


def set_parent_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FederatedNodeType]]]]:
    current_node = FederatedNodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet!")

    parent_node = ParentFederatedNode(data.get("name"), data.get("federated_node_type"), data.get("ip_address"),
                                      data.get("port"), data.get("device_mac"))
    current_node.set_parent_node(parent_node)

    return {
        "message": "Parent node of the current working node set successfully.",
        "node": {
            "name": parent_node.name,
            "federated_node_type": parent_node.federated_node_type,
            "ip_address": parent_node.ip_address,
            "port": parent_node.port,
            "device_address": parent_node.device_mac
        }
    }


def set_child_node(data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, Union[str, FederatedNodeType]]]]:
    current_node = FederatedNodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized!")
    child_node = ChildFederatedNode(data.get("name"), data.get("federated_node_type"), data.get("ip_address"),
                                    data.get("port"), data.get("device_port"))
    current_node.add_child_node(child_node)

    return {
        "message": "A child node of the current working node set successfully.",
        "node": {
            "name": child_node.name,
            "federated_node_type": child_node.federated_node_type,
            "ip_address": child_node.ip_address,
            "port": child_node.port,
            "device_address": child_node.device_mac
        }
    }


def set_children_nodes(data) -> dict[str, str | list[dict[str, str | FederatedNodeType]]]:
    current_node = FederatedNodeState.get_current_node()

    if current_node is None:
        raise ValueError("Current node was not initialized yet!")

    child_nodes = [
        ChildFederatedNode(ind.get("name"), ind.get("federated_node_type"), ind.get("ip_address"),
                           ind.get("port"), ind.get("device_port"))
        for ind in data
    ]

    current_node.add_child_nodes(child_nodes)

    added_children = [
        {
            "name": child.name,
            "federated_node_type": child.federated_node_type,
            "ip_address": child.ip_address,
            "port": child.port,
            "device_mac": child.device_mac
        }
        for child in child_nodes
    ]

    return {
        "message": f"Added {len(added_children)} child nodes to the current working node.",
        "children": added_children,
    }

def _coerce_node_type(x):
    # Accept int or Enum or str; normalize to FederatedNodeType
    if isinstance(x, FederatedNodeType):
        return x
    if isinstance(x, int):
        # Your topology appears to be 1-based: cloud=1, fog=2, edge=3
        if x in (1, 2, 3):
            x = x - 1
        return FederatedNodeType(x)  # will raise if out of range
    if isinstance(x, str):
        # allow "cloud" / "fog" / "edge" or enum names
        s = x.strip().upper()
        alias = {"CLOUD": 0, "FOG": 1, "EDGE": 2,
                 "CLOUD_NODE": 0, "FOG_NODE": 1, "EDGE_NODE": 2}
        return FederatedNodeType(alias[s])
    raise ValueError(f"Unsupported node_type: {x!r}")

def auto_initialization():
    port_str = os.getenv("APP_HOST_PORT")
    logger.info(f"Current running port: {port_str}")
    if not port_str:
        return {"error": "APP_HOST_PORT not set in container environment."}

    try:
        current_port = int(port_str)
    except ValueError:
        return {"error": f"APP_HOST_PORT='{port_str}' is not a valid integer."}

    info = parse_topology_for_port("/app", "federated_topology", current_port)
    logger.info(f"Node info: {info}")

    # Current node
    cur = info["current_node"]
    name = cur.get("label") or cur.get("name")
    ip = cur.get("ip_address")
    port = cur.get("port")
    mac = cur.get("device_mac")
    node_type = _coerce_node_type(cur.get("node_type"))

    current_node = FederatedNode(name, node_type, ip, port, mac)

    # Parent (optional)
    parent = info.get("parent")
    if parent:
        p = ParentFederatedNode(
            parent.get("label") or parent.get("name"),
            _coerce_node_type(parent.get("node_type")),
            parent.get("ip_address"),
            parent.get("port"),
            parent.get("device_mac"),
        )
        current_node.set_parent_node(p)

    # Children
    children_objs = []
    for ch in info.get("children", []):
        c = ChildFederatedNode(
            ch.get("label") or ch.get("name"),
            _coerce_node_type(ch.get("node_type")),
            ch.get("ip_address"),
            ch.get("port"),
            ch.get("device_mac"),
        )
        children_objs.append(c)
    current_node.add_child_nodes(children_objs)

    FederatedNodeState.initialize_node(current_node)
    logger.info(
        "auto_initialization: initialized node=%s type=%s port=%s parent=%s children=%d",
        name, node_type, port, (parent.get("label") if parent else None), len(children_objs)
    )

    # Return JSON-serializable data
    return {
        "message": "Auto-initialized node, parent, and children from topology.",
        "node": {
            "name": name,
            "node_type": node_type.value,   # or node_type.name
            "ip": ip,
            "port": port,
            "device_mac": mac
        }
    }



