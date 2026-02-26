import json
import re
import subprocess
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
from shared.logging_config import logger


class FederatedNodeType(Enum):
    CLOUD_NODE = 0
    FOG_NODE = 1
    EDGE_NODE = 2


def find_topology_file(folder: str, name_substring: str) -> str:
    """
    Search for the newest JSON file in `folder` whose name contains `name_substring`.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"{folder} is not a valid folder")

    matches = [f for f in folder_path.iterdir()
               if f.is_file() and name_substring in f.name and f.suffix.lower() == ".json"]

    if not matches:
        raise FileNotFoundError(f"No file found in {folder} containing '{name_substring}'")

    # sort by modification time (descending) and take newest
    newest_file = max(matches, key=lambda f: f.stat().st_mtime)
    return str(newest_file)

def load_topology(folder: str, name_substring: str) -> Dict[str, Any]:
    """
    Load topology JSON from the newest file matching the substring in the given folder.
    """
    file_path = find_topology_file(folder, name_substring)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def node_info(node: Dict[str, Any]) -> Dict[str, Any]:
    d = node["data"]
    return {
        "id": node["id"],
        "label": d.get("label"),
        "ip_address": d.get("ip_address"),
        "port": d.get("port"),
        "node_type": d.get("node_type"),
        "device_mac": d.get("device_mac"),
    }

def parse_topology_for_port(folder: str, name_substring: str, current_port: int) -> Dict[str, Any]:
    topo = load_topology(folder, name_substring)

    nodes: List[Dict[str, Any]] = topo.get("nodes", [])
    edges: List[Dict[str, Any]] = topo.get("edges", [])

    id_to_node = {n["id"]: n for n in nodes}
    port_to_node = {n["data"].get("port"): n for n in nodes}

    current = port_to_node.get(current_port)
    if current is None:
        raise ValueError(f"No node found with port {current_port}")

    current_id = current["id"]

    parent_ids = [e["source"] for e in edges if e.get("target") == current_id]
    child_ids = [e["target"] for e in edges if e.get("source") == current_id]

    parents = [node_info(id_to_node[p]) for p in parent_ids if p in id_to_node]
    children = [node_info(id_to_node[c]) for c in child_ids if c in id_to_node]

    return {
        "current_node": node_info(current),
        "parent": parents[0] if parents else None,
        "parents": parents,
        "children": children
    }

class FederatedNode:
    def __init__(self, name: str, federated_node_type: FederatedNodeType, ip_address: str, port: int,
                 device_mac: str = None) -> None:
        self.name = name
        self.federated_node_type = federated_node_type
        self.ip_address = ip_address
        self.port = port
        self.parent_node: Optional['ParentFederatedNode'] = None
        self.child_nodes: List[Optional['ChildFederatedNode']] = []
        self.device_mac = MacFetcher.get_mac_address() if device_mac is None else device_mac

    def set_parent_node(self, parent_node: Optional['ParentFederatedNode']):
        self.parent_node = parent_node

    def add_child_node(self, child_node: Optional['ChildFederatedNode']):
        self.child_nodes.append(child_node)

    def add_child_nodes(self, child_nodes: List[Optional['ChildFederatedNode']]):
        self.child_nodes.extend(child_nodes)


class ParentFederatedNode(FederatedNode):
    def __init__(self, name: str, federated_node_type: FederatedNodeType, ip_address: str, port: int, device_mac: str) -> None:
        super().__init__(name, federated_node_type, ip_address, port, device_mac)


class ChildFederatedNode(FederatedNode):
    def __init__(self, name: str, federated_node_type: FederatedNodeType, ip_address: str, port: int, device_mac: str) -> None:
        super().__init__(name, federated_node_type, ip_address, port, device_mac)

class MacFetcher:
    HOST_SYS = "/host_sys"
    CLASS_NET = os.path.join(HOST_SYS, "class", "net")
    _MAC_RE = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$", re.I)

    @classmethod
    def get_mac_address(cls) -> str:
        logger.debug("Starting MAC address retrieval...")
        # 1) Prefer explicit host interface
        host_iface = os.getenv("HOST_IFACE")
        if host_iface:
            logger.debug(f"HOST_IFACE is set to '{host_iface}', trying it first...")
            mac = cls._read_host_mac_by_name(host_iface)
            if mac:
                logger.debug(f"MAC found from HOST_IFACE '{host_iface}': {mac}")
                return mac
            else:
                logger.debug(f"No MAC found for HOST_IFACE '{host_iface}'")

        # 2) Auto-pick a "real" interface from the host
        candidates = cls._candidate_ifaces()
        logger.debug(f"Candidate interfaces after filtering: {candidates}")
        for iface in candidates:
            logger.debug(f"Trying candidate interface '{iface}'...")
            mac = cls._read_host_mac_by_name(iface)
            if mac:
                logger.debug(f"MAC found from candidate '{iface}': {mac}")
                return mac
            else:
                logger.debug(f"No valid MAC found for candidate '{iface}'")

        logger.warning("No valid MAC found, returning all-zero MAC.")
        return "00:00:00:00:00:00"

    @classmethod
    def _candidate_ifaces(cls):
        try:
            names = [n for n in os.listdir(cls.CLASS_NET)]
            logger.debug(f"Interfaces in {cls.CLASS_NET}: {names}")
        except Exception as e:
            logger.error(f"Failed to list interfaces in {cls.CLASS_NET}: {e}")
            return []

        skip_prefixes = (
            "lo", "veth", "br-", "docker", "ifb", "virbr", "vlan", "bond", "macvtap", "tap", "tun"
        )
        names = [n for n in names if not any(n.startswith(p) for p in skip_prefixes)]
        logger.debug(f"Interfaces after skip filter: {names}")

        def rank(name: str) -> int:
            if name.startswith(("enp", "eth", "eno", "ens")): return 0
            if name.startswith(("wlan", "wlp", "enx")):      return 1
            return 2

        sorted_names = sorted(names, key=lambda n: (rank(n), n))
        logger.debug(f"Interfaces after sorting: {sorted_names}")
        return sorted_names

    @classmethod
    def _read_host_mac_by_name(cls, iface: str) -> str | None:
        base = os.path.join(cls.CLASS_NET, iface)
        addr_path = os.path.join(base, "address")

        try:
            if os.path.islink(base):
                link = os.readlink(base)
                logger.debug(f"Interface '{iface}' is a symlink to '{link}'")
                real_dir = os.path.normpath(os.path.join(cls.CLASS_NET, link))
                addr_path = os.path.join(real_dir, "address")

            logger.debug(f"Reading MAC from {addr_path}")
            with open(addr_path, "r") as f:
                mac = f.read().strip().lower()
                logger.debug(f"Read MAC '{mac}' from {iface}")
                if cls._is_valid_mac(mac):
                    return mac
                else:
                    logger.debug(f"MAC '{mac}' is invalid for {iface}")
        except Exception as e:
            logger.debug(f"Failed to read MAC for interface '{iface}': {e}")
        return None

    @classmethod
    def _is_valid_mac(cls, mac: str) -> bool:
        if not mac or not cls._MAC_RE.fullmatch(mac):
            return False
        if mac == "00:00:00:00:00:00":
            return False
        return True
