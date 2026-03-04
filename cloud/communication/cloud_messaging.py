import base64
import hashlib
import json
import os
import threading
import time
from tempfile import NamedTemporaryFile
from typing import Dict
import socket
import random
import pika
import paho.mqtt.client as mqtt
from shared.logging_config import logger
from shared.node_state import FederatedNodeState
from shared.crypto import decrypt_from_b64
from cloud.model.model_aggregation_service import aggregate_received_models
from cloud.model.cloud_evaluation_service import evaluate_global_model
from cloud.communication.cloud_resources_paths import CloudResourcesPaths

# ---------------------------------------------------------------------------
# Minimal stand-in when the full topology stack isn't initialised (local test)
# ---------------------------------------------------------------------------
class _SimpleNode:
    """Fallback node used when FederatedNodeState has no current node."""
    def __init__(self, child_names):
        self.id = "CLOUD_NODE"
        self.child_nodes = [type("N", (), {"name": n})() for n in child_names]

FALLBACK_FOG_NAMES = os.getenv("FALLBACK_FOG_NAMES", "FOG_NODE_1").split(",")


def _get_node():
    node = FederatedNodeState.get_current_node()
    if node is None:
        return _SimpleNode(FALLBACK_FOG_NAMES)
    return node


class CloudMessaging:
    def __init__(
            self,
            cloud_amqp_host: str = os.getenv('CLOUD_RABBITMQ_HOST', 'rabbitmq-cloud'),
            cloud_mqtt_host: str = os.getenv('CLOUD_MQTT_HOST', 'mqtt-cloud'),
            cloud_mqtt_port: int = int(os.getenv('CLOUD_MQTT_PORT', 1883)),
    ):
        self.cloud_amqp_host = cloud_amqp_host
        self.cloud_mqtt_host = cloud_mqtt_host
        self.cloud_mqtt_port = cloud_mqtt_port
        self.fog_models_cache: Dict[str, dict] = {}
        self._recent_fog_models: Dict[str, dict] = {}
        self.round_id = None
        self._round_counter = 0

        try:
            os.makedirs(CloudResourcesPaths.STATUS_FOLDER_PATH.value, exist_ok=True)
        except Exception as e:
            logger.warning("Cloud: failed to ensure status folder: %s", e)

        self._restore_round_id()

    # ------------------------------------------------------------------
    # Round-id persistence
    # ------------------------------------------------------------------

    def _persist_round_id(self, round_id):
        path = CloudResourcesPaths.ROUND_FILE_PATH.value
        self.round_id = round_id
        try:
            data = {"round_id": round_id, "ts": int(time.time())}
            with NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False, suffix=".tmp") as tf:
                json.dump(data, tf)
                tf.flush()
                os.fsync(tf.fileno())
                tmp = tf.name
            os.replace(tmp, path)
            logger.debug("Cloud: persisted round_id=%s", round_id)
        except Exception as e:
            logger.warning("Cloud: failed to persist round_id: %s", e)

    def _restore_round_id(self):
        path = CloudResourcesPaths.ROUND_FILE_PATH.value
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                rid = data.get("round_id")
                if rid is not None:
                    self.round_id = rid
                    logger.info("Cloud: restored round_id=%s from disk", rid)
        except Exception as e:
            logger.warning("Cloud: failed to restore round_id: %s", e)

    # ------------------------------------------------------------------
    # AMQP helpers
    # ------------------------------------------------------------------

    def _create_connection(self, retries=10, delay=5) -> pika.BlockingConnection:
        """Connect to the Cloud RabbitMQ broker (internal port 5672)."""
        for attempt in range(1, retries + 1):
            try:
                return pika.BlockingConnection(pika.ConnectionParameters(
                    host=self.cloud_amqp_host,
                    port=5672,
                    heartbeat=30,
                    blocked_connection_timeout=60,
                    connection_attempts=1,
                    retry_delay=0,
                ))
            except (socket.gaierror, pika.exceptions.AMQPError) as e:
                if attempt == 1:
                    logger.warning("Cloud: AMQP unavailable (%s). Retrying up to %d times…",
                                   e.__class__.__name__, retries)
                if attempt == retries:
                    logger.error("Cloud: could not reach RabbitMQ after %d attempts.", retries)
                    raise
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, 60)

    def _publish_to_fog_queues(self, channel, message_body: bytes):
        """Publish directly to each per-fog durable queue."""
        node = _get_node()
        fogs = getattr(node, "child_nodes", []) or []
        if not fogs:
            logger.warning("Cloud: no fog nodes found; nothing to broadcast.")
            return
        for fog in fogs:
            q = f"cloud_fanout_for_{fog.name}"
            channel.queue_declare(queue=q, durable=True, auto_delete=False)
            channel.basic_publish(
                exchange='',
                routing_key=q,
                body=message_body,
                properties=pika.BasicProperties(delivery_mode=2),
                mandatory=True,
            )
            logger.info("Cloud: enqueued model for fog '%s' → queue '%s'.", fog.name, q)

    # ------------------------------------------------------------------
    # MQTT helpers
    # ------------------------------------------------------------------

    def _mqtt_publish(self, topic: str, message: dict, qos: int = 1,
                      retain: bool = True, retries=10, delay=5):
        client = mqtt.Client(client_id="cloud-publisher", clean_session=True)
        for attempt in range(1, retries + 1):
            try:
                client.connect(self.cloud_mqtt_host, self.cloud_mqtt_port)
                break
            except Exception as e:
                logger.warning("Cloud MQTT connection failed (%d/%d): %s", attempt, retries, e)
                if attempt == retries:
                    logger.error("Cloud: could not connect to MQTT broker after %d retries.", retries)
                    return
                time.sleep(delay)
        client.publish(topic, json.dumps(message), qos=qos, retain=retain)
        client.disconnect()

    # ------------------------------------------------------------------
    # Control-plane (MQTT)
    # ------------------------------------------------------------------

    def notify_all_edges_to_create_local_model(self):
        self._mqtt_publish(topic='cloud/fog/command', message={'command': '0'})
        logger.info("Cloud (MQTT): sent command '0' → edges create local model.")

    def notify_all_edges_to_start_first_training(self, data: Dict[str, any]):
        round_id = data.get("round_id", int(time.time() * 1000))
        self._persist_round_id(round_id)
        cmd = {'command': '1', 'cmd_id': int(time.time() * 1000),
               'round_id': round_id, 'data': data}
        self._mqtt_publish('cloud/fog/command', cmd, qos=1, retain=False)
        logger.info("Cloud (MQTT): sent command '1' → edges start training (round=%s).", round_id)

    # ------------------------------------------------------------------
    # Model broadcast (AMQP) — called automatically after aggregation
    # ------------------------------------------------------------------

    def broadcast_cloud_model(self, data: Dict[str, any] = None):
        """
        Broadcast aggregated cloud model to all fogs via per-fog durable queues.
        Also sends command '2' via MQTT so fogs/edges know a new round is starting.
        """
        if data is None:
            data = {}

        round_id = data.get("round_id", self.round_id or int(time.time() * 1000))
        self._persist_round_id(round_id)

        model_path = CloudResourcesPaths.CLOUD_MODEL_FILE_PATH.value
        if not os.path.exists(model_path):
            logger.error("Cloud: no aggregated model at %s — cannot broadcast.", model_path)
            return

        with open(model_path, "rb") as f:
            model_b64 = base64.b64encode(f.read()).decode('utf-8')

        message = {
            "command": "2",
            "round_id": round_id,
            "model": model_b64,
            "data": data,
        }
        message_body = json.dumps(message).encode('utf-8')

        try:
            conn = self._create_connection()
            ch = conn.channel()
            self._publish_to_fog_queues(ch, message_body)
            conn.close()
            logger.info("Cloud (AMQP): broadcast cloud model to fogs (round=%s).", round_id)
        except Exception as e:
            logger.exception("Cloud: AMQP broadcast failed: %s", e)

        mqtt_cmd = {"command": "2", "round_id": round_id, "cmd_id": int(time.time() * 1000)}
        self._mqtt_publish('cloud/fog/command', mqtt_cmd, qos=1, retain=False)
        logger.info("Cloud (MQTT): sent command '2' → edges re-train with new model (round=%s).", round_id)

    # ------------------------------------------------------------------
    # Aggregation gate
    # ------------------------------------------------------------------

    def is_ready_to_aggregate(self):
        node = _get_node()
        num_expected = len(node.child_nodes) if node else 1

        logger.info("Cloud: received %d/%d fog models.", len(self.fog_models_cache), num_expected)

        if len(self.fog_models_cache) >= num_expected:
            logger.info("Cloud: all fog models received — starting aggregation.")
            try:
                aggregate_received_models(self.fog_models_cache)
            except Exception as e:
                logger.exception("Cloud: aggregation failed: %s", e)
                return False

            self._round_counter += 1
            current_round = self._round_counter
            threading.Thread(
                target=evaluate_global_model,
                args=(current_round,),
                daemon=True,
            ).start()

            self.fog_models_cache.clear()
            logger.info("Cloud: aggregation complete — broadcasting updated model to fogs.")

            threading.Thread(
                target=self.broadcast_cloud_model,
                kwargs={"data": {"round_id": self.round_id}},
                daemon=True,
            ).start()
            return True
        return False

    # ------------------------------------------------------------------
    # AMQP listener — receive fog models (AES-256-GCM decryption)
    # ------------------------------------------------------------------

    def start_amqp_listener(self):
        """
        Blocking loop: consume 'fog_to_cloud_models' queue.
        Αποκρυπτογραφεί τα βάρη με AES-256-GCM (Experiment 4).
        """
        os.makedirs(CloudResourcesPaths.MODELS_FOLDER_PATH.value, exist_ok=True)

        delay, max_delay = 5, 60
        announced_down = False

        while True:
            conn = None
            try:
                conn = self._create_connection()
                ch = conn.channel()
                ch.basic_qos(prefetch_count=1)
                ch.queue_declare(queue='fog_to_cloud_models', durable=True)
                logger.info("Cloud: listening on queue 'fog_to_cloud_models'…")

                if announced_down:
                    logger.info("Cloud: AMQP back online.")
                    announced_down = False
                    delay = 5

                recent_ids: Dict[str, float] = {}
                MSG_TTL = 900

                def _purge_old(now):
                    old = [k for k, t in recent_ids.items() if now - t > MSG_TTL]
                    for k in old:
                        recent_ids.pop(k, None)

                def on_fog_model(ch, method, properties, body):
                    now = time.time()
                    _purge_old(now)

                    msg_id = getattr(properties, "message_id", None)
                    if msg_id:
                        if msg_id in recent_ids:
                            logger.debug("Cloud: duplicate msg_id %s; skipping.", msg_id)
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            return
                        recent_ids[msg_id] = now

                    try:
                        payload = json.loads(body)
                    except Exception:
                        logger.warning("Cloud: non-JSON payload; discarding.")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    fog_name      = payload.get('fog_name')
                    fog_mac       = payload.get('fog_device_mac', '00:00:00:00:00:00')
                    model_b64     = payload.get('model')
                    model_hash    = payload.get('hash')
                    recv_round_id = payload.get('round_id')
                    encrypted     = payload.get('encrypted', False)

                    if self.round_id is not None and recv_round_id is not None \
                            and recv_round_id != self.round_id:
                        logger.warning(
                            "Cloud: round-id mismatch (cloud=%s, fog=%s); ignoring.",
                            self.round_id, recv_round_id)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    if self.round_id is None and recv_round_id is not None:
                        logger.info("Cloud: adopting round_id=%s from Fog.", recv_round_id)
                        self._persist_round_id(recv_round_id)

                    if not fog_name or not model_b64:
                        logger.warning("Cloud: malformed payload (missing fog_name or model).")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    # Content-hash de-dupe (πάνω στο encrypted payload)
                    key = f"{fog_mac}:{fog_name}"
                    if not model_hash:
                        try:
                            model_hash = hashlib.sha256(base64.b64decode(model_b64)).hexdigest()
                        except Exception:
                            logger.warning("Cloud: invalid base64 model from fog %s.", fog_name)
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            return

                    prev = self._recent_fog_models.get(key)
                    if prev and prev["hash"] == model_hash and (now - prev["ts"] < MSG_TTL):
                        logger.debug("Cloud: duplicate content hash from %s; skipping.", fog_name)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    # ── ΑΠΟΚΡΥΠΤΟΓΡΑΦΗΣΗ (Experiment 4) ──────────────────────────
                    try:
                        if encrypted:
                            model_bytes, dec_metrics = decrypt_from_b64(model_b64)
                            logger.info(
                                "Cloud: decrypted model from fog %s | "
                                "decrypt=%.4fs | size=%.1fKB",
                                fog_name,
                                dec_metrics['decrypt_time_s'],
                                dec_metrics['plaintext_size_b'] / 1024,
                            )
                        else:
                            model_bytes = base64.b64decode(model_b64)
                    except Exception as e:
                        logger.error("Cloud: decryption failed for fog %s: %s", fog_name, e)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                    # ─────────────────────────────────────────────────────────────

                    try:
                        model_path = os.path.join(
                            CloudResourcesPaths.MODELS_FOLDER_PATH.value,
                            f'{fog_name}_aggregated_model.keras'
                        )
                        with open(model_path, 'wb') as f:
                            f.write(model_bytes)
                            f.flush()
                            os.fsync(f.fileno())

                        map_id = f"{fog_mac}_{fog_name}"
                        self.fog_models_cache[map_id] = {"model_path": model_path}
                        self._recent_fog_models[key] = {"hash": model_hash, "ts": now}
                        logger.info("Cloud: cached model from fog '%s' (round=%s).",
                                    fog_name, recv_round_id)
                    except Exception as e:
                        logger.error("Cloud: failed to save model from fog %s: %s", fog_name, e)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    self.is_ready_to_aggregate()

                ch.basic_consume(
                    queue='fog_to_cloud_models',
                    on_message_callback=on_fog_model,
                    auto_ack=False,
                )
                ch.start_consuming()

            except (socket.gaierror, pika.exceptions.AMQPError) as e:
                if not announced_down:
                    logger.warning("Cloud: AMQP unavailable (%s). Backing off…", e.__class__.__name__)
                    announced_down = True
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, max_delay)
            except Exception:
                logger.exception("Cloud: unexpected error in AMQP listener; retrying in %ss…", delay)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            finally:
                try:
                    if conn and conn.is_open:
                        conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # MQTT listener (stub)
    # ------------------------------------------------------------------

    def start_mqtt_listener(self):
        logger.info("Cloud: MQTT listener started (no-op in current setup).")
        while True:
            time.sleep(60)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if logging_root not in sys.path:
        sys.path.insert(0, logging_root)

    cloud_msg = CloudMessaging()
    logger.info("Cloud: Starting AMQP listener (standalone mode)…")
    cloud_msg.start_amqp_listener()