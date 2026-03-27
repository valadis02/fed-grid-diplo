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

try:
    from cloud.model.model_compression_service import compress_model
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    logger.warning("Cloud: model_compression_service not found — compression DISABLED.")

QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "none").lower()


class _SimpleNode:
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
    # Model broadcast
    # ------------------------------------------------------------------

    def broadcast_cloud_model(self, data: Dict[str, any] = None):
        if data is None:
            data = {}

        round_id = data.get("round_id", self.round_id or int(time.time() * 1000))
        self._persist_round_id(round_id)

        model_path = CloudResourcesPaths.CLOUD_MODEL_FILE_PATH.value
        if not os.path.exists(model_path):
            logger.error("Cloud: no aggregated model at %s — cannot broadcast.", model_path)
            return

        model_format = "keras"
        if COMPRESSION_AVAILABLE and QUANTIZATION_MODE != "none":
            try:
                logger.info("Cloud: applying compression (mode=%s)…", QUANTIZATION_MODE)
                t_compress = time.perf_counter()
                model_bytes, model_format = compress_model(model_path)
                logger.info("Cloud: compression done | format=%s | size=%.1fKB | time=%.2fs",
                            model_format, len(model_bytes) / 1024,
                            round(time.perf_counter() - t_compress, 2))
            except Exception as e:
                logger.exception("Cloud: compression failed (%s) — sending uncompressed.", e)
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                model_format = "keras"
        else:
            with open(model_path, "rb") as f:
                model_bytes = f.read()

        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        message = {
            "command":      "2",
            "round_id":     round_id,
            "model":        model_b64,
            "model_format": model_format,
            "data":         data,
        }
        message_body = json.dumps(message).encode('utf-8')

        try:
            conn = self._create_connection()
            ch   = conn.channel()
            self._publish_to_fog_queues(ch, message_body)
            conn.close()
            logger.info("Cloud (AMQP): broadcast cloud model | round=%s | size=%.1fKB",
                        round_id, len(model_bytes) / 1024)
        except Exception as e:
            logger.exception("Cloud: AMQP broadcast failed: %s", e)

        mqtt_cmd = {"command": "2", "round_id": round_id, "cmd_id": int(time.time() * 1000)}
        self._mqtt_publish('cloud/fog/command', mqtt_cmd, qos=1, retain=False)
        logger.info("Cloud (MQTT): sent command '2' → edges re-train (round=%s).", round_id)

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
    # AMQP listener
    # ------------------------------------------------------------------

    def start_amqp_listener(self):
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
                    enc_mode      = payload.get('encryption_mode', 'aes')

                    if self.round_id is not None and recv_round_id is not None \
                            and recv_round_id != self.round_id:
                        logger.warning("Cloud: round-id mismatch (cloud=%s, fog=%s); ignoring.",
                                       self.round_id, recv_round_id)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    if self.round_id is None and recv_round_id is not None:
                        self._persist_round_id(recv_round_id)

                    if not fog_name or not model_b64:
                        logger.warning("Cloud: malformed payload.")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    key = f"{fog_mac}:{fog_name}"
                    if not model_hash:
                        try:
                            model_hash = hashlib.sha256(base64.b64decode(model_b64)).hexdigest()
                        except Exception:
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            return

                    prev = self._recent_fog_models.get(key)
                    if prev and prev["hash"] == model_hash and (now - prev["ts"] < MSG_TTL):
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    # ── Αποκρυπτογράφηση ───────────────────────────────────
                    model_path = os.path.join(
                        CloudResourcesPaths.MODELS_FOLDER_PATH.value,
                        f'{fog_name}_aggregated_model.keras'
                    )

                    try:
                        if encrypted and enc_mode == "ckks":
                            # ── CKKS: αποκρυπτογράφηση με secret key ──────
                            import tensorflow as tf
                            from shared.crypto_ckks import decrypt_weights_from_b64
                            ctx_path = "/app/shared/ckks_keys/ckks_secret_context.bin"
                            with open(ctx_path, "rb") as f:
                                secret_ctx_bytes = f.read()

                            weights, dec_metrics = decrypt_weights_from_b64(
                                model_b64, secret_ctx_bytes
                            )
                            logger.info(
                                "Cloud: CKKS decrypted model from fog %s | "
                                "decrypt=%.2fs | params=%d",
                                fog_name,
                                dec_metrics["decrypt_time_s"],
                                dec_metrics["total_params"],
                            )

                            # Ανακατασκεύασε .keras από weights
                            template_path = "/app/shared/ckks_keys/model_template.keras"
                            if not os.path.exists(template_path):
                                raise FileNotFoundError(
                                    f"CKKS: template model not found at {template_path}."
                                )
                            template_model = tf.keras.models.load_model(template_path)
                            template_model.set_weights(weights)

                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            template_model.save(model_path)

                        elif encrypted and enc_mode == "aes":
                            # ── AES: υπάρχουσα λογική αναλλοίωτη ──────────
                            model_bytes, dec_metrics = decrypt_from_b64(model_b64)
                            logger.info(
                                "Cloud: AES decrypted model from fog %s | decrypt=%.4fs",
                                fog_name, dec_metrics['decrypt_time_s'])
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            with open(model_path, 'wb') as f:
                                f.write(model_bytes)
                                f.flush()
                                os.fsync(f.fileno())

                        else:
                            model_bytes = base64.b64decode(model_b64)
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            with open(model_path, 'wb') as f:
                                f.write(model_bytes)

                    except Exception as e:
                        logger.error("Cloud: decryption failed for fog %s: %s", fog_name, e)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    try:
                        map_id = f"{fog_mac}_{fog_name}"
                        self.fog_models_cache[map_id] = {"model_path": model_path}
                        self._recent_fog_models[key]  = {"hash": model_hash, "ts": now}
                        logger.info("Cloud: cached model from fog '%s' (round=%s).",
                                    fog_name, recv_round_id)
                    except Exception as e:
                        logger.error("Cloud: failed to cache model from fog %s: %s", fog_name, e)
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
                logger.exception("Cloud: unexpected error; retrying in %ss…", delay)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            finally:
                try:
                    if conn and conn.is_open:
                        conn.close()
                except Exception:
                    pass

    def start_mqtt_listener(self):
        logger.info("Cloud: MQTT listener started (no-op in current setup).")
        while True:
            time.sleep(60)


if __name__ == "__main__":
    import sys
    logging_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if logging_root not in sys.path:
        sys.path.insert(0, logging_root)
    cloud_msg = CloudMessaging()
    logger.info("Cloud: Starting AMQP listener…")
    cloud_msg.start_amqp_listener()