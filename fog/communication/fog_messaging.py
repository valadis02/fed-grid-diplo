import base64
import hashlib
import json
import os
import threading
import time
import random
import socket

import pika
import paho.mqtt.client as mqtt
from pika import exceptions as px
from shared.logging_config import logger
from shared.node_state import FederatedNodeState
from shared.crypto import encrypt_to_b64, decrypt_from_b64
from fog.communication.fog_resources_paths import FogResourcesPaths
from fog.model.model_aggregation_service import aggregate_models_with_metrics
from shared.utils import delete_files_containing

# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------
FOG_NAME_DEFAULT   = os.getenv("FOG_NAME",   "FOG_NODE_1")
EDGE_NAMES_DEFAULT = os.getenv("EDGE_NAMES", "edge_node_1").split(",")

def _fog_name() -> str:
    node = FederatedNodeState.get_current_node()
    return node.name if node is not None else FOG_NAME_DEFAULT

def _edge_names() -> list:
    node = FederatedNodeState.get_current_node()
    if node is not None:
        children = getattr(node, "child_nodes", []) or []
        if children:
            return [e.name for e in children]
    return EDGE_NAMES_DEFAULT


def _purge_outbox_all():
    try:
        delete_files_containing(
            FogResourcesPaths.OUTBOX_FOLDER_PATH.value, None, [".json", ".keras"]
        )
        logger.info("Fog: purged outbox (all files).")
    except Exception as e:
        logger.warning("Fog: purge outbox failed: %s", e)


class FogMessaging:
    def __init__(self,
                 fog_amqp_host:   str = os.getenv('FOG_RABBITMQ_HOST',  'rabbitmq-fog1'),
                 cloud_amqp_host: str = os.getenv('CLOUD_RABBITMQ_HOST', 'rabbitmq-cloud'),
                 cloud_amqp_port: int = int(os.getenv('CLOUD_RABBITMQ_PORT', '5672')),
                 fog_mqtt_host:   str = os.getenv('FOG_MQTT_HOST',  'mqtt-fog1'),
                 fog_mqtt_port:   int = int(os.getenv('FOG_MQTT_PORT',  1883)),
                 cloud_mqtt_host: str = os.getenv('CLOUD_MQTT_HOST', 'mqtt-cloud'),
                 cloud_mqtt_port: int = int(os.getenv('CLOUD_MQTT_PORT', 1883))):

        self.fog_amqp_host   = fog_amqp_host
        self.cloud_amqp_host = cloud_amqp_host
        self.cloud_amqp_port = cloud_amqp_port
        self.fog_mqtt_host   = fog_mqtt_host
        self.fog_mqtt_port   = fog_mqtt_port
        self.cloud_mqtt_host = cloud_mqtt_host
        self.cloud_mqtt_port = cloud_mqtt_port

        self.edge_models_cache = {}
        self._last_cmd_id    = None
        self._outbox_enabled = os.getenv('FOG_RESUME_OUTBOX_ON_BOOT', 'false').lower() == 'true'
        self.current_round   = None
        self.OUTBOX_TTL_SECS = int(os.getenv('FOG_OUTBOX_TTL_SECS', '86400'))

        # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: timestamp πρώτου μοντέλου ──────────────────────────
        self._round_start_time: float = None

        _purge_outbox_all()
        logger.info("Fog: purged outbox on boot (no active round).")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _create_connection(self, retries=10, delay=5) -> pika.BlockingConnection:
        for attempt in range(1, retries + 1):
            try:
                return pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.fog_amqp_host))
            except pika.exceptions.AMQPConnectionError as e:
                logger.warning("Fog AMQP connection failed (attempt %d/%d): %s", attempt, retries, e)
                if attempt == retries:
                    raise
                time.sleep(delay)

    def _create_connection_to_cloud(self):
        delay, max_delay = 5, 60
        while True:
            try:
                return pika.BlockingConnection(pika.ConnectionParameters(
                    host=self.cloud_amqp_host,
                    port=self.cloud_amqp_port,
                    heartbeat=30,
                    blocked_connection_timeout=60,
                    connection_attempts=1,
                    retry_delay=0,
                ))
            except (socket.gaierror, pika.exceptions.AMQPError) as e:
                time.sleep(delay + random.uniform(0, 1.0))
                delay = min(delay * 2, max_delay)

    def _prune_outbox_for_round(self):
        outbox_dir = FogResourcesPaths.OUTBOX_FOLDER_PATH.value
        try:
            for fname in os.listdir(outbox_dir):
                if not fname.endswith(".json"):
                    continue
                meta_path = os.path.join(outbox_dir, fname)
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception:
                    try: os.remove(meta_path)
                    except: pass
                    continue
                if meta.get("round_id") != self.current_round:
                    try: os.remove(meta_path)
                    except: pass
                    try: os.remove(meta.get("model_file", ""))
                    except: pass
        except FileNotFoundError:
            pass

    # -----------------------------------------------------------------------
    # MQTT listener
    # -----------------------------------------------------------------------

    def start_mqtt_listener(self):
        CLEAR_AFTER_SECONDS = 300
        WATCHDOG_PERIOD = int(os.getenv("FOG_MQTT_WATCHDOG_SECS", "10"))

        fog_name = _fog_name()
        cloud_host, cloud_port = self.cloud_mqtt_host, self.cloud_mqtt_port
        fog_local_host, fog_local_port = self.fog_mqtt_host, self.fog_mqtt_port

        cloud_client = mqtt.Client(client_id=f"fog-cloud-{fog_name}", clean_session=False)
        fog_client   = mqtt.Client(client_id=f"fog-local-{fog_name}", clean_session=False)

        cloud_client.will_set(f"fogs/{fog_name}/status",    payload="offline", qos=1, retain=True)
        fog_client.will_set(f"fog/{fog_name}/local_status", payload="offline", qos=1, retain=True)

        cloud_client.reconnect_delay_set(min_delay=1, max_delay=60)
        fog_client.reconnect_delay_set(min_delay=1, max_delay=60)

        clear_timers = {}

        def _mark_online(client, topic):
            try:
                client.publish(topic, payload="online", qos=1, retain=True)
            except Exception as e:
                logger.warning("MQTT: failed to publish online status to %s: %s", topic, e)

        def on_cloud_connect(client, _userdata, _flags, rc):
            if rc == 0:
                _mark_online(client, f"fogs/{fog_name}/status")
                try:
                    client.subscribe("cloud/fog/command", qos=1)
                    logger.info("Fog: subscribed to cloud/fog/command on CLOUD broker.")
                except Exception as e:
                    logger.warning("Fog: subscribe failed on CLOUD broker: %s", e)
            else:
                logger.error("Fog (cloud client): connect failed, rc=%s", rc)

        def on_cloud_disconnect(_client, _userdata, rc):
            if rc != 0:
                logger.warning("Fog: disconnected from CLOUD MQTT (rc=%s). Will auto-reconnect.", rc)

        def on_cloud_message(_client, _userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
            except Exception as e:
                logger.warning("Fog: bad MQTT payload from cloud: %s", e)
                return

            command = payload.get("command")
            if command is None:
                return

            round_id = payload.get("round_id")
            if round_id is not None and round_id != self.current_round:
                logger.info("Fog: new round_id=%s (was %s) → purging outbox.", round_id, self.current_round)
                self.current_round = round_id
                _purge_outbox_all()
                # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: reset round timer σε νέο round ─────────────
                self._round_start_time = None

            if command in ('1', '2'):
                self._outbox_enabled = True

            cmd_id = payload.get("cmd_id")
            if cmd_id is not None and cmd_id == self._last_cmd_id:
                logger.info("Fog: duplicate cmd_id %s from cloud ignored.", cmd_id)
                return
            self._last_cmd_id = cmd_id

            for edge_name in _edge_names():
                topic = f"fog/{edge_name}/command"
                try:
                    fog_client.publish(topic, json.dumps(payload), qos=1, retain=True)
                    logger.info("Fog (relay): sent command %s to edge %s (retained)", command, edge_name)
                    _schedule_clear(topic)
                except Exception as e:
                    logger.warning("Fog: failed to publish command to %s: %s", topic, e)

        cloud_client.on_connect    = on_cloud_connect
        cloud_client.on_disconnect = on_cloud_disconnect
        cloud_client.on_message    = on_cloud_message

        def on_fog_connect(client, _userdata, _flags, rc):
            if rc == 0:
                _mark_online(client, f"fog/{fog_name}/local_status")
            else:
                logger.error("Fog (local client): connect failed, rc=%s", rc)

        def on_fog_disconnect(_client, _userdata, rc):
            if rc != 0:
                logger.warning("Fog: disconnected from LOCAL MQTT (rc=%s). Will auto-reconnect.", rc)

        fog_client.on_connect    = on_fog_connect
        fog_client.on_disconnect = on_fog_disconnect

        def _schedule_clear(topic):
            t = clear_timers.pop(topic, None)
            if t and t.is_alive():
                t.cancel()
            def _clear():
                try:
                    fog_client.publish(topic, payload=b"", qos=1, retain=True)
                    logger.info("Fog: cleared retained command on %s", topic)
                except Exception as e:
                    logger.warning("Fog: failed to clear retained on %s: %s", topic, e)
                finally:
                    clear_timers.pop(topic, None)
            timer = threading.Timer(CLEAR_AFTER_SECONDS, _clear)
            clear_timers[topic] = timer
            timer.start()

        cloud_client.loop_start()
        fog_client.loop_start()

        try:
            cloud_client.connect_async(cloud_host, cloud_port, keepalive=60)
        except Exception as e:
            logger.warning("Fog: initial connect_async to CLOUD MQTT failed: %s", e)

        try:
            fog_client.connect_async(fog_local_host, fog_local_port, keepalive=60)
        except Exception as e:
            logger.warning("Fog: initial connect_async to LOCAL MQTT failed: %s", e)

        def _watchdog():
            while True:
                try:
                    if not cloud_client.is_connected():
                        try:
                            cloud_client.connect_async(cloud_host, cloud_port, keepalive=60)
                        except Exception as e:
                            logger.debug("Fog: watchdog nudge failed: %s", e)
                    if not fog_client.is_connected():
                        try:
                            fog_client.connect_async(fog_local_host, fog_local_port, keepalive=60)
                        except Exception as e:
                            logger.debug("Fog: local watchdog nudge failed: %s", e)
                except Exception:
                    logger.exception("Fog: MQTT watchdog exception (ignored).")
                time.sleep(WATCHDOG_PERIOD)

        threading.Thread(target=_watchdog, daemon=True).start()

        while True:
            time.sleep(60)

    # -----------------------------------------------------------------------
    # AMQP listener (Cloud → Fog bridge)
    # -----------------------------------------------------------------------

    def start_amqp_listener(self):
        def run():
            fog_name = _fog_name()
            delay, max_delay = 5, 60
            announced_down = False
            next_warn_at   = 0.0

            use_quorum = os.getenv("FOG_USE_QUORUM_QUEUES", "false").lower() == "true"
            queue_args = {"x-queue-type": "quorum"} if use_quorum else None

            while True:
                cloud_conn = None
                try:
                    cloud_params = pika.ConnectionParameters(
                        host=self.cloud_amqp_host,
                        port=self.cloud_amqp_port,
                        heartbeat=30,
                        blocked_connection_timeout=60,
                        connection_attempts=1,
                        retry_delay=0,
                        client_properties={"connection_name": f"fog:{fog_name}:cloud-consumer"},
                    )
                    cloud_conn = pika.BlockingConnection(cloud_params)
                    cloud_ch   = cloud_conn.channel()
                    queue_name = f'cloud_fanout_for_{fog_name}'

                    declare_result = cloud_ch.queue_declare(
                        queue=queue_name,
                        durable=True,
                        auto_delete=False,
                        arguments=queue_args,
                    )
                    logger.info("Fog: declared queue '%s'", declare_result.method.queue)
                    cloud_ch.basic_qos(prefetch_count=1)

                    def on_amqp_model(ch, method, _props, body):
                        try:
                            msg = json.loads(body.decode("utf-8"))
                        except Exception as e:
                            logger.warning("Fog: invalid cloud AMQP payload: %s", e)
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            return

                        if msg.get("command") == "2" or "model" in msg:
                            self._outbox_enabled = True

                        if msg.get("model"):
                            try:
                                os.makedirs(
                                    os.path.dirname(FogResourcesPaths.FOG_MODEL_FILE_PATH.value),
                                    exist_ok=True)
                                with open(FogResourcesPaths.FOG_MODEL_FILE_PATH.value, "wb") as f:
                                    f.write(base64.b64decode(msg["model"]))
                                    f.flush(); os.fsync(f.fileno())
                                logger.info("Fog (AMQP): updated local fog model from cloud broadcast.")
                            except Exception as e:
                                logger.exception("Fog: failed writing fog model: %s", e)

                        rid = msg.get("round_id")
                        if rid is not None and rid != self.current_round:
                            logger.info("Fog: new round_id=%s → purging outbox.", rid)
                            self.current_round = rid
                            _purge_outbox_all()
                            # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: reset round timer ──────────────
                            self._round_start_time = None

                        edge_names = _edge_names()
                        try:
                            fog_params = pika.ConnectionParameters(
                                host=self.fog_amqp_host,
                                heartbeat=30,
                                blocked_connection_timeout=60,
                                client_properties={"connection_name": f"fog:{fog_name}:edge-forwarder"},
                            )
                            with pika.BlockingConnection(fog_params) as fog_conn:
                                fog_ch = fog_conn.channel()
                                for edge_name in edge_names:
                                    edge_q = f"edge_{edge_name}_messages_queue"
                                    fog_ch.queue_declare(queue=edge_q, durable=True, auto_delete=False)
                                    fog_ch.basic_publish(
                                        exchange='',
                                        routing_key=edge_q,
                                        body=json.dumps(msg).encode('utf-8'),
                                        properties=pika.BasicProperties(
                                            delivery_mode=2,
                                            content_type="application/json",
                                        ),
                                    )
                                    logger.info("Fog (AMQP): forwarded cloud model to edge '%s'.", edge_name)
                        except Exception as e:
                            logger.exception("Fog: failed forwarding to edges: %s", e)

                        ch.basic_ack(delivery_tag=method.delivery_tag)

                    cloud_ch.basic_consume(queue=queue_name, on_message_callback=on_amqp_model, auto_ack=False)
                    logger.info("Fog: consuming cloud messages from queue '%s'.", queue_name)

                    if announced_down:
                        logger.info("Fog: cloud AMQP back online.")
                        announced_down = False
                        delay = 5

                    cloud_ch.start_consuming()

                except (socket.gaierror, pika.exceptions.AMQPError) as e:
                    now = time.time()
                    if not announced_down:
                        logger.warning("Fog: cloud AMQP unavailable (%s).", e.__class__.__name__)
                        announced_down = True
                        next_warn_at = now + 30
                    time.sleep(delay + random.uniform(0, 1.0))
                    delay = min(delay * 2, max_delay)
                except Exception:
                    logger.exception("Fog: unexpected error in cloud AMQP bridge; retrying in %ss...", delay)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                finally:
                    try:
                        if cloud_conn and cloud_conn.is_open:
                            cloud_conn.close()
                    except Exception:
                        pass

        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Edge model listener (Edge → Fog)
    # -----------------------------------------------------------------------

    def start_edge_model_listener(self):
        def run():
            delay, max_delay = 5, 60
            while True:
                conn = None
                try:
                    f_name = _fog_name()
                    params = pika.ConnectionParameters(
                        host=self.fog_amqp_host,
                        heartbeat=30,
                        blocked_connection_timeout=60,
                        client_properties={"connection_name": f"fog:{f_name}:edge-consumer"},
                    )
                    conn = pika.BlockingConnection(params)
                    ch   = conn.channel()
                    ch.queue_declare(queue='edge_to_fog_models', durable=True, auto_delete=False)
                    ch.basic_qos(prefetch_count=1)

                    def on_edge_model(ch, method, _props, body):
                        try:
                            payload   = json.loads(body)
                            edge_name = payload.get('edge_name', 'unknown')
                            metrics   = payload.get('metrics', {})
                            model_b64 = payload.get('model')
                            encrypted = payload.get('encrypted', False)
                            enc_mode  = payload.get('encryption_mode', 'aes')

                            if not model_b64:
                                logger.warning("Fog: edge model payload missing 'model' field.")
                                ch.basic_ack(delivery_tag=method.delivery_tag)
                                return

                            # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: timestamp πρώτου μοντέλου ──────
                            if len(self.edge_models_cache) == 0:
                                self._round_start_time = time.perf_counter()
                                logger.info(
                                    "Fog: first edge model received from %s — round timer started.",
                                    edge_name,
                                )

                            if encrypted and enc_mode == "ckks":
                                logger.info(
                                    "Fog: received CKKS model from edge %s "
                                    "(will aggregate in HE domain)",
                                    edge_name,
                                )
                                model_path = os.path.join(
                                    FogResourcesPaths.MODELS_FOLDER_PATH.value,
                                    f"{edge_name}_model.ckks.bin",
                                )
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                with open(model_path, "wb") as mf:
                                    mf.write(base64.b64decode(model_b64))

                                self.edge_models_cache[edge_name] = {
                                    "model_path": model_path,
                                    "metrics":    metrics,
                                    "ckks":       True,
                                }

                            elif encrypted and enc_mode == "aes":
                                model_bytes, dec_metrics = decrypt_from_b64(model_b64)
                                logger.info(
                                    "Fog: decrypted AES model from edge %s | decrypt=%.4fs",
                                    edge_name, dec_metrics['decrypt_time_s'],
                                )
                                model_path = os.path.join(
                                    FogResourcesPaths.MODELS_FOLDER_PATH.value,
                                    f"{edge_name}_model.keras",
                                )
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                with open(model_path, "wb") as mf:
                                    mf.write(model_bytes)

                                self.edge_models_cache[edge_name] = {
                                    "model_path": model_path,
                                    "metrics":    metrics,
                                    "ckks":       False,
                                }

                            else:
                                model_bytes = base64.b64decode(model_b64)
                                model_path = os.path.join(
                                    FogResourcesPaths.MODELS_FOLDER_PATH.value,
                                    f"{edge_name}_model.keras",
                                )
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                with open(model_path, "wb") as mf:
                                    mf.write(model_bytes)

                                self.edge_models_cache[edge_name] = {
                                    "model_path": model_path,
                                    "metrics":    metrics,
                                    "ckks":       False,
                                }

                            logger.info(
                                "Fog: cached model for edge %s | mode=%s | metrics=%s",
                                edge_name, enc_mode, metrics,
                            )
                            self._outbox_enabled = True

                        except Exception as e:
                            logger.exception("Fog: error processing edge model: %s", e)
                        finally:
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                            self.is_ready_to_aggregate()

                    ch.basic_consume(queue='edge_to_fog_models', on_message_callback=on_edge_model, auto_ack=False)
                    logger.info("Fog: listening for trained models from edges...")
                    delay = 5
                    ch.start_consuming()

                except Exception as e:
                    logger.error("Fog: edge model listener error: %s", e)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                finally:
                    try:
                        if conn and conn.is_open:
                            conn.close()
                    except Exception:
                        pass

        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Aggregation gate
    # -----------------------------------------------------------------------

    def is_ready_to_aggregate(self):
        node = FederatedNodeState.get_current_node()
        num_expected = len(node.child_nodes) if node is not None else len(EDGE_NAMES_DEFAULT)

        logger.info("Fog: Received %d/%d models.", len(self.edge_models_cache), num_expected)

        if len(self.edge_models_cache) < num_expected:
            return False

        logger.info("Fog: All edge models received. Ready to aggregate!")

        all_ckks = all(v.get("ckks", False) for v in self.edge_models_cache.values())

        if all_ckks:
            try:
                from shared.crypto_ckks import he_aggregate
                ctx_path = "/app/shared/ckks_keys/ckks_public_context_32768.bin"
                with open(ctx_path, "rb") as f:
                    pub_ctx_bytes = f.read()

                payload_bytes_list = []
                sample_counts = []
                for entry in self.edge_models_cache.values():
                    with open(entry["model_path"], "rb") as f:
                        payload_bytes_list.append(f.read())
                    mse = (entry["metrics"].get("after_training", {}).get("mse")
                           or entry["metrics"].get("mse", 1.0))
                    sample_counts.append(1.0 / (float(mse) + 1e-8))

                logger.info("Fog: starting CKKS HE aggregation for %d clients...",
                            len(payload_bytes_list))
                agg_bytes, agg_metrics = he_aggregate(
                    payload_bytes_list,
                    pub_ctx_bytes,
                    sample_counts=sample_counts,
                )
                logger.info(
                    "Fog: CKKS HE aggregation done | time=%.2fs | size=%.1fKB",
                    agg_metrics["aggregate_time_s"],
                    agg_metrics["result_size_kb"],
                )

                agg_path = os.path.join(
                    FogResourcesPaths.MODELS_FOLDER_PATH.value,
                    "aggregated_fog_model.ckks.bin"
                )
                with open(agg_path, "wb") as f:
                    f.write(agg_bytes)

                self._send_ckks_model_to_cloud(agg_path, agg_bytes)

            except Exception as e:
                logger.exception("Fog: CKKS HE aggregation failed: %s", e)
                self.edge_models_cache.clear()
                return False

        else:
            try:
                # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: πέρνα το round_start_time ─────────────────
                aggregated = aggregate_models_with_metrics(
                    self.edge_models_cache,
                    round_start_time=self._round_start_time,
                )
            except Exception as e:
                logger.exception("Fog: AES aggregation failed: %s", e)
                return False

            if aggregated is None:
                logger.error("Fog: aggregation produced no model.")
                return False

            logger.info("Fog: aggregation succeeded. Sending to cloud.")
            self.send_aggregated_model_to_cloud()

        # ── ΝΕΕΣ ΑΛΛΑΓΕΣ: reset για το επόμενο round ─────────────────────────
        self._round_start_time = None
        self.edge_models_cache.clear()
        return True

    def _send_ckks_model_to_cloud(self, agg_path: str, agg_bytes: bytes):
        fog_name   = _fog_name()
        fog_mac    = "00:00:00:00:00:00"
        model_hash = hashlib.sha256(agg_bytes).hexdigest()
        model_b64  = base64.b64encode(agg_bytes).decode("utf-8")

        body = json.dumps({
            "fog_name":        fog_name,
            "fog_device_mac":  fog_mac,
            "model":           model_b64,
            "encrypted":       True,
            "encryption_mode": "ckks",
            "hash":            model_hash,
            "round_id":        self.current_round,
        }).encode("utf-8")

        try:
            conn = self._create_connection_to_cloud()
            ch   = conn.channel()
            ch.queue_declare(queue="fog_to_cloud_models", durable=True)
            ch.basic_publish(
                exchange="",
                routing_key="fog_to_cloud_models",
                body=body,
                properties=pika.BasicProperties(delivery_mode=2),
            )
            conn.close()
            logger.info(
                "Fog: sent CKKS aggregated model to cloud | size=%.1fKB",
                len(agg_bytes) / 1024,
            )
        except Exception as e:
            logger.exception("Fog: failed to send CKKS model to cloud: %s", e)

    def send_aggregated_model_to_cloud(self):
        model_path = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        self.enqueue_model_for_cloud(model_path, model_bytes=model_bytes)
        try:
            os.remove(model_path)
        except Exception:
            pass
        logger.info("Fog: queued aggregated fog model for cloud uplink.")

    def enqueue_model_for_cloud(self, model_path: str, precomputed_hash: str = None, model_bytes: bytes = None):
        fog_name = _fog_name()
        fog_mac  = "00:00:00:00:00:00"

        outbox_dir = FogResourcesPaths.OUTBOX_FOLDER_PATH.value
        os.makedirs(outbox_dir, exist_ok=True)

        if model_bytes is None:
            with open(model_path, "rb") as f:
                model_bytes = f.read()
        model_hash = precomputed_hash or hashlib.sha256(model_bytes).hexdigest()

        meta_path = os.path.join(outbox_dir, f"{model_hash}.json")
        blob_path = os.path.join(outbox_dir, f"{model_hash}.keras")

        if os.path.exists(meta_path) and os.path.exists(blob_path):
            logger.info("Fog: model %s already queued; skipping duplicate.", model_hash)
            return

        with open(blob_path, "wb") as mf:
            mf.write(model_bytes)
            mf.flush(); os.fsync(mf.fileno())

        if self.current_round is None:
            self.current_round = "TEST_ROUND"

        meta = {
            "fog_name":       fog_name,
            "fog_device_mac": fog_mac,
            "model_file":     blob_path,
            "hash":           model_hash,
            "message_id":     f"{fog_mac}:{fog_name}:{model_hash}",
            "round_id":       self.current_round,
            "ts":             int(time.time()),
        }
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w") as jf:
            json.dump(meta, jf)
            jf.flush(); os.fsync(jf.fileno())
        os.replace(tmp_meta, meta_path)
        logger.info("Fog: queued model %s for later uplink.", model_hash)

    def start_cloud_uplink_worker(self):
        def run():
            delay, max_delay = 5, 60
            announced_down   = False
            next_warn_at     = 0.0

            while True:
                if self.current_round is None:
                    time.sleep(2)
                    continue
                if not self._outbox_enabled:
                    time.sleep(2)
                    continue

                outbox_dir = FogResourcesPaths.OUTBOX_FOLDER_PATH.value
                os.makedirs(outbox_dir, exist_ok=True)
                files = sorted(f for f in os.listdir(outbox_dir) if f.endswith(".json"))
                if not files:
                    time.sleep(2)
                    continue

                cloud_conn = None
                try:
                    cloud_conn = self._create_connection_to_cloud()
                    ch = cloud_conn.channel()
                    ch.confirm_delivery()
                    ch.queue_declare(queue="fog_to_cloud_models", durable=True)

                    def _on_return(_ch, method, props, body):
                        logger.warning("Fog: broker returned message.")
                    ch.add_on_return_callback(_on_return)

                    for fname in files:
                        meta_path = os.path.join(outbox_dir, fname)
                        try:
                            with open(meta_path) as f:
                                meta = json.load(f)
                        except Exception as e:
                            logger.warning("Fog: bad outbox meta %s: %s; removing.", meta_path, e)
                            try: os.remove(meta_path)
                            except: pass
                            continue

                        round_id = meta.get("round_id")
                        if round_id != self.current_round:
                            continue

                        ts         = meta.get("ts", 0)
                        blob_path  = meta.get("model_file") or meta.get("model_path")
                        fog_name   = meta.get("fog_name")
                        fog_mac    = meta.get("fog_device_mac")
                        model_hash = meta.get("hash")
                        msg_id     = meta.get("message_id", f"{fog_mac}:{fog_name}:{model_hash}")

                        if self.OUTBOX_TTL_SECS > 0 and (time.time() - ts) > self.OUTBOX_TTL_SECS:
                            logger.info("Fog: pruning expired outbox item %s.", meta_path)
                            try: os.remove(meta_path)
                            except: pass
                            try: os.remove(blob_path or "")
                            except: pass
                            continue

                        if not blob_path or not os.path.exists(blob_path):
                            logger.warning("Fog: missing blob for %s; dropping.", meta_path)
                            try: os.remove(meta_path)
                            except: pass
                            continue

                        with open(blob_path, "rb") as mf:
                            model_bytes = mf.read()

                        model_b64_enc, enc_metrics = encrypt_to_b64(model_bytes)
                        logger.info(
                            "Fog: AES encrypted model for cloud | encrypt=%.4fs | size=%.1fKB",
                            enc_metrics['encrypt_time_s'],
                            enc_metrics['ciphertext_size_b'] / 1024,
                        )

                        body = json.dumps({
                            "fog_name":        fog_name,
                            "fog_device_mac":  fog_mac,
                            "model":           model_b64_enc,
                            "encrypted":       True,
                            "encryption_mode": "aes",
                            "hash":            model_hash,
                            "round_id":        round_id,
                            "crypto_metrics":  enc_metrics,
                        }).encode("utf-8")

                        props = pika.BasicProperties(
                            delivery_mode=2,
                            content_type="application/json",
                            message_id=msg_id,
                        )

                        try:
                            ch.basic_publish(
                                exchange="",
                                routing_key="fog_to_cloud_models",
                                body=body,
                                properties=props,
                                mandatory=True,
                            )
                            logger.info("Fog: publish confirmed (msg_id=%s).", msg_id)
                            try: os.remove(meta_path)
                            except: pass
                            try: os.remove(blob_path)
                            except: pass
                        except px.UnroutableError:
                            logger.warning("Fog: unroutable publish. Will retry.")
                        except px.NackError:
                            logger.warning("Fog: broker NACKed publish. Will retry.")
                        except px.StreamLostError:
                            logger.warning("Fog: stream lost during publish. Will retry.")

                except (px.NackError, px.UnroutableError, px.StreamLostError,
                        socket.gaierror, pika.exceptions.AMQPError) as e:
                    now = time.time()
                    if not announced_down:
                        logger.warning("Fog: cloud uplink issue (%s).", e.__class__.__name__)
                        announced_down = True
                        next_warn_at = now + 30
                    time.sleep(delay + random.uniform(0, 1.0))
                    delay = min(delay * 2, max_delay)
                except Exception:
                    logger.exception("Fog: unexpected error in uplink worker; retrying in %ss...", delay)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                finally:
                    try:
                        if cloud_conn and cloud_conn.is_open:
                            cloud_conn.close()
                    except Exception:
                        pass

        threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    messaging = FogMessaging()
    logger.info("Fog: Starting all messaging services...")
    messaging.start_edge_model_listener()
    messaging.start_amqp_listener()
    messaging.start_cloud_uplink_worker()
    messaging.start_mqtt_listener()