import base64
import json
import os
import threading
import time
import random
import socket

import pika
import paho.mqtt.client as mqtt

from shared.logging_config import logger
from shared.crypto import encrypt_to_b64
from edge.communication.edge_service import EdgeService
from shared.node_state import FederatedNodeState


class EdgeMessaging:
    def __init__(self,
                 fog_amqp_host: str = os.getenv('FOG_RABBITMQ_HOST', 'rabbitmq-fog1'),
                 fog_mqtt_host: str = os.getenv('FOG_MQTT_HOST', 'mqtt-fog1'),
                 fog_mqtt_port: int = int(os.getenv('FOG_MQTT_PORT', 1883)),
                 edge_service: EdgeService = None):
        """
        :param fog_amqp_host: hostname or IP of the fog's RabbitMQ broker.
        :param fog_mqtt_host: hostname or IP of the fog's MQTT broker.
        :param fog_mqtt_port: port of the fog's MQTT broker.
        """
        self.fog_amqp_host = fog_amqp_host
        self.fog_mqtt_host = fog_mqtt_host
        self.fog_mqtt_port = fog_mqtt_port
        self.edge_service = edge_service
        self._last_cmd_id = None

    def _create_connection(self, retries=10, delay=5) -> pika.BlockingConnection:
        """Helper to create a new RabbitMQ connection with retry logic for fog node."""
        for attempt in range(1, retries + 1):
            try:
                connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.fog_amqp_host))
                return connection
            except pika.exceptions.AMQPConnectionError as e:
                logger.warning(f"Fog AMQP connection failed (attempt {attempt}/{retries}): {e})")
                if attempt == retries:
                    logger.error(
                        f"Could not connect to RabbitMQ broker ({self.fog_amqp_host}) after {retries} retries.")
                    raise
                time.sleep(delay)
        return None

    def start_amqp_listener(self):
        queue_name = f'edge_{FederatedNodeState.get_current_node().name}_messages_queue'

        def run():
            delay = 5
            max_delay = 60
            announced_down = False

            while True:
                conn = None
                try:
                    params = pika.ConnectionParameters(
                        host=self.fog_amqp_host,
                        heartbeat=30,
                        blocked_connection_timeout=60,
                        client_properties={
                            "connection_name": f"edge:{FederatedNodeState.get_current_node().name}:cmd-consumer"},
                    )
                    conn = pika.BlockingConnection(params)
                    ch = conn.channel()
                    ch.basic_qos(prefetch_count=1)
                    ch.queue_declare(queue=queue_name, durable=True, auto_delete=False)

                    def on_amqp_command(ch, method, _props, body):
                        try:
                            msg = json.loads(body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body)
                            cmd = str(msg.get('command'))
                            if cmd == '2':
                                logger.info("Edge %s: received AMQP command 2 (retrain/broadcast).",
                                            FederatedNodeState.get_current_node().name)
                                try:
                                    self.edge_service.retrain_fog_model(msg)
                                    logger.info("Edge %s: retrained local model.",
                                                FederatedNodeState.get_current_node().name)
                                except Exception:
                                    logger.exception("Edge: retrain_fog_model failed")
                            else:
                                logger.debug("Edge: AMQP message ignored (command=%r).", cmd)
                        except Exception:
                            logger.exception("Edge: failed to parse/handle AMQP message")
                        finally:
                            ch.basic_ack(delivery_tag=method.delivery_tag)

                    ch.basic_consume(queue=queue_name, on_message_callback=on_amqp_command, auto_ack=False)
                    logger.info("Edge %s: consuming AMQP commands on %s",
                                FederatedNodeState.get_current_node().name, queue_name)

                    if announced_down:
                        logger.info("Edge: AMQP reconnected.")
                        announced_down = False
                    delay = 5  # reset
                    ch.start_consuming()

                except (socket.gaierror, pika.exceptions.AMQPError) as e:
                    if not announced_down:
                        logger.warning("Edge: AMQP unavailable (%s). Backing off up to %ss...",
                                       e.__class__.__name__, max_delay)
                        announced_down = True
                    logger.debug("Edge: retrying AMQP connect in %ss...", delay)
                    time.sleep(delay + random.uniform(0, 1.0))
                    delay = min(delay * 2, max_delay)

                except Exception:
                    logger.exception("Edge: unexpected AMQP listener error; retrying in %ss", delay)
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

                finally:
                    try:
                        if conn and conn.is_open:
                            conn.close()
                    except Exception:
                        pass

        threading.Thread(target=run, daemon=True).start()

    def start_mqtt_listener(self, retries=10, delay=5) -> None:
        edge_name = FederatedNodeState.get_current_node().name
        topic = f'fog/{edge_name}/command'

        mqtt_client = mqtt.Client(client_id=f"edge-{edge_name}", clean_session=False)
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

        def on_connect(client, userdata, flags, rc):
            sess = flags.get('session present', flags.get('session_present', 0))
            logger.info(f"Edge {edge_name}: MQTT connected rc={rc}, session_present={sess}")
            client.subscribe(topic, qos=1)
            logger.info(f"Edge {edge_name}: (re)subscribed to {topic}")

        def on_disconnect(client, userdata, rc):
            logger.warning(f"Edge {edge_name}: MQTT disconnected rc={rc}, will auto-reconnect...")

        def on_mqtt_command(_client, _userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())

                cmd_id = payload.get("cmd_id")
                if cmd_id is not None and cmd_id == self._last_cmd_id:
                    logger.info(f"Edge: duplicate cmd_id {cmd_id} from fog ignored.")
                    return
                self._last_cmd_id = cmd_id

                cmd = str(payload.get('command'))
                logger.info(f"Edge {edge_name}: received MQTT command: {cmd}")
                if cmd == '0':
                    EdgeService.create_local_edge_model()
                elif cmd == '1':
                    self.edge_service.train_edge_local_model(payload)
                else:
                    logger.debug(f"Edge {edge_name}: ignoring MQTT command={cmd!r}")
            except Exception as e:
                logger.exception(f"Edge: error while handling MQTT command: {e}")

        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.on_message = on_mqtt_command

        for attempt in range(1, retries + 1):
            try:
                mqtt_client.connect(self.fog_mqtt_host, self.fog_mqtt_port)
                break
            except Exception as e:
                logger.warning(f"MQTT connection failed (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    logger.error(
                        f"Could not connect to MQTT broker ({self.fog_mqtt_host}:{self.fog_mqtt_port}) after {retries} retries.")
                    raise
                time.sleep(delay)

        mqtt_client.loop_forever()

    def send_trained_model(self, model_path: str, metrics: dict) -> None:
        """Send an AES-256-GCM encrypted trained model and metrics to the fog.

        Experiment 4: μετράμε χρόνο κρυπτογράφησης και μέγεθος payload.
        Τα crypto_metrics αποθηκεύονται στο payload για εξαγωγή από logs.
        """
        edge_name = FederatedNodeState.get_current_node().name

        # ── 1. Διαβάζουμε το μοντέλο ────────────────────────────────────────
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        # ── 2. AES-256-GCM κρυπτογράφηση ────────────────────────────────────
        model_b64_enc, enc_metrics = encrypt_to_b64(model_bytes)

        # ── 3. Χτίζουμε payload ──────────────────────────────────────────────
        payload = {
            "edge_mac":       FederatedNodeState.get_current_node().device_mac,
            "edge_name":      edge_name,
            "model":          model_b64_enc,
            "encrypted":      True,
            "metrics":        metrics,
            "crypto_metrics": enc_metrics,
            # encrypt_time_s, plaintext_size_b, ciphertext_size_b, overhead_bytes
        }

        # ── 4. Αποστολή μέσω RabbitMQ ────────────────────────────────────────
        connection = self._create_connection()
        channel = connection.channel()
        channel.queue_declare(queue='edge_to_fog_models', durable=True)

        t_send_start = time.perf_counter()
        message_body = json.dumps(payload).encode('utf-8')
        channel.basic_publish(
            exchange='',
            routing_key='edge_to_fog_models',
            body=message_body,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        t_send_end = time.perf_counter()
        connection.close()

        logger.info(
            "Edge %s: sent encrypted model to fog | "
            "encrypt=%.4fs | payload=%.1fKB | send=%.4fs",
            edge_name,
            enc_metrics['encrypt_time_s'],
            enc_metrics['ciphertext_size_b'] / 1024,
            t_send_end - t_send_start,
        )
        