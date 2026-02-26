import threading
import time
from typing import Callable, Any
from shared.logging_config import logger


class MonitoringThread(threading.Thread):
    def __init__(
            self,
            target: Callable[..., Any],
            sleep_time: float = 1,
            *args: Any,
            **kwargs: Any
    ):
        super().__init__()
        self._stop_event = threading.Event()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.delay = sleep_time

    def run(self):
        logger.info("Monitoring thread started.")
        while not self._stop_event.is_set():
            self.target(*self.args, **self.kwargs)
            time.sleep(self.delay)
        logger.info("Monitoring thread stopping...")

    def stop(self):
        self._stop_event.set()
