# pipeline/reader.py
import queue
import threading
import time

import cv2
import numpy as np
from loguru import logger
import config


class ReaderThread(threading.Thread):
    """OpenCV 读帧线程，将 BGR 帧放入 frame_queue。满时丢弃最旧帧以保持实时性。"""

    def __init__(
        self,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        rtsp_url: str = config.RTSP_INPUT,
    ) -> None:
        super().__init__(daemon=True, name="reader")
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.rtsp_url = rtsp_url

    def run(self) -> None:
        for attempt in range(config.FFMPEG_RETRY_MAX):
            if self.stop_event.is_set():
                return
            logger.info("Reader: attempt {}/{}", attempt + 1, config.FFMPEG_RETRY_MAX)
            try:
                self._read_loop()
            except Exception:
                logger.exception("Reader: unexpected error")
            if not self.stop_event.is_set():
                logger.info("Reader: reconnecting in {}s…", config.FFMPEG_RETRY_DELAY)
                time.sleep(config.FFMPEG_RETRY_DELAY)
        logger.error("Reader: max retries reached, giving up")

    def _read_loop(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("Reader: cannot open {}", self.rtsp_url)
            return

        read_count = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Reader: read failed after {} frames", read_count)
                    break
                read_count += 1
                frame = cv2.resize(frame, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
                # 满时弹出最旧帧再放新帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
                if read_count <= 5 or read_count % 100 == 0:
                    logger.info(
                        "Reader: read {} frames, qsize={}", read_count, self.frame_queue.qsize()
                    )
        finally:
            cap.release()
