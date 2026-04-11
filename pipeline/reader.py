# pipeline/reader.py
import queue
import subprocess
import threading
import time
import logging
import numpy as np
import config

logger = logging.getLogger(__name__)

_FRAME_SIZE = config.INPUT_WIDTH * config.INPUT_HEIGHT * 3  # BGR24 字节数


def _build_read_cmd(rtsp_url: str) -> list[str]:
    return [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-rtsp_transport", "tcp",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", rtsp_url,
        "-vf", f"scale={config.INPUT_WIDTH}:{config.INPUT_HEIGHT}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]


class ReaderThread(threading.Thread):
    """FFmpeg 读帧线程，将 BGR 帧放入 frame_queue。满时丢弃最旧帧以保持实时性。"""

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
            logger.info("Reader: attempt %d/%d", attempt + 1, config.FFMPEG_RETRY_MAX)
            try:
                self._read_loop()
            except Exception:
                logger.error("Reader: unexpected error", exc_info=True)
            if not self.stop_event.is_set():
                logger.info("Reader: reconnecting in %ds…", config.FFMPEG_RETRY_DELAY)
                time.sleep(config.FFMPEG_RETRY_DELAY)
        logger.error("Reader: max retries reached, giving up")

    def _read_loop(self) -> None:
        proc = subprocess.Popen(
            _build_read_cmd(self.rtsp_url),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while not self.stop_event.is_set():
                raw = proc.stdout.read(_FRAME_SIZE)
                if len(raw) != _FRAME_SIZE:
                    logger.warning("Reader: stream ended or short read (%d bytes)", len(raw))
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)
                )
                # 满时弹出最旧帧再放新帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
        finally:
            proc.terminate()
            proc.wait()
