# pipeline/writer.py
import queue
import subprocess
import threading
import numpy as np
from loguru import logger
import config


def _build_write_cmd(rtsp_url: str) -> list[str]:
    return [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{config.INPUT_WIDTH}x{config.INPUT_HEIGHT}",
        "-r", "25",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", config.OUTPUT_BITRATE,
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]


class WriterThread(threading.Thread):
    """FFmpeg 推流线程，将 output_queue 中的标注帧编码后推送至 RTSP。"""

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        rtsp_url: str = config.RTSP_OUTPUT,
    ) -> None:
        super().__init__(daemon=True, name="writer")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.rtsp_url = rtsp_url

    def run(self) -> None:
        proc = subprocess.Popen(
            _build_write_cmd(self.rtsp_url),
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    frame: np.ndarray = self.output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    proc.stdin.write(frame.tobytes())
                    proc.stdin.flush()
                except BrokenPipeError:
                    logger.error("Writer: broken pipe, RTSP server may have stopped")
                    break
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
            logger.info("Writer: stopped")
