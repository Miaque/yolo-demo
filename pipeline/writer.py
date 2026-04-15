# pipeline/writer.py
import os
import queue
import subprocess
import threading
import time
import numpy as np
from loguru import logger
from config import settings

_OUTPUT_FPS = 25


def _build_file_cmd(output_path: str) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{settings.INPUT_WIDTH}x{settings.INPUT_HEIGHT}",
        "-r", str(_OUTPUT_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-b:v", settings.OUTPUT_BITRATE,
        output_path,
    ]


class WriterThread(threading.Thread):
    """FFmpeg 写文件线程，按固定 FPS 写帧。无新帧时重复上一帧以保持时长正确。"""

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        output_path: str,
    ) -> None:
        super().__init__(daemon=True, name="writer")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def run(self) -> None:
        proc = subprocess.Popen(
            _build_file_cmd(self.output_path),
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        interval = 1.0 / _OUTPUT_FPS
        current_frame: np.ndarray | None = None
        next_time = time.monotonic()
        write_count = 0

        try:
            while not self.stop_event.is_set():
                # 排空队列，只保留最新帧
                got_new = False
                try:
                    while True:
                        current_frame = self.output_queue.get_nowait()
                        got_new = True
                except queue.Empty:
                    pass

                if current_frame is not None:
                    try:
                        proc.stdin.write(current_frame.tobytes())
                        proc.stdin.flush()
                        write_count += 1
                        if write_count % 100 == 0 or got_new:
                            logger.info(
                                "Writer: wrote {} frames (new={}, qsize={})",
                                write_count, got_new, self.output_queue.qsize(),
                            )
                    except BrokenPipeError:
                        logger.error("Writer: broken pipe")
                        break
                else:
                    if write_count == 0:
                        logger.debug("Writer: waiting for first frame…")

                next_time += interval
                sleep_time = next_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
            logger.info("Writer: saved to {}", self.output_path)
