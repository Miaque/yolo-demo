# main.py
import queue
import signal
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from loguru import logger
from config import settings
import overlay
import state
from pipeline.detector import FaceDetector
from pipeline.embedder import EmbedderThread
from pipeline.reader import ReaderThread
from pipeline.tracker import FaceTracker
from pipeline.writer import WriterThread
from pipeline.aligner import FaceAligner


def _crop_with_margin(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """BBox 周围外扩 FACE_CROP_MARGIN 比例，裁剪人脸区域。

    在检测到的人脸边界框外扩一定比例（默认 20%），确保裁剪出的人脸
    包含更多上下文信息，有利于后续 embedding 提取和识别。
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * settings.FACE_CROP_MARGIN), int(bh * settings.FACE_CROP_MARGIN)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    return frame[cy1:cy2, cx1:cx2].copy()


def _load_known_faces(aligner: FaceAligner) -> dict[str, np.ndarray]:
    """启动时从 faces/ 目录加载已知人脸 embedding，使用与管线相同的 FaceAligner。"""
    from pathlib import Path

    faces_dir = Path(settings.FACES_DIR)
    if not faces_dir.is_dir():
        logger.warning("Faces directory '{}' not found, all faces will be Unknown", settings.FACES_DIR)
        return {}

    known_faces: dict[str, np.ndarray] = {}
    for img_path in sorted(faces_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: {}", img_path)
            continue

        result = aligner.align(img)
        if result is None:
            logger.warning("No face detected in: {}", img_path)
            continue

        name = img_path.stem
        known_faces[name] = result.embedding
        logger.info(
            "Loaded known face: {} | emb_norm={:.4f} | emb_shape={}",
            name,
            float(np.linalg.norm(result.embedding)),
            result.embedding.shape,
        )

    logger.info("Total known faces loaded: {}", len(known_faces))
    return known_faces


def run() -> None:
    """启动人脸检测、跟踪、识别管线的主循环。

    管线采用单进程多线程架构，四个线程通过 bounded Queue 通信：

        ReaderThread ──frame_queue──> Main Thread ──face_crop_queue──> EmbedderThread
                                            │                                  │
                                            └──output_queue──> WriterThread     │
                                                 (FFmpeg MP4)                  │
                                                  ▲                            │
                                                  └── state.py (thread-safe) ──┘

    Main thread 负责核心检测与跟踪逻辑，每隔 DETECT_INTERVAL 帧执行两阶段
    YOLO 检测（person → face），其余帧仅执行 ByteTrack 预测。
    新出现的 track ID 会被裁剪出人脸区域并送入 EmbedderThread 进行 embedding 提取。
    """
    stop_event = threading.Event()

    # 创建线程间通信队列，设置最大容量以防止内存无限增长
    frame_queue: queue.Queue = queue.Queue(maxsize=settings.FRAME_QUEUE_SIZE)
    face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=settings.OUTPUT_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()

    # 创建 FaceAligner 用于加载已知人脸，确保 embedding 空间与管线一致
    # 加载完成后立即关闭以释放资源
    known_face_aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)
    try:
        known_faces = _load_known_faces(known_face_aligner)
    finally:
        known_face_aligner.close()

    # 根据当前时间生成输出文件路径，确保每次运行产生唯一文件名
    output_path = f"{settings.OUTPUT_DIR}/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    reader = ReaderThread(frame_queue, stop_event)
    embedder = EmbedderThread(face_crop_queue, stop_event, known_faces=known_faces)
    writer = WriterThread(output_queue, stop_event, output_path=output_path)

    known_ids: set[int] = set()

    def _handle_signal(sig, _frame) -> None:
        logger.info("Signal {} received, shutting down…", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    reader.start()
    embedder.start()
    writer.start()
    logger.info("Pipeline started. Input: {}, Output: {}", settings.RTSP_INPUT, output_path)

    frame_count = 0
    track_results: list[tuple[int, list[int]]] = []  # 存储当前帧的跟踪结果

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                # 队列为空时记录日志（每 50 帧一次），避免刷屏
                if frame_count > 0 and frame_count % 50 == 0:
                    logger.info("Main: frame_queue empty after {} frames", frame_count)
                continue

            frame_count += 1
            # 每 100 帧打印一次队列状态，用于性能监控
            if frame_count % 100 == 0:
                logger.info(
                    "Main: frame {} | frame_q={} crop_q={} out_q={}",
                    frame_count, frame_queue.qsize(),
                    face_crop_queue.qsize(), output_queue.qsize(),
                )

            # 每隔 DETECT_INTERVAL 帧执行一次完整检测 + 跟踪更新
            # 其余帧仅执行跟踪预测，以平衡精度与性能
            if frame_count % settings.DETECT_INTERVAL == 0:
                t0 = time.monotonic()
                detections = detector.detect(frame)
                t1 = time.monotonic()
                # 用检测结果更新跟踪器，获得带 track_id 的跟踪结果
                track_results = tracker.update(detections, frame)
                t2 = time.monotonic()
                # 前 20 帧及之后每 100 帧打印检测/跟踪耗时，用于性能分析
                if frame_count <= 20 or frame_count % 100 == 0:
                    logger.info(
                        "Main: frame {} detect={:.0f}ms track={:.0f}ms dets={}",
                        frame_count, (t1 - t0) * 1000, (t2 - t1) * 1000,
                        len(detections),
                    )

                # 清理已消失 track 的 embedding 缓存，防止内存泄漏
                for removed_id in tracker.removed_ids():
                    known_ids.discard(removed_id)
                    state.remove_embedding(removed_id)
            else:
                # 非检测帧：仅执行跟踪预测，降低计算开销
                track_results = tracker.predict()

            # 对新出现的 track ID，裁剪人脸区域并送入 EmbedderThread 做 embedding 提取
            for track_id, bbox in track_results:
                if track_id not in known_ids:
                    known_ids.add(track_id)
                    face_crop = _crop_with_margin(frame, bbox)
                    try:
                        face_crop_queue.put_nowait((track_id, face_crop))
                    except queue.Full:
                        pass  # EmbedderThread 队列已满，跳过此帧

            # 获取当前最新的 embedding 和识别结果快照，用于绘制标注
            emb_snapshot, name_snapshot = state.snapshot()
            annotated = overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)
            try:
                output_queue.put_nowait(annotated)
            except queue.Full:
                logger.warning("Main: output_queue full, dropping frame {}", frame_count)

    finally:
        stop_event.set()
        for t in (reader, embedder, writer):
            t.join(timeout=5)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    run()
