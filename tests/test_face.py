import os
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# [生产级组件] 多线程视频读取器
# 作用：防止主线程做 AI 推理时，摄像头/RTSP流的缓冲区塞满导致严重延迟甚至花屏
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 最小化缓冲区
        self.status, self.frame = self.capture.read()
        self.running = True

        # 开启后台线程不断读取最新帧
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while self.running:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)  # 轻微休眠，防止抢占过多 CPU

    def get_frame(self):
        return self.status, self.frame.copy() if self.status else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()


# ==========================================
# 主业务逻辑
# ==========================================
def main():
    print("⏳ 正在初始化纯 CPU 环境的模型...")

    # 1. 初始化 InsightFace
    # 强制使用纯 CPU。为了保证帧率，这里使用轻量级模型 'buffalo_sc'
    # 如果您对精度要求极高（哪怕 1 FPS 也能接受），可以改为 'buffalo_l'
    app = FaceAnalysis(
        name="buffalo_sc", root="./models", providers=["CPUExecutionProvider"]
    )

    # ctx_id=-1 代表使用 CPU，det_size 设置检测输入大小（越小越快，但漏检小脸）
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ 模型加载完成！")

    # 2. 创建人脸抓拍存储目录
    output_dir = "./captured_faces"
    os.makedirs(output_dir, exist_ok=True)

    # 3. 接入视频流 (这里用 0 代表本地测试摄像头，可替换为 'rtsp://...' 或 'video.mp4')
    video_source = 0
    camera = ThreadedCamera(video_source)

    # 4. 业务控制参数
    PROCESS_EVERY_N_FRAMES = 5  # 纯CPU下，不要每帧都算。每 5 帧做一次 AI 推理
    frame_count = 0
    last_faces = []  # 缓存上一次检测到的人脸结果，用于中间帧的平滑显示

    print("▶️ 开始处理视频流，按 'q' 退出...")

    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        status, frame = camera.get_frame()
        if not status or frame is None:
            continue

        frame_count += 1
        fps_frame_count += 1

        # ==========================================
        # 核心 AI 处理逻辑 (降频执行，保护 CPU)
        # ==========================================
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # 执行检测和识别 (提取 512 维特征向量)
            faces = app.get(frame)
            last_faces = faces  # 更新缓存结果

            # 处理抓拍逻辑
            for idx, face in enumerate(faces):
                if face.det_score < 0.5:  # 过滤低置信度
                    continue

                # 获取 512 维特征 (生产中将此变量发往 Milvus/数据库)
                # embedding = face.embedding

                # 裁剪并保存人脸照片
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(frame.shape[1], bbox[2]),
                    min(frame.shape[0], bbox[3]),
                )
                face_img = frame[y1:y2, x1:x2]

                if face_img.size > 0:
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(output_dir, f"face_{timestamp}_{idx}.jpg")
                    cv2.imwrite(save_path, face_img)

        # ==========================================
        # 画面渲染逻辑 (每帧都执行，保证视频视觉流畅)
        # ==========================================
        for face in last_faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            score = face.det_score

            if score >= 0.5:
                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 写字 (模拟身份识别结果)
                label = f"Person: {score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # 计算并显示处理帧率 (FPS)
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            current_fps = fps_frame_count / elapsed_time
            fps_start_time = time.time()
            fps_frame_count = 0
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f} (CPU Mode)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # 显示输出画面
        cv2.imshow("InsightFace - CPU Pipeline", frame)

        # 监听退出按键
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 5. 安全清理资源
    print("🛑 正在停止流处理...")
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
