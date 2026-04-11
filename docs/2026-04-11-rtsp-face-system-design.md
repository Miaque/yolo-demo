# RTSP 实时人脸处理系统设计文档

**日期：** 2026-04-11  
**项目：** yolo-demo  
**目标平台：** CPU-only（树莓派 / 中等 x86）  
**目标人群：** 亚洲人种

---

## 1. 系统目标

基于树莓派摄像头推送的 RTSP 视频流，实时完成：

1. 人脸检测与跟踪（每 5 帧 YOLO 检测，帧间 ByteTrack 预测）
2. 新目标出现时异步提取 InsightFace 特征向量（Embedding）
3. 将标注后的视频流重新推送为 RTSP 供下游订阅

---

## 2. 技术栈

| 组件 | 版本 / 说明 |
|------|------------|
| Python | 3.11+ |
| FFmpeg | 系统安装，用于读帧和推流 |
| YOLOv8n（person） | ultralytics，标准 COCO 模型 |
| YOLOv8n-face | 专用人脸检测权重 |
| ByteTrack | boxmot 库封装 |
| InsightFace | 本地 Python SDK，buffalo_sc 模型（亚洲人种优化）|
| OpenCV | 帧格式转换与 overlay 绘制 |

---

## 3. 整体架构

单进程多线程方案：

```
┌─────────────────────────────────────────────────────┐
│                      主进程                          │
│                                                      │
│  读帧线程                                            │
│  FFmpeg subprocess（rtsp://localhost:8554/cam）      │
│      │ 原始帧（BGR bytes，640×360）                  │
│      ▼                                               │
│  frame_queue（maxsize=2，满时丢弃旧帧）              │
│      │                                               │
│      ▼                                               │
│  ┌──────────────────────────────────┐               │
│  │       主线程（Pipeline Loop）    │               │
│  │                                  │               │
│  │  每帧：ByteTrack.predict()       │               │
│  │  每 5 帧：                       │               │
│  │    YOLO person → 取上半身区域    │               │
│  │    → YOLO face → ByteTrack.update│               │
│  │                                  │               │
│  │  新 ID → face_crop_queue ────────┼──→ InsightFace│
│  │                                  │       线程     │
│  │  overlay(frame, tracks,          │       │        │
│  │          embedding_store)        │       ▼        │
│  │                                  │  embedding_store│
│  └──────────────────────────────────┘  (dict + Lock) │
│      │                                               │
│      ▼                                               │
│  output_queue（maxsize=2）                           │
│      │                                               │
│      ▼                                               │
│  推流线程                                            │
│  FFmpeg subprocess → rtsp://localhost:8554/processed │
└─────────────────────────────────────────────────────┘
```

**线程列表：**

| 线程 | 职责 |
|------|------|
| 读帧线程 | FFmpeg 解码 → frame_queue |
| 主线程 | YOLO + ByteTrack + overlay |
| InsightFace 线程 | 消费 face_crop_queue，写 embedding_store |
| 推流线程 | 消费 output_queue，FFmpeg 编码推 RTSP |

---

## 4. 模块划分

```
yolo-demo/
├── main.py                  # 入口：解析参数，启动所有线程
├── pipeline/
│   ├── __init__.py
│   ├── reader.py            # FFmpeg 读帧，写入 frame_queue
│   ├── detector.py          # YOLO person + face 两阶段检测
│   ├── tracker.py           # ByteTrack 封装，维护 track_id → bbox
│   ├── embedder.py          # InsightFace 线程，消费 face_crop_queue
│   └── writer.py            # FFmpeg 推 RTSP 输出流
├── overlay.py               # 纯函数：在帧上叠加 BBox、ID、Embedding 状态
├── state.py                 # 共享状态：embedding_store + threading.Lock
├── config.py                # 所有可配置参数
├── docs/
│   └── 2026-04-11-rtsp-face-system-design.md
└── pyproject.toml
```

---

## 5. 核心数据流与关键逻辑

### 5.1 主线程 Pipeline Loop

```
frame_count = 0
known_ids = set()

loop:
  frame ← frame_queue.get()
  frame_count += 1

  if frame_count % 5 == 0:
    persons = yolo_person.detect(frame)
    faces = []
    for person_bbox in persons:
      upper = upper_half(person_bbox)           # 取上 50% 区域
      if height(upper) < 40: continue          # 过小跳过
      face_list = yolo_face.detect(frame[upper])
      faces += to_global_coords(face_list, upper)

    track_results = tracker.update(faces, frame)
  else:
    track_results = tracker.predict()

  for track_id, bbox in track_results:
    if track_id not in known_ids:
      known_ids.add(track_id)
      face_crop = crop_with_margin(frame, bbox, margin=0.2)
      face_crop_queue.put_nowait((track_id, face_crop))  # 满则丢弃

  # 清理已消失的 track
  for removed_id in tracker.removed_ids():
    known_ids.discard(removed_id)
    with lock:
      embedding_store.pop(removed_id, None)

  annotated = overlay(frame, track_results, embedding_store)
  output_queue.put_nowait(annotated)
```

### 5.2 InsightFace 线程

```
app = insightface.app.FaceAnalysis(
    name="buffalo_sc",           # 轻量模型，亚洲人种优化
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(320, 320))

loop:
  track_id, face_crop = face_crop_queue.get()
  results = app.get(face_crop)
  if results:
    embedding = results[0].embedding            # 512-dim float32
    with lock:
      embedding_store[track_id] = embedding
```

### 5.3 上半身裁剪规则

- 取 person bbox 纵向上 50% 区域作为人脸检测区域
- 裁剪区域高度 < 40px 时跳过（距离过远，人脸过小）
- face crop 提交给 InsightFace 前添加 20% margin，改善关键点定位

---

## 6. 视频流 I/O

### 6.1 输入读帧命令

```python
[
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-rtsp_transport", "tcp",
    "-probesize", "32",
    "-analyzeduration", "0",
    "-i", "rtsp://localhost:8554/cam",
    "-vf", "scale=640:360",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-",
]
```

### 6.2 输出推流命令

```python
[
    "ffmpeg",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "640x360",
    "-r", "25",
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-b:v", "800k",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://localhost:8554/processed",
]
```

### 6.3 队列参数

| 队列 | maxsize | 满时行为 |
|------|---------|---------|
| frame_queue | 2 | 丢弃旧帧（读帧线程跳过） |
| face_crop_queue | 4 | put_nowait 失败即丢弃 |
| output_queue | 2 | put_nowait 失败即丢弃 |

---

## 7. 错误处理

### 7.1 RTSP 断流重连

```python
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds

for attempt in range(MAX_RETRIES):
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) == 0:
            break                      # 流中断，退出内循环
        frame_queue.put_nowait(frame)
    proc.wait()
    time.sleep(RETRY_DELAY)
```

### 7.2 线程异常隔离

- 每个线程主循环用 `try/except Exception` 包裹
- 捕获异常后设置 `error_event`（`threading.Event`）
- 主线程监听 `error_event`，触发时依次 join 所有线程并退出

### 7.3 InsightFace 调用失败

- `app.get()` 返回空列表：跳过，不写 `embedding_store`
- 下次同 track_id 重新出现时（ID 未从 known_ids 清除）不会重试
- 若需重试：从 `known_ids` 中移除该 ID，下次检测时重新触发

---

## 8. 性能预估（CPU-only）

| 阶段 | 耗时估算 |
|------|---------|
| YOLO person 检测（每 5 帧）| 80–150ms |
| YOLO face 检测（上半身内）| 30–60ms |
| ByteTrack 更新 | <5ms |
| InsightFace Embedding（异步）| 100–300ms |
| overlay + 推流编码 | <10ms |

- 主线程在 25fps（40ms/帧）下，非检测帧 <5ms，检测帧约 110–210ms（会造成掉帧）
- 掉帧由 frame_queue maxsize=2 的丢帧策略吸收，不影响实时性

---

## 9. 测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | detector / tracker / overlay / state 并发读写 | pytest |
| 集成测试 | 本地 .mp4 替代 RTSP 输入，端到端验证 | pytest + 本地视频 |
| 手动验证 | 接树莓派实际流，VLC 订阅 processed 流查看标注 | VLC |

```
tests/
├── test_detector.py
├── test_tracker.py
├── test_overlay.py
├── test_state.py
└── test_pipeline_integration.py
```

---

## 10. 配置参数（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| RTSP_INPUT | rtsp://localhost:8554/cam | 输入流地址 |
| RTSP_OUTPUT | rtsp://localhost:8554/processed | 输出流地址 |
| INPUT_WIDTH | 640 | 处理分辨率宽 |
| INPUT_HEIGHT | 360 | 处理分辨率高 |
| DETECT_INTERVAL | 5 | 每隔几帧触发 YOLO |
| BYTETRACK_MAX_AGE | 30 | track 消失多少帧后删除 |
| FACE_MIN_HEIGHT | 40 | 人脸区域最小高度（px） |
| FACE_CROP_MARGIN | 0.2 | InsightFace 裁剪 margin |
| FACE_CROP_QUEUE_SIZE | 4 | InsightFace 队列深度 |
| FFMPEG_RETRY_MAX | 5 | 断流最大重试次数 |
| FFMPEG_RETRY_DELAY | 3 | 重试等待秒数 |
