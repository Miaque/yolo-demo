# config.py
RTSP_INPUT: str = "rtsp://192.168.50.48:8554/cam"
OUTPUT_DIR: str = "output"

INPUT_WIDTH: int = 640
INPUT_HEIGHT: int = 360

DETECT_INTERVAL: int = 5          # 每隔几帧触发 YOLO 检测
BYTETRACK_MAX_AGE: int = 30       # track 消失多少帧后视为已移除

FACE_MIN_HEIGHT: int = 40         # 上半身裁剪区域最小高度（像素）
FACE_CROP_MARGIN: float = 0.2     # InsightFace 裁剪时的外扩比例

FRAME_QUEUE_SIZE: int = 2
FACE_CROP_QUEUE_SIZE: int = 4
OUTPUT_QUEUE_SIZE: int = 2

FFMPEG_RETRY_MAX: int = 5
FFMPEG_RETRY_DELAY: int = 3       # 断流重试间隔（秒）

OUTPUT_BITRATE: str = "800k"

# YOLO 模型路径
# yolo11n.pt 会由 ultralytics 自动下载
# yolov8n-face.pt 需手动下载：
#   https://github.com/akanametov/yolo-face/releases
PERSON_MODEL: str = "yolo11n.pt"
FACE_MODEL: str = "yolov8n-face.pt"

# InsightFace 模型名称（buffalo_sc 为轻量亚洲优化版）
INSIGHTFACE_MODEL: str = "buffalo_sc"
INSIGHTFACE_DET_SIZE: tuple[int, int] = (320, 320)

FACES_DIR: str = "faces"
RECOGNITION_THRESHOLD: float = 0.4
