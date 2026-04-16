from pydantic_settings import BaseSettings, SettingsConfigDict


class InputSettings(BaseSettings):
    """输入/输出相关配置"""

    model_config = SettingsConfigDict(env_file=".env")

    RTSP_INPUT: str = "rtsp://localhost:8554/cam"
    INPUT_WIDTH: int = 1280
    INPUT_HEIGHT: int = 720
    OUTPUT_DIR: str = "output"
    OUTPUT_BITRATE: str = "800k"


class DetectionSettings(BaseSettings):
    """YOLO 检测相关配置"""

    # 检测后端："model" 使用本地 YOLO 两阶段模型，"api" 使用人脸特征提取 HTTP 接口
    DETECTOR_BACKEND: str = "model"

    # YOLO 模型路径
    # yolo11n.pt 会由 ultralytics 自动下载
    # yolov8n-face.pt 需手动下载：
    # https://github.com/akanametov/yolo-face/releases
    DETECT_INTERVAL: int = 5  # 每隔几帧触发 YOLO 检测
    PERSON_MODEL: str = "models/yolo11s.pt"
    FACE_MODEL: str = "models/yolov11s-face.pt"


class TrackingSettings(BaseSettings):
    """ByteTrack 跟踪相关配置"""

    BYTETRACK_MAX_AGE: int = 30  # track 消失多少帧后视为已移除


class FaceSettings(BaseSettings):
    """人脸识别相关配置"""

    FACE_MIN_HEIGHT: int = 40  # 上半身裁剪区域最小高度（像素）
    FACE_CROP_MARGIN: float = 0.2  # InsightFace 裁剪时的外扩比例
    INSIGHTFACE_MODEL: str = "buffalo_sc"
    INSIGHTFACE_DET_SIZE: tuple[int, int] = (320, 320)
    FACES_DIR: str = "faces"
    RECOGNITION_THRESHOLD: float = 0.4
    ALIGNMENT_BACKEND: str = "insightface"  # "insightface" | "api"


class QueueSettings(BaseSettings):
    """队列大小配置"""

    FRAME_QUEUE_SIZE: int = 2
    FACE_CROP_QUEUE_SIZE: int = 4
    OUTPUT_QUEUE_SIZE: int = 8


class NetworkSettings(BaseSettings):
    """网络/重试相关配置"""

    FFMPEG_RETRY_MAX: int = 5
    FFMPEG_RETRY_DELAY: int = 3  # 断流重试间隔（秒）
    FACE_API_BASE_URL: str = "http://localhost:18088"
    FACE_API_TIMEOUT: float = 30.0


class Settings(
    InputSettings,
    DetectionSettings,
    TrackingSettings,
    FaceSettings,
    QueueSettings,
    NetworkSettings,
):
    """统一配置入口，从 .env 文件和环境变量读取"""

    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )


settings = Settings()
