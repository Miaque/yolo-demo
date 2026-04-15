# pipeline/face_api.py
"""人脸特征提取 API 客户端。"""

import requests
from loguru import logger

from config import settings
from schemas.face_feature import FaceFeatureRequest, FaceFeatureResponse


class FaceApiClient:
    """封装人脸特征提取 HTTP 接口的客户端（同步）。"""

    def __init__(
        self,
        base_url: str = settings.FACE_API_BASE_URL,
        timeout: float = settings.FACE_API_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def get_face_feature(self, img_base64: str) -> FaceFeatureResponse:
        """调用 getFaceFeature 接口，传入图片 Base64，返回人脸特征结果。

        Args:
            img_base64: 图片的 Base64 编码字符串。

        Returns:
            FaceFeatureResponse: 解析后的响应数据。

        Raises:
            requests.HTTPError: 当 HTTP 状态码非 2xx 时抛出。
        """
        request_body = FaceFeatureRequest(imgBase64=img_base64)

        logger.info("FaceApiClient: POST /face/getFaceFeature")

        resp = self._session.post(
            f"{self._base_url}/face/getFaceFeature",
            json=request_body.model_dump(by_alias=True),
            timeout=self._timeout,
        )
        resp.raise_for_status()

        response = FaceFeatureResponse.model_validate(resp.json())
        logger.info(
            "FaceApiClient: statusCode={} faces={} time={}ms",
            response.status_code,
            sum(len(faces) for faces in response.data),
            response.time,
        )
        return response

    def close(self) -> None:
        """关闭底层 HTTP Session。"""
        self._session.close()

    def __enter__(self) -> "FaceApiClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
