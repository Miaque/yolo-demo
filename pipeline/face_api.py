# pipeline/face_api.py
"""人脸特征提取 API 客户端。"""

import httpx
from loguru import logger

from config import settings
from schemas.face_feature import FaceFeatureRequest, FaceFeatureResponse


class FaceApiClient:
    """封装人脸特征提取 HTTP 接口的客户端（异步）。"""

    def __init__(
        self,
        base_url: str = settings.FACE_API_BASE_URL,
        timeout: float = settings.FACE_API_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def get_face_feature(self, img_base64: str) -> FaceFeatureResponse:
        """调用 getFaceFeature 接口，传入图片 Base64，返回人脸特征结果。

        Args:
            img_base64: 图片的 Base64 编码字符串。

        Returns:
            FaceFeatureResponse: 解析后的响应数据。

        Raises:
            httpx.HTTPStatusError: 当 HTTP 状态码非 2xx 时抛出。
        """
        client = self._get_client()
        request_body = FaceFeatureRequest(imgBase64=img_base64)

        logger.info("FaceApiClient: POST /face/getFaceFeature")

        resp = await client.post(
            "/face/getFaceFeature",
            json=request_body.model_dump(by_alias=True),
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

    async def close(self) -> None:
        """关闭底层 HTTP 连接。"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FaceApiClient":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()
