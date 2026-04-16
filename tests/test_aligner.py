"""FaceAligner 单元测试。"""

import base64

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pipeline.aligner import AlignmentResult, FaceAligner, ARCFACE_REF_POINTS
from config import settings


class TestInsightFaceBackend:
    """InsightFace 后端测试。"""

    @patch("insightface.app.FaceAnalysis")
    def test_align_returns_result_on_valid_crop(self, mock_fa_cls):
        """InsightFace 返回有效 Face → 得到 AlignmentResult。"""
        mock_app = MagicMock()
        mock_face = MagicMock()
        mock_face.kps = ARCFACE_REF_POINTS.copy()  # 关键点正好是参考点
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert isinstance(result, AlignmentResult)
        assert result.aligned_face.shape == (112, 112, 3)
        assert result.embedding.shape == (512,)
        assert result.landmarks.shape == (5, 2)
        aligner.close()

    @patch("insightface.app.FaceAnalysis")
    def test_align_returns_none_on_no_face(self, mock_fa_cls):
        """InsightFace 无检测结果 → 返回 None。"""
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result is None
        aligner.close()

    @patch("insightface.app.FaceAnalysis")
    def test_aligned_face_size(self, mock_fa_cls):
        """对齐图尺寸为 112x112x3。"""
        mock_app = MagicMock()
        mock_face = MagicMock()
        mock_face.kps = ARCFACE_REF_POINTS.copy()
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((300, 300, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result.aligned_face.shape == (112, 112, 3)
        aligner.close()


class TestApiBackend:
    """API 后端测试。"""

    @patch("pipeline.face_api.FaceApiClient")
    def test_align_api_backend(self, mock_client_cls):
        """API 返回关键点和 feature → 正确解析。"""
        # 构造模拟关键点 [x0..x4, y0..y4]
        ref = ARCFACE_REF_POINTS
        points = [int(ref[i, 0]) for i in range(5)] + [int(ref[i, 1]) for i in range(5)]

        # 构造模拟 feature (base64 编码的 float32 数组)
        emb = np.ones(512, dtype=np.float32)
        feature_b64 = base64.b64encode(emb.tobytes()).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_face_data = MagicMock()
        mock_face_data.face_pos.points = points
        mock_face_data.feature = feature_b64
        mock_resp.data = [[mock_face_data]]
        mock_client = MagicMock()
        mock_client.get_face_feature.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        aligner = FaceAligner(backend="api")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert isinstance(result, AlignmentResult)
        assert result.aligned_face.shape == (112, 112, 3)
        assert result.embedding.shape == (512,)
        aligner.close()

    @patch("pipeline.face_api.FaceApiClient")
    def test_align_api_backend_no_landmarks(self, mock_client_cls):
        """API 返回空关键点 → 返回 None。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_face_data = MagicMock()
        mock_face_data.face_pos.points = []
        mock_face_data.feature = ""
        mock_resp.data = [[mock_face_data]]
        mock_client = MagicMock()
        mock_client.get_face_feature.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        aligner = FaceAligner(backend="api")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result is None
        aligner.close()


class TestGeneral:
    """通用测试。"""

    def test_invalid_backend_raises(self):
        """无效后端字符串 → ValueError。"""
        with pytest.raises(ValueError, match="不支持的 backend"):
            FaceAligner(backend="invalid")

    @patch("insightface.app.FaceAnalysis")
    def test_close_releases_resources_insightface(self, mock_fa_cls):
        """InsightFace 后端 close() 不报错。"""
        aligner = FaceAligner(backend="insightface")
        aligner.close()  # 不应抛出异常

    @patch("pipeline.face_api.FaceApiClient")
    def test_close_releases_resources_api(self, mock_client_cls):
        """API 后端 close() 调用 FaceApiClient.close()。"""
        aligner = FaceAligner(backend="api")
        aligner.close()
        mock_client_cls.return_value.close.assert_called_once()

    @patch("insightface.app.FaceAnalysis")
    def test_default_backend_from_config(self, mock_fa_cls):
        """不传 backend 参数时使用配置默认值。"""
        mock_app = MagicMock()
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner()  # 默认从 settings 读
        assert aligner._backend == settings.ALIGNMENT_BACKEND
        aligner.close()

    @patch("insightface.app.FaceAnalysis")
    def test_affine_transform_identity(self, mock_fa_cls):
        """输入关键点等于参考点时，仿射变换接近单位变换。"""
        mock_app = MagicMock()
        mock_face = MagicMock()
        mock_face.kps = ARCFACE_REF_POINTS.copy()
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        # 创建包含参考点位置的 112x112 图像
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        for pt in ARCFACE_REF_POINTS:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(crop, (x, y), 3, (255, 255, 255), -1)

        aligner = FaceAligner(backend="insightface")
        result = aligner.align(crop)

        # 参考点处画了白点，对齐后这些白点应仍在相同位置附近
        assert result.aligned_face.shape == (112, 112, 3)
        for pt in ARCFACE_REF_POINTS:
            x, y = int(pt[0]), int(pt[1])
            assert result.aligned_face[y, x, 0] > 200  # 白点应可见
        aligner.close()
