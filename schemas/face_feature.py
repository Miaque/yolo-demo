# schemas/face_feature.py
"""人脸特征提取接口的请求/响应模型。"""

from pydantic import BaseModel, Field


class FaceFeatureRequest(BaseModel):
    """获取人脸特征请求。"""

    img_base64: str = Field(..., alias="imgBase64", description="图片 Base64 编码字符串")


class FacePosition(BaseModel):
    """人脸位置信息。"""

    x1: int = Field(..., description="人脸框左上角 X 坐标")
    x2: int = Field(..., description="人脸框右下角 X 坐标")
    y1: int = Field(..., description="人脸框左上角 Y 坐标")
    y2: int = Field(..., description="人脸框右下角 Y 坐标")
    probability: float = Field(..., description="人脸检测置信度 (0~1)")
    score: float = Field(0.0, description="人脸评分（预留字段，默认 0）")
    points: list[int] = Field(
        default_factory=list,
        description="5 个人脸关键点坐标，依次为 [左眼x, 左眼y, 右眼x, 右眼y, 鼻尖x, 鼻尖y, 左嘴角x, 左嘴角y, 右嘴角x, 右嘴角y]",
    )
    angle: float = Field(0.0, description="人脸角度（度）")


class RaceInfo(BaseModel):
    """种族信息，键为人种类别编号（字符串），值为置信度。"""

    race_0: float = Field(0.0, alias="0", description="黄色人种置信度")
    race_1: float = Field(0.0, alias="1", description="白色人种置信度")
    race_2: float = Field(0.0, alias="2", description="黑色人种置信度")


class SunglassesSurgicalmask(BaseModel):
    """墨镜/口罩佩戴信息。键为类别编号（字符串），值为置信度。"""

    sunglasses: float = Field(0.0, alias="0", description="佩戴墨镜置信度")
    surgicalmask: float = Field(0.0, alias="1", description="佩戴口罩置信度")


class FaceFeatureData(BaseModel):
    """单张人脸的特征提取结果。"""

    face_img: str = Field("", alias="faceImg", description="人脸裁剪图片 Base64（可能为空）")
    feature: str = Field(..., description="人脸特征向量（Base64 或 HEX 编码字符串）")
    face_pos: FacePosition = Field(..., alias="facePos", description="人脸位置信息")
    sex: int = Field(..., description="性别 (0=女, 1=男)")
    age: int = Field(..., description="估计年龄")
    front: float = Field(..., description="正脸置信度 (0~1，越高越正)")
    race: RaceInfo = Field(default_factory=RaceInfo, description="种族置信度")
    sunglasses_surgicalmask: SunglassesSurgicalmask = Field(
        default_factory=SunglassesSurgicalmask,
        alias="sunglassesSurgicalmask",
        description="墨镜/口罩佩戴置信度",
    )
    quality_evaluation: float = Field(
        0.0,
        alias="qualityEvaluation",
        description="人脸质量评估分数 (0~1，越高越好)",
    )
    beard: float = Field(0.0, description="胡须置信度")
    glasses: float = Field(0.0, description="眼镜置信度")
    feature_score: float = Field(0.0, alias="featureScore", description="特征质量分数")
    feature_gender: float = Field(0.0, alias="featureGender", description="特征性别分数")
    has_human_face: float = Field(
        0.0, alias="hasHumanFace", description="是否包含人脸的置信度 (0~1)"
    )


class FaceFeatureResponse(BaseModel):
    """获取人脸特征响应。"""

    detail: str = Field(..., description="响应详情描述")
    messages: str = Field(..., description="响应消息")
    status_code: int = Field(..., alias="statusCode", description="HTTP 状态码")
    data: list[list[FaceFeatureData]] = Field(
        default_factory=list,
        description="人脸特征数据，外层为图片列表，内层为每张图片中检测到的人脸列表",
    )
    time: int = Field(0, description="接口处理耗时（毫秒）")
