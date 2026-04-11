### NUC IP地址
192.168.50.207

### 树莓派5 IP地址
192.168.50.48


### 树莓派IMX219摄像头推送视频流

```
rpicam-vid \
--timeout 0 \
--width 1280 \
--height 720 \
--framerate 25 \
--codec h264 \
--profile high \
--level 4.1 \
--inline \
--intra 25 \
--bitrate 2000000 \
--nopreview \
--libav-format h264 \
-o - | \
ffmpeg \
-fflags nobuffer \
-flags low_delay \
-f h264 \
-probesize 500K \
-analyzeduration 500K \
-i - \
-vcodec copy \
-f rtsp \
-rtsp_transport tcp \
rtsp://localhost:8554/cam
```

### NUC测试拉取视频流

```
ffplay rtsp://192.168.50.48:8554/cam
```

### 使用yolov11 cli测试效果

目标检测

```
yolo detect predict \
model=yolo11s.pt \
source="rtsp://192.168.50.48:8554/cam" \
classes=0 \
imgsz=960 \
conf=0.4 \
show=True \
stream_buffer=False
```

目标跟踪

```
yolo track \
model=yolo11s.pt \
source="rtsp://192.168.50.48:8554/cam" \
classes=0 \
imgsz=960 \
tracker=bytetrack.yaml \
show=True \
stream_buffer=False
```

### 使用OpenVINO加速

添加依赖


```
uv add openvino openvino-dev
```

导出openvino模型

```
yolo export \
model=yolo11s.pt \
format=openvino \
imgsz=960
```

测试目标检测


```
yolo detect predict \
model=yolo11s_openvino_model \
source="rtsp://192.168.50.48:8554/cam" \
classes=0 \
imgsz=960 \
conf=0.4 \
show=True \
stream_buffer=False
```
