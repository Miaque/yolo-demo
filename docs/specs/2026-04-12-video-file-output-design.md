# 视频文件输出设计

## Context

当前 WriterThread 通过 FFmpeg 将标注帧推送到 RTSP (`rtsp://localhost:8554/processed`)。改为直接保存到本地 `output/` 目录下的 mp4 文件，不再依赖 RTSP server。

## 决策

- **方案 A（FFmpeg pipe）**：保持当前 WriterThread 架构，仅改 FFmpeg 输出目标为文件
- **保存方式**：单文件，运行期间所有帧写入同一个 mp4
- **编码质量**：保持 ultrafast preset

## 改动清单

### `config.py`
- 移除 `RTSP_OUTPUT`
- 新增 `OUTPUT_DIR: str = "output"`

### `pipeline/writer.py`
- `_build_write_cmd` → `_build_file_cmd(output_path)`：移除 `-f rtsp -rtsp_transport tcp`，直接输出 mp4
- `WriterThread.__init__`：移除 `rtsp_url`，改为 `output_path: str`；启动时 `os.makedirs` 确保目录存在
- `run()`：使用 `_build_file_cmd(self.output_path)`

### `main.py`
- 用 `datetime` 生成文件名 `output/output_YYYYmmdd_HHMMSS.mp4`
- 创建 `WriterThread(output_queue, stop_event, output_path=output_path)`
- 日志中打印输出文件路径

### `.gitignore`
- 添加 `output/`

## 数据流

```
ReaderThread (RTSP → frame_queue)
    ↓
主循环 (检测 + 追踪 + 标注)
    ↓
WriterThread (output_queue → output/output_xxx.mp4)
```

输入侧不变，仅 WriterThread 的 FFmpeg 输出目标从 RTSP 改为本地 mp4 文件。
