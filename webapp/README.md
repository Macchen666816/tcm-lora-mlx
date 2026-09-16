# 中医模型对比 Web 应用

从克隆后的本仓库根目录启动：

```bash
.venv-mlx/bin/python webapp/server.py --port 8088
```

浏览器访问 `http://127.0.0.1:8088`。

服务使用 Python 标准库提供网页和 API，不需要额外安装 FastAPI、Flask 或 Node 依赖。第一次分别调用基座和 LoRA 时会加载模型，之后会复用内存中的实例。

接口：

- `GET /api/health`：服务、设备和模型加载状态。
- `GET /api/info`：展示用的模型与训练指标。
- `POST /api/generate`：请求体包含 `question`、`variant` 和 `max_tokens`。

两套模型固定使用同一份 `evaluation/SAFETY_SYSTEM_PROMPT.txt`。该应用仅用于课程实验，不可作为诊疗、处方或剂量建议工具。
