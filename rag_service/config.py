"""RAG 微服务配置：环境变量优先，其次项目根 .env 文件，最后内置默认值。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = Path(__file__).resolve().parent


def load_env_file() -> None:
    """读取项目根 .env（不覆盖已有环境变量），便于答辩机一键启动。"""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# ---- MySQL（知识文档持久层，向量化前的原文存这里） ----
MYSQL_HOST = os.getenv("TCM_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("TCM_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("TCM_MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("TCM_MYSQL_PASSWORD", "123456")
MYSQL_DATABASE = os.getenv("TCM_MYSQL_DATABASE", "lora")
MYSQL_ENABLED = os.getenv("TCM_DB_ENABLED", "auto").lower() != "false"

# ---- 嵌入模型（本地 HuggingFace / sentence-transformers） ----
EMBED_MODEL_NAME = os.getenv(
    "TCM_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DEVICE = os.getenv("TCM_EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE = int(os.getenv("TCM_EMBED_BATCH_SIZE", "32"))

# ---- FAISS 索引持久化目录 ----
INDEX_DIR = Path(os.getenv("TCM_INDEX_DIR", str(RAG_DIR / "index")))

# ---- 检索参数 ----
DEFAULT_TOP_K = int(os.getenv("TCM_RAG_TOP_K", "3"))
# 滤除阈值：语义分低于该值「且」无词汇命中才丢弃（混合检索下可放低）
MIN_SCORE = float(os.getenv("TCM_RAG_MIN_SCORE", "0.20"))
# 混合检索：词汇路（bigram BM25）在加权 RRF 中的权重与融合常数
LEXICAL_WEIGHT = float(os.getenv("TCM_RAG_LEXICAL_WEIGHT", "4.0"))
RRF_K = int(os.getenv("TCM_RAG_RRF_K", "60"))

# ---- LLM（同伴 MacBook 上的 LoRA）----
# mode:
#   auto    先按 url 真实调用，连不上自动降级为本地模拟（默认，演示不中断）
#   webapp  调用 webapp/server.py 的 POST /api/generate
#   openai  调用 mlx_lm.server 的 OpenAI 兼容 /v1/chat/completions
#   mock    强制本地模拟（完全离线调试用）
LLM_MODE = os.getenv("TCM_LLM_MODE", "auto").lower()
LLM_URL = os.getenv("TCM_LLM_URL", "")  # 例：http://192.168.x.x:8088
LLM_TIMEOUT = float(os.getenv("TCM_LLM_TIMEOUT", "8"))
LLM_VARIANT = os.getenv("TCM_LLM_VARIANT", "lora")  # base / lora
LLM_MAX_TOKENS = int(os.getenv("TCM_LLM_MAX_TOKENS", "256"))

# ---- 服务 ----
SERVICE_HOST = os.getenv("TCM_RAG_HOST", "0.0.0.0")  # 0.0.0.0 供局域网内同伴访问
SERVICE_PORT = int(os.getenv("TCM_RAG_PORT", "8090"))

# ---- API Key（可选；为空则不鉴权，答辩演示建议留空简化流程） ----
API_KEY = os.getenv("TCM_RAG_API_KEY", "")

# ---- 知识库种子数据（首次启动导入 MySQL） ----
# 三立场 RAG 数据集：aligned / ambiguous / opposed（见 data_rag_stance/DATASET_CARD.md）
SEED_FILES = [
    PROJECT_DIR / "data_rag_stance" / "rag_stance_dataset.jsonl",
]

# ---- 消融实验：检索时启用的立场 ----
#   aligned   与微调立场同向（默认，最安全，等价于常规知识库）
#   ambiguous 立场模糊/信息不足
#   opposed   与微调立场反向（含数据审计隔离的不安全建议，仅实验用）
#   all       三立场混合
DEFAULT_STANCE = os.getenv("TCM_RAG_STANCE", "aligned")
VALID_STANCES = ("all", "aligned", "ambiguous", "opposed")
