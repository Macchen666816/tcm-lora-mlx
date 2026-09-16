-- RAG 微服务数据表（库：lora）
-- 用法：mysql -uroot -p lora < database/init.sql
-- 服务启动时若检测到库/表缺失也会自动执行本文件的等价建表语句。

CREATE DATABASE IF NOT EXISTS lora DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE lora;

-- 向量化前的知识文档（RAG 知识库的持久层）
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    external_id   VARCHAR(128)    NOT NULL COMMENT '业务侧唯一 ID（如 jsonl 里的 id）',
    title         VARCHAR(512)    NOT NULL COMMENT '文档标题（通常为问题）',
    content       TEXT            NOT NULL COMMENT '文档正文（向量化前的原文）',
    source        VARCHAR(255)    NOT NULL DEFAULT 'manual' COMMENT '来源（文件名#行 等）',
    metadata      JSON            NULL COMMENT '扩展元数据（topic/category 等）',
    content_hash  CHAR(64)        NOT NULL COMMENT 'content 的 SHA-256，用于幂等导入',
    enabled       TINYINT(1)      NOT NULL DEFAULT 1 COMMENT '是否参与检索',
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_external_id (external_id),
    KEY idx_enabled (enabled),
    KEY idx_content_hash (content_hash)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = 'RAG 知识文档（向量化前）';

-- 每次 query 的链路 trace：原始问题 + RAG 增强后的提示词
CREATE TABLE IF NOT EXISTS query_traces (
    id                CHAR(36)     NOT NULL COMMENT 'trace UUID',
    query_text        TEXT         NOT NULL COMMENT '用户原始问题',
    augmented_prompt  MEDIUMTEXT   NULL COMMENT 'RAG 增强后送入 LLM 的完整提示词',
    rag_enabled       TINYINT(1)   NOT NULL DEFAULT 1,
    rag_status        VARCHAR(32)  NOT NULL DEFAULT 'retrieved' COMMENT 'retrieved/empty/disabled',
    rag_latency_ms    INT          NOT NULL DEFAULT 0,
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_created_at (created_at)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = 'RAG 链路 trace';

-- trace 对应的检索明细（命中了哪些知识文档、相似度多少）
CREATE TABLE IF NOT EXISTS rag_retrievals (
    id                     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id               CHAR(36)     NOT NULL,
    rank_no                INT          NOT NULL COMMENT '相似度排名（1 起）',
    document_external_id   VARCHAR(128) NOT NULL,
    title                  VARCHAR(512) NOT NULL,
    source                 VARCHAR(255) NOT NULL DEFAULT '',
    score                  DOUBLE       NOT NULL COMMENT '余弦相似度',
    excerpt                TEXT         NULL COMMENT '命中内容（截断）',
    PRIMARY KEY (id),
    KEY idx_trace (trace_id),
    CONSTRAINT fk_retrieval_trace FOREIGN KEY (trace_id) REFERENCES query_traces (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = 'RAG 检索明细';

-- LLM（同伴电脑上的 LoRA）返回结果；inference_mode 标记真实/降级模拟
CREATE TABLE IF NOT EXISTS model_outputs (
    id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id          CHAR(36)    NOT NULL,
    variant           VARCHAR(16) NOT NULL COMMENT 'base/lora',
    response          MEDIUMTEXT  NOT NULL COMMENT '模型回答',
    elapsed_seconds   DOUBLE      NOT NULL DEFAULT 0,
    character_count   INT         NOT NULL DEFAULT 0,
    inference_mode    VARCHAR(32) NOT NULL DEFAULT 'unknown' COMMENT 'remote/degraded-mock',
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_trace_variant (trace_id, variant),
    CONSTRAINT fk_output_trace FOREIGN KEY (trace_id) REFERENCES query_traces (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = 'LLM 输出记录（含降级标记）';
