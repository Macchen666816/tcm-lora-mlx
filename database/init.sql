-- RAG 微服务数据表（库：lora）
-- 用法：mysql -uroot -p lora < database/init.sql
-- 服务启动时也会自动执行本文件的等价建表语句（幂等）。
--
-- 变更记录（2026-09-16）：知识文档表由 `knowledge_documents` 升级为
-- `rag_stance_documents`，新增 `stance` 立场列（aligned/ambiguous/opposed），
-- 以支持「RAG 资料与微调立场同向/模糊/反向」的消融实验。旧表已删除。

CREATE DATABASE IF NOT EXISTS lora DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE lora;

-- 旧的知识文档表（无立场字段）：按新方案删除
DROP TABLE IF EXISTS knowledge_documents;

-- 三立场 RAG 知识文档（向量化前的原文）
CREATE TABLE IF NOT EXISTS rag_stance_documents (
    id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    external_id      VARCHAR(128)    NOT NULL COMMENT '业务侧唯一 ID（如 aligned-0001-xxxx）',
    stance           ENUM('aligned','neutral','opposed') NOT NULL
                     COMMENT '与微调立场的关系：同向(安全对齐)/中立(无安全拦截)/反向(恶意诱导)',
    title            VARCHAR(512)    NOT NULL COMMENT '文档标题（原问题）',
    content          TEXT            NOT NULL COMMENT '文档正文（向量化前的原文）',
    topic            VARCHAR(64)     NOT NULL DEFAULT '' COMMENT '主题标签',
    origin           VARCHAR(128)    NOT NULL DEFAULT '' COMMENT '来源文件',
    origin_id        VARCHAR(128)    NOT NULL DEFAULT '' COMMENT '源记录 ID（可追溯）',
    origin_split     VARCHAR(32)     NOT NULL DEFAULT '' COMMENT '源切分 train/test/valid',
    exclusion_reason VARCHAR(255)    NOT NULL DEFAULT '' COMMENT '被数据审计剔除的原因（立场冲突标签）',
    stance_note      VARCHAR(255)    NOT NULL DEFAULT '' COMMENT '立场标注说明',
    risk_level       VARCHAR(16)     NOT NULL DEFAULT '' COMMENT '风险等级 safe/medium/high',
    intent_tag       VARCHAR(64)     NOT NULL DEFAULT '' COMMENT '反向组的对抗注入标签（强对抗为组合标签）',
    adversarial_strength VARCHAR(16) NOT NULL DEFAULT '' COMMENT '对抗强度 strong/moderate',
    paired_id        CHAR(16)        NOT NULL DEFAULT '' COMMENT '中立/反向同题配对 ID',
    content_hash     CHAR(16)        NOT NULL DEFAULT '' COMMENT '正文哈希，用于幂等导入',
    enabled          TINYINT(1)      NOT NULL DEFAULT 1,
    created_at       TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_external_id (external_id),
    KEY idx_stance (stance),
    KEY idx_topic (topic),
    KEY idx_enabled (enabled)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = '三立场 RAG 知识文档（消融实验用）';

-- 每次 query 的链路 trace（含本次使用的 RAG 立场，便于消融实验归因）
CREATE TABLE IF NOT EXISTS query_traces (
    id                CHAR(36)     NOT NULL COMMENT 'trace UUID',
    query_text        TEXT         NOT NULL COMMENT '用户原始问题',
    augmented_prompt  MEDIUMTEXT   NULL COMMENT 'RAG 增强后送入 LLM 的完整提示词',
    rag_enabled       TINYINT(1)   NOT NULL DEFAULT 1,
    rag_status        VARCHAR(32)  NOT NULL DEFAULT 'retrieved' COMMENT 'retrieved/empty/disabled',
    rag_stance        VARCHAR(16)  NOT NULL DEFAULT 'all' COMMENT '本次检索启用的立场',
    rag_latency_ms    INT          NOT NULL DEFAULT 0,
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_created_at (created_at),
    KEY idx_stance (rag_stance)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci
  COMMENT = 'RAG 链路 trace';

-- trace 对应的检索明细
CREATE TABLE IF NOT EXISTS rag_retrievals (
    id                     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    trace_id               CHAR(36)     NOT NULL,
    rank_no                INT          NOT NULL COMMENT '相似度排名（1 起）',
    document_external_id   VARCHAR(128) NOT NULL,
    title                  VARCHAR(512) NOT NULL,
    source                 VARCHAR(255) NOT NULL DEFAULT '',
    stance                 VARCHAR(16)  NOT NULL DEFAULT '' COMMENT '命中文档的立场',
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
