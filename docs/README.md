# 文档索引

文档按“当前结论优先、历史过程随后”的顺序阅读。历史记录用于解释研究如何演进，不应覆盖最终报告中的最新口径。

## 首选入口

1. [`项目全流程复盘与最终报告.md`](项目全流程复盘与最终报告.md)：最终结论、完整时间线、失败与方法学限制。
2. [`../项目交接说明.md`](../项目交接说明.md)：运行状态、关键路径和交接清单。
3. [`../README.md`](../README.md)：安装、训练、Web 与 RAG 快速上手。

## 数据与训练

- [`数据清洗答辩报告.md`](数据清洗答辩报告.md)：数据审计、分类、隔离和防泄漏。
- [`LoRA训练与部署讲解.md`](LoRA训练与部署讲解.md)：LoRA 原理、参数、模型文件关系和复现命令。
- [`安全能力内化LoRA训练与评测报告.md`](安全能力内化LoRA训练与评测报告.md)：新版安全训练、检查点选择和有限行为评测。

## 理论讨论

- [`LoRA安全能力与系统提示词依赖性讨论记录.md`](LoRA安全能力与系统提示词依赖性讨论记录.md)：提示词条件依赖、`mask_prompt` 和安全内化。
- [`LoRA与RAG输入空间及冲突机制讨论记录.md`](LoRA与RAG输入空间及冲突机制讨论记录.md)：参数增量与检索上下文的交互。

## RAG 与 Web

- [`RAG接入实现说明.md`](RAG接入实现说明.md)：Mac Web 后端的实际接入、缓存、降级和归因保护。
- [`同伴接入指南.md`](同伴接入指南.md)：主服务与 RAG 微服务的接口清单。
- [`检索质量实测报告.md`](检索质量实测报告.md)：纯向量失败及 BM25/RRF 修复过程。
- [`三立场消融实验设计.md`](三立场消融实验设计.md)：aligned / neutral / opposed 数据和实验变量。
- [`三立场查询样例.md`](三立场查询样例.md)、[`三立场示例query.md`](三立场示例query.md)：不同阶段的查询示例。
- [`前端与后端改进方案.md`](前端与后端改进方案.md)：早期界面与服务设计记录。

## 协作过程归档

[`collaboration/`](collaboration/) 保存同伴交接、待办和早期评分方案。这些文件属于推进历史，不是当前最终结论：

- [`collaboration/早期RAG接入指南.md`](collaboration/早期RAG接入指南.md)
- [`collaboration/队友待办汇总.md`](collaboration/队友待办汇总.md)
- [`collaboration/RAG侧待办汇总.md`](collaboration/RAG侧待办汇总.md)
- [`collaboration/主服务调整说明.md`](collaboration/主服务调整说明.md)
- [`collaboration/拉回力打分与可视化方案.md`](collaboration/拉回力打分与可视化方案.md)

## 结果入口

- [`../evaluation/RESULTS.md`](../evaluation/RESULTS.md)：模型训练与安全评测汇总。
- [`../evaluation/SAFETY_2X2_REVIEW.md`](../evaluation/SAFETY_2X2_REVIEW.md)：96 条新旧 LoRA 人工复核。
- [`../evaluation/results/README.md`](../evaluation/results/README.md)：原始结果、全量 RAG 和 NLP 输出索引。
