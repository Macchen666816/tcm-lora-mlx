# 数据审计输出

此目录由 `scripts/audit_dataset.py` 生成，用于在不修改原始 `data/` 的前提下进行全量分类和风险标记。

运行方式：

```bash
python3 scripts/audit_dataset.py
```

注意：候选问答集只经过确定性规则筛选，不代表其中的医学事实已经核验，也不是最终训练集。
