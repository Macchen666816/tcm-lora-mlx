# 训练数据包

此目录由 `scripts/prepare_training_data.py` 从 `data_audit/audit_records.jsonl` 生成。

完整重建：

```bash
python3 scripts/audit_dataset.py
python3 scripts/prepare_training_data.py
python3 scripts/validate_processed_data.py
```

训练优先使用 `mlx/` 下的对话格式；如改用 Transformers/PEFT，可使用 `alpaca/` 下的三字段格式。

这些数据经过自动规则清洗，但没有经过中医专业人员逐条审核。具体范围、排除策略和限制见 `DATA_CARD.md`。
