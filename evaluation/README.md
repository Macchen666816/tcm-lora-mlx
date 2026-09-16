# 独立安全评测集

`adversarial_safety.jsonl` 是手工设计的提示词测试集，不来自原始训练或测试数据，也不会进入 LoRA 训练。

每条记录包含：

- `question`：对抗问题；
- `expected_behaviors`：回答应该体现的行为；
- `prohibited_behaviors`：回答中不应出现的行为。

它用于比较基座模型与微调模型的安全行为，不能用于判断具体中医知识的医学正确性。评测时应固定模型版本、系统提示语和生成参数，并保存原始回答。
