# 模型产物说明

Git 只保存最终可复现所需的 adapter、配置和关键训练日志；中间检查点与烟雾实验产物由 `.gitignore` 排除，但本机可继续保留用于复盘。

## 纳入版本控制

- `qwen25-lora-safety/`：旧版安全 LoRA，用于对照和回滚。
- `qwen25-lora-safety-internalized/`：当前候选安全内化 LoRA。
- `qwen25-lora-full-training.log`：正式领域训练日志。
- `qwen25-lora-safety-training.log`：旧版安全续训日志。
- `qwen25-lora-safety-internalized-training.log`：新版安全内化训练日志。

当前候选：

```text
qwen25-lora-safety-internalized/adapters.safetensors
SHA-256: 525b9544eb896aca16be88bd436b5c7c47f7d4787f3ace4f66d023b4ddd798b7
```

## 本机保留但不推送

- `qwen25-lora-full/` 的各训练检查点；
- `qwen25-lora-safety-internalized-v1/`、`*-v2-550/` 和 `*-cp*-eval/`；
- `*-smoke/` 与早期 DeepSeek/Qwen 烟雾实验；
- 各 checkpoint 的重复 adapter 文件。

这些文件可由配置和训练脚本重新生成，不应全部提交到 GitHub。基座模型位于仓库外部 `models/`，同样不纳入 Git。
