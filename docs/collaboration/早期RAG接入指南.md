# 同伴接入指南（MacBook 侧，5 分钟搞定）

> 目标：让 MacBook 上的 webapp 在推理前调用 Jette 电脑上的 RAG 微服务。
> 链路：`用户提问 → webapp → [RAG 微服务 192.168.108.82:8090] → LoRA 推理 → 前端展示`

---

## 0. 前置确认

```bash
# 在 MacBook 终端执行，确认能访问到 Jette 的 RAG 服务
curl http://192.168.108.82:8090/health
```

预期返回（关键四项）：

```json
{"status":"ok","document_count":98,"index_backend":"faiss",
 "embedder_backend":"sentence-transformers/...","database":"connected"}
```

- 如果**超时/拒绝连接** → 让 Jette 以管理员身份运行 `scripts/allow_rag_port.ps1` 放行 8090；
- 确认两台机器在**同一个热点/Wi-Fi**下。

## 1. 拉取最新代码

```bash
cd <项目目录>
git pull origin main          # 拿到 rag_service/ 与 docs/
```

## 2. 改造 webapp 后端（参考实现）

> ✅ **本仓库已完成接入**，实际实现是一个超集（`RagClient`：字段兼容、同题缓存、冷却自动恢复、
> 回写、`/api/rag/health`），见 [`../RAG接入实现说明.md`](../RAG接入实现说明.md)。
> 本节保留为**接口约定 + 最小参考实现**，方便对照。

在文件顶部加入配置：

```python
import os, json, time, urllib.request

RAG_URL = os.getenv("TCM_RAG_URL", "http://192.168.108.82:8090")
RAG_AVAILABLE = True
RAG_COOLDOWN = 15.0           # 失败后冷却秒数，到期自动重试
_rag_cooldown_until = 0.0
```

在 `/api/generate` 处理逻辑中，`question` 校验之后、`RUNTIME.generate(...)` 之前插入：

```python
# 函数里要给模块级变量赋值，必须声明 global，否则会抛 UnboundLocalError
global _rag_cooldown_until

rag_payload = None
if RAG_AVAILABLE and time.monotonic() >= _rag_cooldown_until:
    try:
        request = urllib.request.Request(
            f"{RAG_URL}/prepare",
            data=json.dumps({"query": question, "top_k": 3, "enabled": True},
                            ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            rag_payload = json.loads(response.read().decode("utf-8"))
        question = rag_payload["augmented_prompt"]      # 只替换送入模型的问题
    except Exception as exc:
        print(f"RAG 不可用，降级直答：{exc}", flush=True)
        _rag_cooldown_until = time.monotonic() + RAG_COOLDOWN   # 冷却，而不是永久关闭
```

> ⚠️ 两个容易踩的点：
> 1. 早期版本写 `RAG_AVAILABLE = False` 却没写 `global`，会在读取时直接抛 `UnboundLocalError`；
> 2. 「一次失败后永久不再重试」与 §5 验收清单里「恢复服务后下一次提问重新带上 RAG」互相矛盾，
>    应该用**冷却时间**而不是永久关闭。另外界面现在是**四格并发**（基座/LoRA × 有无检索），
>    一轮发四次请求，其中两次带检索；如果两次都各自重新检索就会拿到不同的增强提示词，
>    必须对 `(query, top_k)` 做短期缓存 + 单飞，保证两个带检索的面板共用同一份结果。

然后把返回结果改为带上 RAG 信息：

```python
result = RUNTIME.generate(variant, question, max_tokens)
self._send_json({**result, "rag": rag_payload})          # 原返回结构不变，只是多一个 rag 字段
```

**注意**：`system prompt`（`evaluation/SAFETY_SYSTEM_PROMPT.txt`）保持不变，
只把 `question` 换成增强后的提示词 —— 这样 LoRA 链路本身零改动。

## 3. 环境变量与启动

本机虚拟环境建在工作区根目录（比仓库根高一层），所以从工作区根用脚本全路径启动：

```bash
cd "/Users/chenyixuan/Desktop/Study/大模型实训/实训项目选题"
export TCM_RAG_URL="http://192.168.108.82:8090"
.venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --host 0.0.0.0 --port 8088
```

若虚拟环境就在仓库根目录（队友 clone 后自建的情形），则回到原来的写法：

```bash
.venv-mlx/bin/python webapp/server.py --host 0.0.0.0 --port 8088
```

`--host 0.0.0.0` 让同一局域网内的同伴也能用浏览器打开你这台机器上的界面；只本机看用默认的 `127.0.0.1` 即可。

## 4. 前端可视化（"RAG 后提示词"面板）

> ✅ 已实现：结果区改成 **2 × 2 四宫格**（A 基座 / B LoRA / C 基座+检索 / D LoRA+检索），
> 一次提问四格同出；再配一个全局「检索增强链路」面板（命中列表 + 完整提示词 +
> trace_id / 状态 / 耗时 / C-D 一致性），四格各带一个状态小徽标，顶栏有 RAG 连通徽标。
> 原方案「每张卡片下各挂一个面板」会重复渲染同一份内容，因此改成全局一个面板。

在回答卡片下方加一个可折叠面板，渲染 `response.rag`：

| 展示区块 | 字段 |
|---|---|
| 检索命中列表（标题/相似度/来源） | `rag.results[]` |
| **增强后提示词全文**（等宽字体、可复制） | `rag.augmented_prompt` |
| 链路元信息（trace_id、rag_status、耗时 ms） | `rag.trace_id` / `rag.rag_status` / `rag.rag_latency_ms` |

状态徽标建议：`retrieved` 绿色"RAG 已增强" / `empty` 灰色"未命中" /
RAG 不可用（rag 为 null）黄色"RAG 已降级（直答）"。

## 5. 验收清单

- [ ] `curl /health` 返回 `database=connected`
- [ ] 页面提问后，回答下方能看到检索命中的资料标题
- [ ] 展开能看到完整的增强提示词
- [ ] 关掉 Jette 的 RAG 服务后提问，仍能出回答（降级徽标为黄色）
- [ ] 恢复服务后，下一次提问重新带上 RAG

## 6. 可选：回写模型输出（补全证据链）

真实 LoRA 输出后可选回写，形成 MySQL 三表串联
（`query_traces` → `rag_retrievals` → `model_outputs`）：

```bash
curl -X POST "http://192.168.108.82:8090/traces/{trace_id}/outputs" \
  -H "Content-Type: application/json" \
  -d '{"variant":"lora","response":"...","elapsed_seconds":0.6,
       "character_count":120,"inference_mode":"remote-webapp"}'
```

答辩时可直接查库展示完整链路证据。
