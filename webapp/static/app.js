const questionInput = document.querySelector("#question");
const charCount = document.querySelector("#charCount");
const clearButton = document.querySelector("#clearButton");
const compareButton = document.querySelector("#compareButton");
const buttonText = document.querySelector("#buttonText");
const maxTokensSelect = document.querySelector("#maxTokens");
const ragTopKSelect = document.querySelector("#ragTopK");
const errorMessage = document.querySelector("#errorMessage");
const runtimeStatus = document.querySelector("#runtimeStatus");
const runtimeText = document.querySelector("#runtimeText");
const ragStatus = document.querySelector("#ragStatus");
const ragStatusText = document.querySelector("#ragStatusText");
const summaryRag = document.querySelector("#summaryRag");
const summarySystem = document.querySelector("#summarySystem");
const matrixConsistency = document.querySelector("#matrixConsistency");
const pretestStatus = document.querySelector("#pretestStatus");

const SYSTEM_PRESET = "role_only";
const MAX_QUESTION_CHARS = 4000;
const PANELS = [
  { id: "baseNone", variant: "base", stance: "none", useRag: false, column: "无 RAG", tag: "BASE" },
  { id: "baseAligned", variant: "base", stance: "aligned", useRag: true, column: "正向引导", tag: "BASE" },
  { id: "baseOpposed", variant: "base", stance: "opposed", useRag: true, column: "负向误导", tag: "BASE" },
  { id: "loraNone", variant: "lora", stance: "none", useRag: false, column: "无 RAG", tag: "LORA" },
  { id: "loraAligned", variant: "lora", stance: "aligned", useRag: true, column: "正向引导", tag: "LORA" },
  { id: "loraOpposed", variant: "lora", stance: "opposed", useRag: true, column: "负向误导", tag: "LORA" },
];
const PRETEST_PANELS = [
  { id: "legacyTraining", variant: "legacy_lora", system: "training", useRag: false, stance: "none", column: "45 字训练提示", tag: "LEGACY", kind: "pretest" },
  { id: "legacyRole", variant: "legacy_lora", system: "role_only", useRag: false, stance: "none", column: "10 字中性提示", tag: "LEGACY", kind: "pretest" },
  { id: "internalizedTraining", variant: "lora", system: "training", useRag: false, stance: "none", column: "45 字训练提示", tag: "NEW LORA", kind: "pretest" },
  { id: "internalizedRole", variant: "lora", system: "role_only", useRag: false, stance: "none", column: "10 字中性提示", tag: "NEW LORA", kind: "pretest" },
];
const ALL_PANELS = [...PANELS, ...PRETEST_PANELS];
const RAG_STATES = {
  retrieved: { label: "retrieved", tone: "ok" },
  empty: { label: "empty", tone: "muted" },
  degraded: { label: "degraded", tone: "warn" },
  disabled: { label: "disabled", tone: "muted" },
  off: { label: "disabled", tone: "muted" },
};

let running = false;
let latestResults = {};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function panelMarkup(panel) {
  const modelLabel = panel.tag || (panel.variant === "base" ? "BASE" : "LORA");
  const modelClass = panel.variant === "base"
    ? "base"
    : panel.variant === "legacy_lora" ? "legacy" : "lora";
  const danger = panel.stance === "opposed"
    ? '<span class="danger-mark">对抗性模式</span>'
    : "";
  return `
    <header class="cell-header">
      <div>
        <span class="model-tag ${modelClass}">${modelLabel}</span>
        <h2>${escapeHtml(panel.column)}</h2>
        ${danger}
      </div>
      <div class="cell-actions">
        <button type="button" data-action="expand" hidden>展开</button>
        <button type="button" data-action="copy" hidden>复制</button>
      </div>
    </header>
    <div class="cell-badges" aria-live="polite">
      <span data-badge="mode">待运行</span>
      <span data-badge="rag">—</span>
      <span data-badge="hits">命中 —</span>
    </div>
    <div class="result-body" data-output aria-live="polite">
      <div class="empty-state"><span>等待结果</span></div>
    </div>
    <footer class="result-meta">
      <span data-meta="state">待命</span>
      <span data-meta="time">耗时 —</span>
      <span data-meta="chars">字数 —</span>
    </footer>`;
}

function initializePanels() {
  ALL_PANELS.forEach((panel) => {
    document.querySelector(`#${panel.id}Panel`).innerHTML = panelMarkup(panel);
  });
}

function updateCharacterCount() {
  charCount.textContent = `${questionInput.value.length} / ${MAX_QUESTION_CHARS}`;
}

function setServiceState(element, state, message) {
  element.classList.remove("ready", "error");
  if (state) element.classList.add(state);
  element.querySelector("span:last-child").textContent = message;
}

function ragStateOf(result) {
  const value = String(result?.rag?.rag_status || "degraded").toLowerCase();
  return RAG_STATES[value] ? value : "degraded";
}

function setLoading(panel) {
  const card = document.querySelector(`#${panel.id}Panel`);
  card.querySelector("[data-output]").innerHTML =
    '<div class="loading-state"><span class="spinner" aria-hidden="true"></span><span>生成中</span></div>';
  card.querySelector('[data-meta="state"]').textContent = "生成中";
  card.querySelector('[data-meta="time"]').textContent = "耗时 —";
  card.querySelector('[data-meta="chars"]').textContent = "字数 —";
  card.querySelectorAll(".cell-actions button").forEach((button) => { button.hidden = true; });
}

function setPanelError(panel, message) {
  const card = document.querySelector(`#${panel.id}Panel`);
  const output = card.querySelector("[data-output]");
  output.textContent = message;
  output.dataset.raw = "";
  card.querySelector('[data-meta="state"]').textContent = "失败";
  card.querySelector('[data-badge="mode"]').textContent = "error";
}

function setResult(panel, result) {
  const card = document.querySelector(`#${panel.id}Panel`);
  const output = card.querySelector("[data-output]");
  const response = result.response || "模型没有返回文字。";
  output.textContent = response;
  output.dataset.raw = response;
  output.classList.toggle("collapsed", response.length > 220);

  const expand = card.querySelector('[data-action="expand"]');
  expand.hidden = response.length <= 220;
  expand.textContent = "展开";
  card.querySelector('[data-action="copy"]').hidden = !result.response;

  const mode = result.inference_mode || "unknown";
  card.querySelector('[data-badge="mode"]').textContent = mode;
  if (panel.kind === "pretest") {
    const systemBadge = card.querySelector('[data-badge="rag"]');
    systemBadge.textContent = panel.system;
    systemBadge.className = panel.system === "role_only" ? "ok" : "muted";
    card.querySelector('[data-badge="hits"]').textContent =
      `${Number(result.system_prompt?.length || 0)} 字 system`;
  } else {
    const ragState = ragStateOf(result);
    const ragInfo = RAG_STATES[ragState];
    const hitCount = Array.isArray(result.rag?.results) ? result.rag.results.length : 0;
    const ragBadge = card.querySelector('[data-badge="rag"]');
    ragBadge.textContent = ragInfo.label;
    ragBadge.className = ragInfo.tone;
    card.querySelector('[data-badge="hits"]').textContent =
      panel.useRag ? `命中 ${hitCount}` : "未检索";
  }
  card.querySelector('[data-meta="state"]').textContent = "完成";
  card.querySelector('[data-meta="time"]').textContent =
    `耗时 ${Number(result.elapsed_seconds || 0).toFixed(2)}s`;
  card.querySelector('[data-meta="chars"]').textContent =
    `字数 ${Number(result.character_count || response.length)}`;
}

async function generate(panel, question, maxTokens) {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      variant: panel.variant,
      max_tokens: maxTokens,
      use_rag: panel.useRag,
      stance: panel.stance === "none" ? "aligned" : panel.stance,
      top_k: Number(ragTopKSelect.value),
      system: panel.system || SYSTEM_PRESET,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function riskTags(item) {
  const fields = [
    ["stance", item.stance],
    ["risk", item.risk_level],
    ["strength", item.adversarial_strength],
    ["intent", item.intent_tag],
  ];
  const tags = fields
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => `<span>${escapeHtml(key)} · ${escapeHtml(value)}</span>`);
  return tags.length ? `<div class="risk-tags">${tags.join("")}</div>` : "";
}

function renderDocuments(results) {
  if (!results.length) return '<p class="empty-evidence">未返回检索文档。</p>';
  return `<ol class="document-list">${results.map((item) => `
    <li>
      <div class="document-head">
        <strong>${escapeHtml(item.index ?? "")}. ${escapeHtml(item.title || "未命名资料")}</strong>
        <span>${item.score === null || item.score === undefined ? "" : escapeHtml(item.score)}</span>
      </div>
      ${riskTags(item)}
      ${item.snippet ? `<p>${escapeHtml(item.snippet)}</p>` : ""}
      ${item.source ? `<small>${escapeHtml(item.source)}</small>` : ""}
    </li>`).join("")}</ol>`;
}

function pairStatus(stance) {
  const base = latestResults[`base${stance[0].toUpperCase()}${stance.slice(1)}`];
  const lora = latestResults[`lora${stance[0].toUpperCase()}${stance.slice(1)}`];
  if (!base || !lora) return { ok: false, text: "结果不完整" };
  const left = base.rag || {};
  const right = lora.rag || {};
  const samePrompt = base.prompt_sent === lora.prompt_sent;
  const sameTrace = (left.trace_id || null) === (right.trace_id || null);
  const sameStance = left.stance === stance && right.stance === stance;
  const degraded = ragStateOf(base) === "degraded" || ragStateOf(lora) === "degraded";
  if (degraded && samePrompt) return { ok: true, text: "降级直答 · 两模型输入一致" };
  return {
    ok: samePrompt && sameTrace && sameStance,
    text: samePrompt && sameTrace && sameStance
      ? "配对一致 · prompt 与 trace_id 相同"
      : "配对不一致 · 本列不可归因",
  };
}

function renderEvidenceColumn(stance, result, pair) {
  const target = document.querySelector(`#${stance}Evidence`);
  const status = document.querySelector(`#${stance}EvidenceStatus`);
  if (!result) {
    target.innerHTML = '<p class="empty-evidence">本列没有可用结果。</p>';
    status.textContent = "失败";
    return;
  }

  const rag = result.rag || {};
  const docs = Array.isArray(rag.results) ? rag.results : [];
  const state = ragStateOf(result);
  status.textContent = stance === "none"
    ? "disabled"
    : `${state} · ${docs.length} 条`;
  const pairLine = stance === "none" ? "基座与 LoRA 均使用原始问题" : pair.text;
  const prompt = result.prompt_sent || rag.prompt || rag.original_question || "";
  target.innerHTML = `
    <dl class="evidence-meta">
      <div><dt>状态</dt><dd>${escapeHtml(state)}</dd></div>
      <div><dt>trace_id</dt><dd>${escapeHtml(rag.trace_id || "—")}</dd></div>
      <div><dt>配对检查</dt><dd class="${pair.ok ? "ok-text" : "warn-text"}">${escapeHtml(pairLine)}</dd></div>
    </dl>
    ${stance === "none" ? "" : renderDocuments(docs)}
    <details class="prompt-detail">
      <summary>实际 user prompt</summary>
      <pre>${escapeHtml(prompt || "—")}</pre>
    </details>
    <details class="prompt-detail">
      <summary>Chat template 原始输入</summary>
      <pre>${escapeHtml(result.full_prompt || "—")}</pre>
    </details>`;
}

function renderEvidence() {
  const nonePair = { ok: true, text: "输入一致" };
  const alignedPair = pairStatus("aligned");
  const opposedPair = pairStatus("opposed");
  renderEvidenceColumn("none", latestResults.loraNone || latestResults.baseNone, nonePair);
  renderEvidenceColumn("aligned", latestResults.loraAligned || latestResults.baseAligned, alignedPair);
  renderEvidenceColumn("opposed", latestResults.loraOpposed || latestResults.baseOpposed, opposedPair);

  const alignedPrompt = latestResults.loraAligned?.prompt_sent;
  const opposedPrompt = latestResults.loraOpposed?.prompt_sent;
  const ragDegraded = [
    latestResults.baseAligned,
    latestResults.loraAligned,
    latestResults.baseOpposed,
    latestResults.loraOpposed,
  ].some((result) => result && ragStateOf(result) === "degraded");
  const separated = Boolean(alignedPrompt && opposedPrompt && alignedPrompt !== opposedPrompt);
  const ok = alignedPair.ok && opposedPair.ok && separated;
  matrixConsistency.textContent = ragDegraded
    ? "RAG 已降级 · 本轮无法验证立场隔离"
    : ok
      ? "归因检查通过 · 两列独立且各自配对一致"
      : "归因检查未通过 · 查看各列证据";
  matrixConsistency.className = `audit-state ${ok ? "ok" : "warn"}`;
}

async function runComparison() {
  if (running) return;
  const question = questionInput.value.trim();
  if (!question) {
    errorMessage.textContent = "请输入实验问题。";
    questionInput.focus();
    return;
  }
  if (question.length > MAX_QUESTION_CHARS) {
    errorMessage.textContent = `问题不能超过 ${MAX_QUESTION_CHARS} 个字符。`;
    return;
  }

  running = true;
  latestResults = {};
  errorMessage.textContent = "";
  compareButton.disabled = true;
  buttonText.textContent = "十格生成中";
  matrixConsistency.textContent = "正在核对";
  matrixConsistency.className = "audit-state";
  pretestStatus.textContent = "四格生成中";
  pretestStatus.className = "audit-state";
  ALL_PANELS.forEach(setLoading);

  const maxTokens = Number(maxTokensSelect.value);
  const jobs = ALL_PANELS.map(async (panel) => {
    try {
      const result = await generate(panel, question, maxTokens);
      latestResults[panel.id] = result;
      setResult(panel, result);
    } catch (error) {
      setPanelError(panel, error.message);
    }
  });
  await Promise.all(jobs);
  renderEvidence();
  const pretestCompleted = PRETEST_PANELS.filter((panel) => latestResults[panel.id]).length;
  pretestStatus.textContent = pretestCompleted === PRETEST_PANELS.length
    ? "4 / 4 完成 · 同题同参数"
    : `${pretestCompleted} / ${PRETEST_PANELS.length} 完成`;
  pretestStatus.className = `audit-state ${pretestCompleted === PRETEST_PANELS.length ? "ok" : "warn"}`;

  const completed = Object.keys(latestResults).length;
  if (completed < ALL_PANELS.length) {
    errorMessage.textContent = `完成 ${completed}/${ALL_PANELS.length} 格，请检查服务日志。`;
  }
  compareButton.disabled = false;
  buttonText.textContent = "再次运行";
  running = false;
}

async function checkHealth() {
  try {
    const [healthResponse, promptsResponse] = await Promise.all([
      fetch("/api/health", { cache: "no-store" }),
      fetch("/api/prompts", { cache: "no-store" }),
    ]);
    if (!healthResponse.ok || !promptsResponse.ok) throw new Error("health check failed");
    const health = await healthResponse.json();
    const prompts = await promptsResponse.json();
    const fixed = (prompts.presets || []).find((item) => item.key === SYSTEM_PRESET);
    const systemOk = prompts.default === SYSTEM_PRESET && fixed?.character_count === 10;
    summarySystem.textContent = systemOk ? "中性 · 10 字" : "配置不一致";
    summarySystem.className = systemOk ? "" : "warn-text";
    setServiceState(runtimeStatus, "ready", "本地推理可用");

    const rag = health.rag || {};
    if (rag.enabled && rag.reachable) {
      const docs = rag.document_count ? ` · ${rag.document_count} 篇` : "";
      setServiceState(ragStatus, "ready", `RAG 已连接${docs}`);
      summaryRag.textContent = rag.index_backend || "已连接";
    } else if (!rag.enabled) {
      setServiceState(ragStatus, "error", "RAG 已关闭");
      summaryRag.textContent = "已关闭";
    } else {
      setServiceState(ragStatus, "error", "RAG 不可达");
      summaryRag.textContent = "降级直答";
    }
  } catch {
    setServiceState(runtimeStatus, "error", "本地服务未连接");
    setServiceState(ragStatus, "error", "RAG 状态未知");
    summaryRag.textContent = "未知";
  }
}

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    updateCharacterCount();
    questionInput.focus();
  });
});

document.querySelector("main").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest(".result-panel");
  const output = card.querySelector("[data-output]");
  if (button.dataset.action === "expand") {
    const expanded = output.classList.toggle("expanded");
    output.classList.toggle("collapsed", !expanded);
    button.textContent = expanded ? "收起" : "展开";
    return;
  }
  if (button.dataset.action === "copy" && output.dataset.raw) {
    await navigator.clipboard.writeText(output.dataset.raw);
    const original = button.textContent;
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = original; }, 1000);
  }
});

questionInput.addEventListener("input", updateCharacterCount);
questionInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runComparison();
});
clearButton.addEventListener("click", () => {
  questionInput.value = "";
  updateCharacterCount();
  errorMessage.textContent = "";
  questionInput.focus();
});
compareButton.addEventListener("click", runComparison);

initializePanels();
updateCharacterCount();
checkHealth();
window.setInterval(checkHealth, 20000);
