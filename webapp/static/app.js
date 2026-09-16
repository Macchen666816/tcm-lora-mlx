const questionInput = document.querySelector("#question");
const charCount = document.querySelector("#charCount");
const clearButton = document.querySelector("#clearButton");
const compareButton = document.querySelector("#compareButton");
const buttonText = document.querySelector("#buttonText");
const maxTokensSelect = document.querySelector("#maxTokens");
const errorMessage = document.querySelector("#errorMessage");
const runtimeStatus = document.querySelector("#runtimeStatus");
const runtimeText = document.querySelector("#runtimeText");

let running = false;

function updateCharacterCount() {
  charCount.textContent = `${questionInput.value.length} / 1000`;
}

function setRuntime(state, message) {
  runtimeStatus.classList.remove("ready", "error");
  if (state) runtimeStatus.classList.add(state);
  runtimeText.textContent = message;
}

function setLoading(variant, message) {
  const output = document.querySelector(`#${variant}Output`);
  const panel = document.querySelector(`#${variant}Panel`);
  const copyButton = panel.querySelector(".copy-button");
  copyButton.hidden = true;
  output.dataset.raw = "";
  output.innerHTML = `
    <div class="loading-state">
      <span class="spinner" aria-hidden="true"></span>
      <span>${message}</span>
    </div>`;
  document.querySelector(`#${variant}Meta`).innerHTML = `
    <span>状态：生成中</span><span>耗时：—</span><span>字数：—</span>`;
}

function setWaiting(variant) {
  const output = document.querySelector(`#${variant}Output`);
  output.innerHTML = `
    <div class="loading-state">
      <span class="spinner" aria-hidden="true"></span>
      <span>等待基座模型完成</span>
    </div>`;
  document.querySelector(`#${variant}Meta`).innerHTML = `
    <span>状态：排队中</span><span>耗时：—</span><span>字数：—</span>`;
}

function setResult(variant, result) {
  const output = document.querySelector(`#${variant}Output`);
  const panel = document.querySelector(`#${variant}Panel`);
  const copyButton = panel.querySelector(".copy-button");
  output.textContent = result.response || "模型没有返回文字。";
  output.dataset.raw = result.response || "";
  copyButton.hidden = !result.response;
  document.querySelector(`#${variant}Meta`).innerHTML = `
    <span>状态：完成</span>
    <span>耗时：${result.elapsed_seconds.toFixed(2)} 秒</span>
    <span>字数：${result.character_count}</span>`;
}

function setPanelError(variant, message) {
  const output = document.querySelector(`#${variant}Output`);
  output.textContent = message;
  output.dataset.raw = "";
  document.querySelector(`#${variant}Meta`).innerHTML = `
    <span>状态：失败</span><span>耗时：—</span><span>字数：—</span>`;
}

async function generate(variant, question, maxTokens) {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, variant, max_tokens: maxTokens }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "本地模型请求失败");
  return payload;
}

async function runComparison() {
  if (running) return;
  const question = questionInput.value.trim();
  if (!question) {
    errorMessage.textContent = "请输入一个中医问题。";
    questionInput.focus();
    return;
  }

  running = true;
  errorMessage.textContent = "";
  compareButton.disabled = true;
  buttonText.textContent = "正在生成";
  setLoading("base", "正在加载并运行基座模型");
  setWaiting("lora");

  const maxTokens = Number(maxTokensSelect.value);
  try {
    const baseResult = await generate("base", question, maxTokens);
    setResult("base", baseResult);
    setLoading("lora", "正在加载并运行 LoRA 模型");
  } catch (error) {
    setPanelError("base", error.message);
    setLoading("lora", "正在尝试运行 LoRA 模型");
  }

  try {
    const loraResult = await generate("lora", question, maxTokens);
    setResult("lora", loraResult);
    setRuntime("ready", "两套本地模型已加载");
  } catch (error) {
    setPanelError("lora", error.message);
    errorMessage.textContent = error.message;
  } finally {
    running = false;
    compareButton.disabled = false;
    buttonText.textContent = "再次对比";
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const health = await response.json();
    const loaded = health.loaded_variants.length;
    setRuntime("ready", loaded === 2 ? "两套本地模型已加载" : "本地服务已就绪");
  } catch {
    setRuntime("error", "本地服务未连接");
  }
}

questionInput.addEventListener("input", updateCharacterCount);
questionInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runComparison();
});

clearButton.addEventListener("click", () => {
  questionInput.value = "";
  updateCharacterCount();
  questionInput.focus();
});

compareButton.addEventListener("click", runComparison);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    updateCharacterCount();
    questionInput.focus();
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const output = document.querySelector(`#${button.dataset.copy}`);
    await navigator.clipboard.writeText(output.dataset.raw || output.textContent);
    const original = button.textContent;
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = original; }, 1200);
  });
});

updateCharacterCount();
checkHealth();
