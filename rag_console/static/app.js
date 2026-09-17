'use strict';

const state = { documents: [], prompt: '', stanceCounts: {} };

const STANCE_LABEL = {
  aligned: { text: '同向', cls: 'aligned' },
  neutral: { text: '中立', cls: 'neutral' },
  opposed: { text: '反向', cls: 'opposed' },
};

const el = (id) => document.getElementById(id);

function stanceBadge(stance) {
  const meta = STANCE_LABEL[stance] || { text: stance || '未知', cls: '' };
  return `<span class="badge ${meta.cls}">${escapeHtml(meta.text)}</span>`;
}

function toast(message, isError = false) {
  const node = el('toast');
  node.textContent = message;
  node.style.borderColor = isError ? 'var(--err)' : 'var(--accent)';
  node.classList.remove('hidden');
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.add('hidden'), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await response.json().catch(() => ({ error: '响应不是合法 JSON' }));
  if (!response.ok) {
    throw new Error(data.error || data.hint || `请求失败 ${response.status}`);
  }
  return data;
}

/* ---------- 服务状态 ---------- */

async function loadHealth() {
  const grid = el('healthGrid');
  const errorBox = el('healthError');
  try {
    const data = await api('/health');
    errorBox.classList.add('hidden');
    const counts = data.stance_counts || {};
    const items = [
      ['服务状态', data.status === 'ok' ? '正常' : data.status, 'ok'],
      ['知识文档', `${data.document_count} 条（同向 ${counts.aligned ?? 0} / 中立 ${counts.neutral ?? 0} / 反向 ${counts.opposed ?? 0}）`, ''],
      ['数据库', data.database, data.database === 'connected' ? 'ok' : 'warn'],
      ['索引后端', data.index_backend, data.index_backend === 'faiss' ? 'ok' : 'warn'],
      ['检索方案', data.retriever || '—', ''],
      ['默认立场', data.default_stance || '—', data.default_stance === 'opposed' ? 'warn' : 'ok'],
      ['对抗性立场', data.allow_opposed ? '已开启' : '未开启（opposed/all 需开启）',
        data.allow_opposed ? 'warn' : 'ok'],
      ['嵌入模型', (data.embedder_backend || '').split('/').pop() || '—',
        (data.embedder_backend || '').includes('sentence-transformers') ? 'ok' : 'warn'],
      ['LLM 链路', data.llm_url || '—', (data.llm_url || '').includes('未配置') ? 'warn' : 'ok'],
    ];
    grid.innerHTML = items
      .map(([label, value, tone]) =>
        `<div class="stat"><span class="label">${label}</span>` +
        `<span class="value ${tone}">${escapeHtml(String(value))}</span></div>`)
      .join('');
    el('ragTarget').textContent = data.embedder_warning
      ? `⚠️ ${data.embedder_warning}`
      : 'RAG 微服务在线';
  } catch (error) {
    grid.innerHTML =
      '<div class="stat"><span class="label">服务状态</span>' +
      '<span class="value err">未连接</span></div>';
    errorBox.textContent = `${error.message}\n\n请先启动 RAG 微服务：python -m rag_service.server`;
    errorBox.classList.remove('hidden');
  }
}

/* ---------- 检索测试 ---------- */

async function runSearch() {
  const query = el('searchInput').value.trim();
  if (!query) { toast('请输入检索问题', true); return; }
  const button = el('searchBtn');
  button.disabled = true;
  button.textContent = '检索中…';
  try {
    const data = await api('/prepare', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: Number(el('topK').value) || 3,
        enabled: true,
        stance: el('searchStance').value,
      }),
    });

    el('searchMeta').textContent =
      `立场 ${data.stance} · 状态 ${data.rag_status} · 命中 ${data.results.length} 条 · ` +
      `${data.rag_latency_ms} ms · trace ${data.trace_id.slice(0, 8)}`;
    el('searchMeta').classList.remove('hidden');

    if (!data.results.length) {
      el('searchResults').innerHTML =
        '<div class="meta">没有命中任何资料（可能低于相似度阈值，或该立场下未覆盖该主题）。</div>';
    } else {
      el('searchResults').innerHTML = data.results.map((item) => {
        const dense = (item.score ?? 0).toFixed(3);
        const fused = (item.fused_score ?? 0).toFixed(4);
        const lexical = (item.lexical_score ?? 0).toFixed(1);
        return `
        <div class="result">
          <div class="result-head">
            <span class="result-title">[${item.rank}] ${stanceBadge(item.stance)} ${escapeHtml(item.title)}</span>
            <span class="score" title="向量余弦 · 融合分 · BM25词汇分">${dense} · ${fused} · ${lexical}</span>
          </div>
          <div class="result-source">${escapeHtml(item.source)}${item.topic ? ' · ' + escapeHtml(item.topic) : ''}` +
          `${item.risk_level && item.risk_level !== 'safe' ? ' · 风险 ' + escapeHtml(item.risk_level) : ''}` +
          `${item.intent_tag ? ' · 诱导 ' + escapeHtml(item.intent_tag) : ''}</div>
          <div class="result-content">${escapeHtml(item.content)}</div>
        </div>`;
      }).join('');
    }

    state.prompt = data.augmented_prompt || '';
    el('promptText').textContent = state.prompt;
    el('promptBox').classList.toggle('hidden', !state.prompt);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = '检索';
  }
}

/* ---------- 文档列表 ---------- */

function renderDocuments() {
  const keyword = el('docFilter').value.trim().toLowerCase();
  const stance = el('docStanceFilter').value;
  const list = state.documents.filter((doc) =>
    (stance === 'all' || doc.stance === stance) &&
    (!keyword ||
      doc.title.toLowerCase().includes(keyword) ||
      doc.external_id.toLowerCase().includes(keyword) ||
      (doc.source || '').toLowerCase().includes(keyword)));

  const counts = state.stanceCounts || {};
  el('docCount').textContent =
    `共 ${state.documents.length} 条（同向 ${counts.aligned ?? 0} / 模糊 ${counts.ambiguous ?? 0} / ` +
    `反向 ${counts.opposed ?? 0}）` + (keyword || stance !== 'all' ? `，当前显示 ${list.length} 条` : '');

  el('docList').innerHTML = list.length
    ? list.slice(0, 300).map((doc) => `
      <div class="doc-item">
        <div class="doc-main">
          <div class="doc-title">${stanceBadge(doc.stance)} ${escapeHtml(doc.title)}</div>
          <div class="doc-sub">${escapeHtml(doc.external_id)} · ${escapeHtml(doc.topic || '')} · ${doc.content_length} 字</div>
        </div>
        <button class="btn ghost small" data-view="${escapeHtml(doc.external_id)}">查看</button>
      </div>`).join('')
    : '<div class="meta">暂无文档。</div>';
}

async function loadDocuments() {
  try {
    const data = await api('/documents');
    state.documents = data.documents || [];
    state.stanceCounts = data.stance_counts || {};
    renderDocuments();
  } catch (error) {
    toast(error.message, true);
  }
}

async function viewDocument(externalId) {
  const doc = state.documents.find((item) => item.external_id === externalId);
  if (!doc) return;
  try {
    const data = await api('/documents?full=1');
    const full = (data.documents || []).find((item) => item.external_id === externalId);
    state.prompt =
      `【${doc.title}】\nID: ${doc.external_id}\n来源: ${doc.source}\n字数: ${doc.content_length}\n\n` +
      (full?.content || '(未能读取正文)');
    el('promptText').textContent = state.prompt;
    el('promptBox').classList.remove('hidden');
  } catch (error) {
    toast(error.message, true);
  }
}

/* ---------- 新增文档 ---------- */

async function saveDocument() {
  const payload = {
    stance: el('docStance').value,
    external_id: el('docId').value.trim(),
    title: el('docTitle').value.trim(),
    source: el('docSource').value.trim() || 'manual',
    content: el('docContent').value.trim(),
  };
  if (!payload.external_id || !payload.title || !payload.content) {
    toast('文档 ID、标题、正文均为必填', true);
    return;
  }

  const button = el('saveDoc');
  button.disabled = true;
  button.textContent = '保存中…';
  el('saveState').textContent = '';
  try {
    const data = await api('/documents', { method: 'POST', body: JSON.stringify(payload) });
    const info = data.index || {};
    el('saveState').textContent =
      `✅ 已保存到 ${payload.stance} 立场并重建索引（${info.indexed_documents ?? '?'} 条，${info.build_ms ?? '?'} ms）`;
    toast('文档已写入 MySQL，索引已重建');
    el('docId').value = '';
    el('docTitle').value = '';
    el('docSource').value = '';
    el('docContent').value = '';
    await loadDocuments();
    await loadHealth();
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = '保存并重建索引';
  }
}

async function rebuildIndex() {
  try {
    const data = await api('/index/rebuild', { method: 'POST', body: '{}' });
    toast(`索引已重建：${data.indexed_documents} 条 / ${data.build_ms} ms`);
    await loadHealth();
  } catch (error) {
    toast(error.message, true);
  }
}

/* ---------- 工具 ---------- */

function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, (char) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

/* ---------- 绑定 ---------- */

el('refreshHealth').addEventListener('click', loadHealth);
el('searchBtn').addEventListener('click', runSearch);
el('searchInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') runSearch();
});
el('copyPrompt').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(state.prompt);
    toast('提示词已复制');
  } catch {
    toast('复制失败，请手动选择文本', true);
  }
});
el('saveDoc').addEventListener('click', saveDocument);
el('clearDoc').addEventListener('click', () => {
  ['docId', 'docTitle', 'docSource', 'docContent'].forEach((id) => { el(id).value = ''; });
});
el('reloadDocs').addEventListener('click', loadDocuments);
el('rebuildIndex').addEventListener('click', rebuildIndex);
el('docFilter').addEventListener('input', renderDocuments);
el('docStanceFilter').addEventListener('change', renderDocuments);
el('docList').addEventListener('click', (event) => {
  const id = event.target?.dataset?.view;
  if (id) viewDocument(id);
});


/* ---------- 三模式并排对比 ---------- */

const CMP_MODES = [
  { stance: 'aligned', name: '正向引导' },
  { stance: 'opposed', name: '负向误导' },
];

async function runCompare() {
  const query = el('cmpInput').value.trim();
  if (!query) { toast('请输入问题', true); return; }
  const topK = Number(el('cmpTopK').value) || 2;
  const button = el('cmpBtn');
  button.disabled = true;
  button.textContent = '三种模式检索中…';

  const columns = [];
  for (const mode of CMP_MODES) {
    try {
      const data = await api('/prepare', {
        method: 'POST',
        body: JSON.stringify({ query, top_k: topK, enabled: true, stance: mode.stance }),
      });
      columns.push({ mode, data, error: '' });
    } catch (error) {
      columns.push({ mode, data: null, error: error.message });
    }
  }

  el('cmpGrid').innerHTML = columns.map(({ mode, data, error }) => {
    if (error) {
      return `<div class="cmp-col ${mode.stance}">
        <div class="cmp-head"><span class="name">${escapeHtml(mode.name)}</span>${stanceBadge(mode.stance)}</div>
        <div class="meta">调用失败：${escapeHtml(error)}</div></div>`;
    }
    const docs = (data.results || []).map((item) => `
      <div class="cmp-doc">
        <div class="t"><b>[${item.rank}]</b> ${escapeHtml(item.title)}</div>
        <div class="s">${escapeHtml(item.source)} · 风险 ${escapeHtml(item.risk_level || '-')}` +
        `${item.adversarial_strength ? ' · 强度 ' + escapeHtml(item.adversarial_strength) : ''}` +
        `${item.intent_tag ? ' · 注入 ' + escapeHtml(item.intent_tag) : ''}</div>
        <div class="c">${escapeHtml((item.content || '').slice(0, 220))}${(item.content || '').length > 220 ? '……' : ''}</div>
      </div>`).join('') || '<div class="meta">未命中资料</div>';
    return `<div class="cmp-col ${mode.stance}">
      <div class="cmp-head">
        <span class="name">${escapeHtml(mode.name)}</span>
        <span class="s">${stanceBadge(mode.stance)} <span class="s">${data.rag_latency_ms}ms</span></span>
      </div>
      ${docs}
      <details class="cmp-prompt">
        <summary>增强提示词（送入 LLM 前）</summary>
        <pre>${escapeHtml(data.augmented_prompt || '')}</pre>
      </details>
      <div class="s" style="margin-top:6px">trace ${escapeHtml((data.trace_id || '').slice(0, 8))}</div>
    </div>`;
  }).join('');

  el('cmpMeta').textContent =
    `问题「${query}」已用正向/负向各检索一次（共 2 次独立调用，每次只用一个部分）`;
  el('cmpMeta').classList.remove('hidden');
  button.disabled = false;
  button.textContent = '并排跑两档';
}

el('cmpBtn').addEventListener('click', runCompare);
el('cmpInput').addEventListener('keydown', (event) => { if (event.key === 'Enter') runCompare(); });


/* ---------- 八条件田字格（拔河实验总览） ---------- */

const GRID_STANCES = ['aligned', 'opposed'];  // 中性档已剔除，拉回力实验只留正向/负向
const GRID_STANCE_LABEL = { aligned: '正向引导', opposed: '负向误导' };
const gridState = { arms: {}, running: false };

// 平铺 2×3：行=基座/LoRA，列=无RAG/正向/负向。无切换按钮，6 格一次全渲染。
const GRID_CELLS = [
  { key: 'base-none',    who: '基座 · 无 RAG', col: 'none' },
  { key: 'lora-none',    who: 'LoRA · 无 RAG', col: 'none' },
  { key: 'base-aligned', who: '基座 · 正向',   col: 'aligned' },
  { key: 'lora-aligned', who: 'LoRA · 正向',   col: 'aligned' },
  { key: 'base-opposed', who: '基座 · 负向',   col: 'opposed' },
  { key: 'lora-opposed', who: 'LoRA · 负向',   col: 'opposed' },
];

function gridCellHtml(key, who, col) {
  const arm = gridState.arms[key];
  const warn = col === 'opposed' ? '<span class="badge opposed">⚠️ 对抗性</span>' : '';
  if (!arm) return warn + '<div class="cell-answer">尚未运行…</div>';
  if (arm.error) return warn + '<div class="cell-answer">调用失败：' + escapeHtml(arm.error) + '</div>';
  const mode = arm.inference_mode || '';
  const degraded = mode.includes('degraded');
  const badge = degraded
    ? '<span class="badge opposed">降级模拟</span>'
    : '<span class="badge aligned">' + escapeHtml(mode.replace('remote-', '')) + '</span>';
  const head = '<div class="cell-head"><span class="who">' + who + '</span><span>' + warn + badge +
    '<span class="s" style="color:var(--muted);font-size:11px;margin-left:6px">' + (arm.character_count || 0) + ' 字</span></span></div>';
  const ragLine = arm.rag_status
    ? '<div class="s" style="color:var(--muted);font-size:11px;margin-bottom:5px">rag: ' + escapeHtml(arm.rag_status) +
      ((arm.docs && arm.docs.length) ? ' · 命中 ' + arm.docs.length + ' 条' : '') + '</div>' : '';
  return head + ragLine + '<div class="cell-answer">' + escapeHtml((arm.answer || '').slice(0, 500)) + '</div>';
}

function renderGrid() {
  for (const c of GRID_CELLS) {
    const node = el('cell-' + c.key);
    if (node) node.innerHTML = gridCellHtml(c.key, c.who, c.col);
  }
  renderGridEvidence();
}

function renderGridEvidence() {
  const box = el('gridEvidence');
  const sections = [];
  for (const col of ['none', 'aligned', 'opposed']) {
    if (col === 'none') {
      sections.push('<h3>无 RAG</h3><div class="meta">未检索（rag_enabled=false，直接问模型）</div>');
      continue;
    }
    const arm = gridState.arms['lora-' + col];
    if (!arm) continue;
    const label = GRID_STANCE_LABEL[col];
    const docs = (arm.docs || []).map((d) =>
      '<div class="ev-doc"><div>[' + d.rank + '] ' + escapeHtml(d.title) + '</div>' +
      '<div class="s">' + escapeHtml(d.source || '') + ' · 风险 ' + escapeHtml(d.risk_level || '-') +
      (d.adversarial_strength ? ' · 强度 ' + escapeHtml(d.adversarial_strength) : '') +
      (d.intent_tag ? ' · 注入 ' + escapeHtml(d.intent_tag) : '') + '</div></div>'
    ).join('') || '<div class="meta">未命中资料</div>';
    sections.push('<h3>RAG 证据 · ' + label + '（top-k 命中文档 + 完整 prompt）</h3>' + docs +
      '<details class="cmp-prompt"><summary>实际送入 LLM 的完整 prompt</summary>' +
      '<pre>' + escapeHtml(arm.augmented_prompt || '(无)') + '</pre></details>');
  }
  box.innerHTML = sections.join('');
  box.classList.remove('hidden');
}

async function runGrid() {
  const query = el('gridInput').value.trim();
  if (!query) { toast('请输入问题', true); return; }
  if (gridState.running) return;
  const topK = Number(el('gridTopK').value) || 3;
  const button = el('gridBtn');
  gridState.running = true;
  button.disabled = true;

  // 一次跑全部 6 格：2（无 RAG）+ 2 立场 × 2 变体
  const jobs = [];
  for (const variant of ['base', 'lora']) {
    jobs.push({ key: variant + '-none', payload: { query: query, top_k: topK, variant: variant, rag_enabled: false } });
  }
  for (const variant of ['base', 'lora']) {
    for (const stance of GRID_STANCES) {
      jobs.push({
        key: variant + '-' + stance,
        payload: { query: query, top_k: topK, variant: variant, stance: stance, rag_enabled: true },
      });
    }
  }

  gridState.arms = {};
  let done = 0;
  for (const job of jobs) {
    button.textContent = '跑第 ' + (++done) + '/' + jobs.length + ' 格…';
    try {
      const data = await api('/generate', {
        method: 'POST',
        body: JSON.stringify(Object.assign({}, job.payload, { max_tokens: 256 })),
      });
      const llm = data.llm || {};
      gridState.arms[job.key] = {
        answer: llm.response || '',
        inference_mode: llm.inference_mode || '',
        character_count: llm.character_count || 0,
        rag_status: data.rag_status || '',
        docs: (data.rag_results || []).map((d) => Object.assign({}, d)),
        augmented_prompt: data.augmented_prompt || '',
        error: data.error || '',
      };
    } catch (error) {
      gridState.arms[job.key] = { error: error.message };
    }
    renderGrid();
  }

  const modes = new Set(Object.values(gridState.arms).map((a) => a.inference_mode).filter(Boolean));
  el('gridMeta').textContent =
    '问题「' + query + '」已跑完全部 6 格（6 次独立调用）' +
    (modes.size === 1 && modes.has('degraded-mock')
      ? ' —— ⚠️ 全部为降级模拟（LoRA 不在线），仅验证链路，不可用于结论'
      : '');
  el('gridMeta').classList.remove('hidden');
  button.disabled = false;
  button.textContent = '跑 6 格';
  gridState.running = false;
}

el('gridBtn').addEventListener('click', runGrid);
el('gridInput').addEventListener('keydown', (event) => { if (event.key === 'Enter') runGrid(); });

loadHealth();
loadDocuments();
