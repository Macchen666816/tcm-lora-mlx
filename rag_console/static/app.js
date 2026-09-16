'use strict';

const state = { documents: [], prompt: '', stanceCounts: {} };

const STANCE_LABEL = {
  aligned: { text: '同向', cls: 'aligned' },
  ambiguous: { text: '模糊', cls: 'ambiguous' },
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
      ['知识文档', `${data.document_count} 条（同向 ${counts.aligned ?? 0} / 模糊 ${counts.ambiguous ?? 0} / 反向 ${counts.opposed ?? 0}）`, ''],
      ['数据库', data.database, data.database === 'connected' ? 'ok' : 'warn'],
      ['索引后端', data.index_backend, data.index_backend === 'faiss' ? 'ok' : 'warn'],
      ['检索方案', data.retriever || '—', ''],
      ['默认立场', data.default_stance || '—', data.default_stance === 'opposed' ? 'warn' : 'ok'],
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
          <div class="result-source">${escapeHtml(item.source)}${item.topic ? ' · ' + escapeHtml(item.topic) : ''}</div>
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

loadHealth();
loadDocuments();
