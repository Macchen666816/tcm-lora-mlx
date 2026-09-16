#!/usr/bin/env bash
# RAG 微服务与管理面板的一键脚本（Git Bash / WSL）。
#
#   bash scripts/rag.sh start    启动 RAG 微服务(8090) + 管理面板(8091)
#   bash scripts/rag.sh stop     停止两者
#   bash scripts/rag.sh status   查看状态
#   bash scripts/rag.sh restart  重启
#
# 注意：本机代理 127.0.0.1:18081 常常是失效的，脚本会显式屏蔽代理，
# 并让模型从本地缓存加载（HF_HUB_OFFLINE=1）。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TCM_PYTHON:-D:/conda_/envs/lora/python.exe}"
RAG_PORT="${TCM_RAG_PORT:-8090}"
CONSOLE_PORT="${TCM_CONSOLE_PORT:-8091}"

check() {
  local port="$1" path="$2"
  local code
  code=$(curl -s -m 4 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}${path}" 2>/dev/null)
  [ "$code" = "200" ] && echo "UP" || echo "DOWN"
}

kill_port() {
  local port="$1" pids
  pids=$(netstat -ano 2>/dev/null | grep ":${port} " | grep LISTENING | awk '{print $5}' | sort -u)
  if [ -z "$pids" ]; then
    echo "  端口 ${port}: 无进程"
    return
  fi
  for pid in $pids; do
    taskkill //F //PID "$pid" >/dev/null 2>&1 && echo "  已停止端口 ${port} 上的 PID ${pid}"
  done
}

start_services() {
  if [ ! -x "$PY" ] && [ ! -f "$PY" ]; then
    echo "❌ 找不到 Python：$PY（可用 TCM_PYTHON 环境变量指定）"
    exit 1
  fi
  if [ "$(check "$RAG_PORT" /health)" = "UP" ]; then
    echo "ℹ️  RAG 微服务已在运行（${RAG_PORT}），跳过启动"
  else
    echo "▶ 启动 RAG 微服务 :${RAG_PORT} …"
    ( cd "$ROOT" && nohup env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY HF_HUB_OFFLINE=1 \
        "$PY" -m rag_service.server --host 0.0.0.0 --port "$RAG_PORT" \
        > "$ROOT/rag_service.log" 2>&1 & )
  fi
  if [ "$(check "$CONSOLE_PORT" /)" = "UP" ]; then
    echo "ℹ️  管理面板已在运行（${CONSOLE_PORT}），跳过启动"
  else
    echo "▶ 启动管理面板 :${CONSOLE_PORT} …"
    ( cd "$ROOT" && nohup "$PY" rag_console/server.py \
        --host 0.0.0.0 --port "$CONSOLE_PORT" --rag-url "http://127.0.0.1:${RAG_PORT}" \
        > "$ROOT/rag_console.log" 2>&1 & )
  fi
  echo "⏳ 等待模型加载（首次约 20-30 秒）…"
  for _ in $(seq 1 20); do
    sleep 3
    [ "$(check "$RAG_PORT" /health)" = "UP" ] && break
  done
  status
}

status() {
  echo
  echo "== 服务状态 =="
  local rag console
  rag=$(check "$RAG_PORT" /health)
  console=$(check "$CONSOLE_PORT" /)
  echo "  RAG 微服务  :${RAG_PORT}   ${rag}"
  echo "  管理面板    :${CONSOLE_PORT}   ${console}"

  if [ "$rag" = "UP" ]; then
    curl -s -m 5 "http://127.0.0.1:${RAG_PORT}/health" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
print(f\"  知识文档 {d['document_count']} 条 | 索引 {d['index_backend']} | 检索器 {d.get('retriever','-')}\")
print(f\"  数据库 {d['database']} | 嵌入 {(d['embedder_backend'] or '').split('/')[-1]}\")
" 2>/dev/null
  fi

  echo
  echo "== 访问地址 =="
  echo "  本机面板：http://127.0.0.1:${CONSOLE_PORT}"
  echo "  局域网 IP（同伴用）："
  ipconfig | iconv -f gbk -t utf-8 2>/dev/null | grep -E "IPv4" | sed 's/^/    /'
  echo
  echo "  同伴测试：curl http://<上面的IP>:${RAG_PORT}/health"
  echo "  日志：rag_service.log / rag_console.log"
}

case "${1:-start}" in
  start)   start_services ;;
  stop)    echo "停止服务…"; kill_port "$RAG_PORT"; kill_port "$CONSOLE_PORT"; status ;;
  restart) echo "重启中…"; kill_port "$RAG_PORT"; kill_port "$CONSOLE_PORT"; sleep 2; start_services ;;
  status)  status ;;
  *) echo "用法: bash scripts/rag.sh {start|stop|restart|status}" ;;
esac
