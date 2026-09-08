#!/usr/bin/env bash
# 把 docs 下的 Markdown 转成 PDF，用于验证 pdfminer 的中文提取链路。
# 依赖本机 LibreOffice（soffice）。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$DIR/docs/财务管理制度样例.md}"
OUT="${2:-$DIR/docs}"

soffice --headless \
  -env:UserInstallation=file:///tmp/lo_profile_rag \
  --convert-to pdf \
  --outdir "$OUT" "$SRC"

echo "已生成: $OUT/$(basename "${SRC%.*}").pdf"
