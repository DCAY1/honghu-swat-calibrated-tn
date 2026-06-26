#!/bin/bash
# 在可访问 GitHub 的终端中运行此脚本，完成仓库创建与首次推送。
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

OWNER="DCAY1"
REPO_NAME="honghu-swat-calibrated-tn"
REMOTE="https://github.com/${OWNER}/${REPO_NAME}.git"

echo "==> 检查 GitHub 连接..."
if ! curl -fsI --connect-timeout 10 https://github.com >/dev/null; then
  echo "无法连接 github.com，请检查网络或代理后重试。"
  exit 1
fi

if command -v gh >/dev/null 2>&1; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "请先登录 GitHub CLI: gh auth login"
    gh auth login -h github.com -p https -w
  fi
  LOGIN="$(gh api user -q .login)"
  OWNER="${LOGIN}"
  REMOTE="https://github.com/${OWNER}/${REPO_NAME}.git"
  if ! gh repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
    echo "==> 创建远程仓库 ${OWNER}/${REPO_NAME} ..."
    gh repo create "${REPO_NAME}" --public --source=. --remote=origin --push
    echo "完成: https://github.com/${OWNER}/${REPO_NAME}"
    exit 0
  fi
fi

git remote set-url origin "$REMOTE"
echo "==> 推送到 ${REMOTE}"
git push -u origin main
echo "完成: https://github.com/${OWNER}/${REPO_NAME}"
