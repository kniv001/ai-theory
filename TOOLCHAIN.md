# 工具记录（TOOLCHAIN.md）

> 理论研究使用的全部工具/环境/技能。新工具首次使用时在此登记。
> 记录原则：只记录实际可用/已确认的；未确认的进"待办"。

## 环境检查结果（2026-08-16）

| 工具 | 状态 | 备注 |
|---|---|---|
| git | ✅ 可用 | d:\vs 仓库，分步提交便于回溯 |
| python | ✅ 3.14.6（系统）/ 3.13.13（conda） | 仿真验证候选 |
| conda | ✅ 26.3.2 | D:\vs\conda |
| pandoc | ❌ 未装 | LaTeX 方案需要时再装 |
| xelatex | ❌ 未装 | 数学写作暂用 Markdown 内联公式 |

## 推理与对话

- **Claude Code（本会话）** — 主推理伙伴；长会话用 /loop、/compact 管理
- **dual-agent 本地系统** — 可编程执行端（后续脚本/仿真/自动化时启用）
- **持久记忆**（`~/.claude/projects/.../memory/`）— 跨会话锚定

## 文献与检索

- **WebSearch / WebFetch** — arXiv、论文、教科书检索
- （需要时）arXiv API / Semantic Scholar

## 形式化与数学

- Markdown 内联数学（当前轻量方案，`$...$`）
- 待办：如需要正式排版 → 装 pandoc + MiKTeX；如需要机器验证 → 评估 Lean 4

## 仿真验证（若锚定需要）

- Python 3 + NumPy（数值验证）
- 待办：PyTorch / JAX（玩具网络，需要时确认）

## 待办清单

- [ ] 首轮理论推导后，把实际用到的工具回填
