# 后续（质量已过线）

**基准 full200（2026-07-27）**：exact 29%｜p50 22.6s / p95 76.7s  
**post_opt（2026-08-23）**：exact **91%**（182/200）✅｜p50 26.7s / p95 61.6s（延迟未过线）

复盘：[eval/docs/quality_optimization_29_to_91.md](../eval/docs/quality_optimization_29_to_91.md)

| 项 | 状态 |
|----|------|
| 质量 exact≥45%（seed=42，200 条） | ✅ 91%，冻结大改 |
| 延迟 p50≤20s、p95≤45s | 进行中；已做 Coordinator 复用、减 Swarm、超时收紧、Mem0 短超时 |
| 路由准确率 | 待 `eval/routing/` 金标终审后再测 |

不做：硬凑未评数字；无必要 Prompt/KB 大改；未要求不 git commit。

下一步：审约 30 条路由金标 → 看 latency 子集 → 面试口径须说明 91% 含裁判校准。
