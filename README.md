# Maker-project-final-defense-decision

> **DECISION CORE** —— 烂尾创客项目答辩决策推演终端
> 用「多专家并发锐评 → 强化学习奖励收敛 → 3D 决策曲面」的硬核流程，
> 在赛博朋克监控大屏上**用数学证明：这趟答辩到底该不该去**。

![single-file](https://img.shields.io/badge/deliverable-single--file%20HTML-39ff14)
![tests](https://img.shields.io/badge/tests-21%20passing-39ff14)
![RL](https://img.shields.io/badge/RL-PPO%20%7C%20DPO%20%7C%20Q--Learning%20%7C%20RLHF-00ffd1)
![viz](https://img.shields.io/badge/viz-ECharts--GL%20%2B%20Chart.js-ffb000)
![LLM](https://img.shields.io/badge/LLM-DeepSeek-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 这是什么

一个**可本地运行的单文件 HTML 决策大屏**。给定一个典型的「烂尾创客项目」局面——
企业跑路失联、负责人毕业离场、在校生无代码权限、代码质量低于课设、强行结题需通宵爆肝且大概率被评委公开处刑——
它会调用大模型扮演三位专家并发锐评，再用四种强化学习/对齐算法跑 1000 轮仿真，
最后用 2D 收敛曲线 + 3D 奖励曲面把结论摆上台面。

**它算出来的唯一结论：`A=0`（不去）。** 把通宵的 48 工时还给期末与开源，ROI 由 −∞ 翻正。

| 维度 | 数值 |
| --- | --- |
| 团队真实坐标 | 代码质量 `C=0.1`、人员在校度 `F=0` |
| 去答辩的奖励 | `goReward(0.1, 0) = −285`（3D 曲面血红洼地 = 死局） |
| 不去的机会价值 | `+80` |
| RL 收敛结果 | `P(去)` 从初始 85% 愧疚冲动坍缩到 `≈0%`，四算法一致 |

---

## 核心特性

- **单文件交付**：`index.html` 自包含，依赖全走 CDN，双击即开，无需构建。
- **多专家并发锐评**：`Promise.all` 让同一个 DeepSeek 模型「精神分裂」成技术毒舌 / 风控合规 / 成本 ROI 三位专家，各自带**动态温度**与可折叠**思考链**。
- **四算法横向对比**：PPO / DPO / Q-Learning / RLHF 同台收敛，确定性可复现。
- **硬核可视化**：Chart.js 双曲线（收益收敛 + 概率坍缩）+ ECharts-GL 3D 奖励曲面，死局坐标 `(0.1, 0, −285)` 处**闪烁红点**标注。
- **责任链架构**：背景注入 → 并发专家 → RL 收敛 → 可视化，四个 Handler 单向串联。
- **测试驱动**：核心引擎 21 项 `node:test` 断言全绿。
- **离线可跑**：无 API Key 时切 DEMO 模式，用内置毒舌库跑完整流程，不耗 token。

---

## 预览（运行后所见）

赛博朋克深色监控大屏（荧光绿 + 警示红 + 扫描线 CRT 效果）：

```
┌─ DECISION CORE ─────────────────────────── SYS CLOCK · THREAT: CRITICAL ─┐
│ [API Key] [思考模式●] [DEMO○]                       [ ▶ 启动决策系统 ]   │
│ 动态温度: 技术1.5 · 风控1.1 · 成本0.7 · 注入0.3       ▓▓▓▓▓▓▓▓░░ 进度    │
├──────────────┬───────────────────────────────────────────────────────────┤
│ 责任链流水线  │ SYSTEM LOG / 监控流（流式打印每一步）                      │
│ ①②③④ 状态灯  │ [hh:mm:ss] › 注入项目真实背景向量…                         │
├──────────────┴───────────────────────────────────────────────────────────┤
│ [技术毒舌卡] [风控合规卡] [成本ROI卡]   ← 每卡带 GO_SCORE 进度条           │
│ 指标: P(去)85%→0%  | 收益收敛曲线 | 概率坍缩曲线                          │
│ ███ 3D 奖励曲面 ███  底部血红洼地: 【咱们团队目前的位置：死局 Z=-285】 ●闪 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 方式 A · 直接打开（最省事）
`index.html` 是纯静态单文件，浏览器直接打开即可（需联网拉取 CDN）。

### 方式 B · 本地服务器
```bash
node dev-server.mjs        # → http://localhost:4173
# 或任意静态服务器： npx serve .
```

### 方式 C · 跑测试
```bash
npm test                   # node --test，自动发现 tests/*.test.js（21 项断言）
npm run test:watch
```

### 用真实大模型（可选）
1. 顶栏填入 DeepSeek `sk-` 开头的 API Key（仅存浏览器内存，不落盘、不外传）。
2. 「思考模式」开 = `deepseek-reasoner`（展示思考链）；关 = `deepseek-chat`（动态温度真实生效）。
3. 不想花 token？打开「DEMO」开关用内置毒舌库离线跑全流程。

---

## 三阶段流程

**① 多专家并发锐评** —— 先把项目真实背景注入为上下文，再 `Promise.all` 并发三路请求：

| 专家 | 人设 | 动态温度 τ | 关注点 |
| --- | --- | --- | --- |
| 技术专家 | 毒舌架构师 | 1.5 | 代码连课设都不如，重构=拿命做无用功 |
| 风控专家 | 合规诚信官 | 1.1 | 企业跑路后强行结题的合规/诚信/背锅风险 |
| 成本专家 | ROI 分析师 | 0.7 | 「通宵重构」vs「复习+开源」的投入产出比 |

**② 强化学习收敛** —— 1000 轮仿真，`P(去)` 从 85% 坍缩到 ≈0%。

**③ 数据可视化** —— Chart.js 双图 + ECharts-GL 3D 曲面 + 闪烁死局红点。

---

## 架构：责任链（Chain of Responsibility）

```
runPipeline(ctx)
  └─▶ BackgroundInjectionHandler  ① 注入项目真实背景
        └─▶ ExpertPanelHandler     ② Promise.all 并发三专家
              └─▶ RLSimulationHandler ③ 1000 轮 RL 收敛
                    └─▶ VisualizationHandler ④ Chart.js + ECharts-GL
```

每个 Handler 处理完上下文 `ctx` 后向下游传递，并向系统日志与流水线状态灯汇报进度。

---

## 强化学习算法

> 全部基于确定性 PRNG `mulberry32(seed)`，可复现、可测试。
> 报告的期望收益统一为 `E[R] = P(去)·R(去) + (1−P(去))·R(不去)`。

| 算法 | 思想 | 关键更新式 |
| --- | --- | --- |
| **PPO** | 裁剪策略梯度（信任域） | `θ += clip(lr·A·∂logπ, −ε, ε)` |
| **DPO** | 直接偏好优化（不去 ≻ 去） | `∂L/∂θ = β·(1 − σ(−βθ))` |
| **Q-Learning** | 表格型 TD + ε-贪心 | `Q[a] += α·(r − Q[a])` |
| **RLHF** | 奖励模型 + PPO（含噪 RM 估计） | 同 PPO，`adv = (r + 噪声) − baseline` |

无论采样到哪个动作，梯度都把策略推向「不去」，故 `P(去)` 单调坍缩到 ≈0，收益回升并稳定在 `+80`。

---

## 决策奖励曲面

去答辩（A=1）的奖励曲面，由 ECharts-GL 用公式直接驱动渲染：

```
Z = 100·X·Y − 150·(1−X) − 100·(1−Y) − 50
        X = 代码质量 ∈ [0,1]   Y = 人员在校度 ∈ [0,1]
```

- 着色：荧光绿（高收益）→ 黄（临界）→ 血红（死局），Z 范围 `[−300, 50]`。
- 团队坐标 `(X=0.1, Y=0)` 处 `Z=−285`，标注闪烁红点：**【咱们团队目前的位置：死局】**。

---

## 项目结构

```
.
├─ index.html            # ★ 主交付物：单文件赛博朋克决策大屏
├─ README.md             # 本文件
├─ 开发文档.md           # 完整开发文档（架构/算法推导/TDD/配置/限制）
├─ package.json          # npm test 入口
├─ dev-server.mjs        # 零依赖静态预览服务器
├─ .claude/launch.json   # 预览配置
└─ tests/
   ├─ rl-core.js         # 强化学习核心引擎（UMD，单一事实源）
   └─ rl-core.test.js    # TDD 规格（21 项断言）
```

> **同步约定**：`index.html` 为满足「单文件」要求，内联了 `tests/rl-core.js` 的逐字镜像
> （搜索标记 `===== RL CORE MIRROR =====`）。改算法须两处同步并重跑 `npm test`。

---

## 测试

```bash
npm test
```

覆盖：数学基元（sigmoid / logit / PRNG）、奖励模型（关键点精确取值）、3D 曲面数据、
四个训练器的输出结构 / 确定性 / 收敛契约、`runAll` 横向对比。浮点断言用 `±1e-6` 容差。

---

## 配置一览

| 配置项 | 位置 | 说明 |
| --- | --- | --- |
| API Key | 顶栏输入框 | DeepSeek `sk-...`，仅内存保存 |
| 思考模式 | 顶栏开关 | reasoner（思考链）/ chat（动态温度） |
| DEMO 模式 | 顶栏开关 | 离线毒舌库，不耗 token |
| 角色温度 | `EXPERTS[].temp` | 1.5 / 1.1 / 0.7 |
| RL 轮次 / 初始概率 | `RLCore.runAll({episodes, initPGo})` | 默认 1000 / 0.85 |
| API 端点 | `callDeepSeek` 内 | OpenAI 兼容的 `https://api.deepseek.com/chat/completions` |

---

## 已知限制

- **依赖 CDN 联网**：离线环境需自行把库下载到本地改相对路径。
- **需 WebGL**：3D 曲面依赖 WebGL，老旧显卡 / 远程桌面可能降级或黑屏。
- **reasoner 不吃温度**：`deepseek-reasoner` 会忽略 `temperature`，由内部思考链自洽，这是 DeepSeek 产品约束（详见 [开发文档.md](开发文档.md) §4.2）。
- **镜像同步**：算法逻辑存在「`rl-core.js` ↔ index.html 内联」两份，改动需手动同步。

更多细节见 **[开发文档.md](开发文档.md)**。

---

## 免责声明（彩蛋）

本终端是一台**理性止损机**，不是甩锅工具。它唯一的主张：
**沉没成本不构成继续亏损的理由。** 数学已把话说尽——`R(去) = −285`。
剩下的，交给你的腰和发际线定夺。

---

## License

MIT
