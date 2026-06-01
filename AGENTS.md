# AGENTS.md (AI 编程代理开发准则)

本文件是本项目所有 AI 编程代理（Agents）、代码生成模型以及自动化工具的**核心宪法**。在修改或新增代码时，必须严格遵守以下范式。

## 1. 架构目标：LangGraph 多智能体协同
本项目致力于引入 **LangGraph** 框架，将多专家（毒舌架构师、合规风控、成本 ROI）的并发路由、模型动态温度控制交由图结构状态机自动化调度。
- **Agent职责**：每个节点 Node 作为一个独立的 Expert Agent。
- **状态流转**：通过 StateGraph 控制上下文与评价分数的自动流转，不再使用扁平的 `Promise.all` 硬编码。

## 2. 设计模式：面向对象与责任链 (Chain of Responsibility)
在软件工程模块化层面，强制实施面向对象程序设计（OOP）的责任链范式：
- **职责单一**：将请求传递链上的状态流转严格分离，例如 `DataIngestion` -> `LangGraphRouting` -> `RLSimulation` -> `Visualization`。
- **消除冗余**：**严禁写冗余和重复调用的函数！** 所有的公共处理逻辑必须抽象为基类 (Base Class) 或混入类 (Mixin)。
- **模块化边界**：不允许面面俱到的“上帝类 (God Class)”或几千行的单文件单体脚本。前后端以及算法逻辑必须完全解耦。

## 3. 开发范式：测试驱动开发 (TDD - Test-Driven Development)
不允许“先写代码，再补测试”的草台班子做法。
- **测试先行**：增加任何新接口、新算法或新的 LangGraph 节点前，必须先在 `tests/` 中定义好输入输出、边界条件、失败处理的断言。
- **行为契约**：遵守已有核心数学或逻辑的事实标准，即使后台架构大换血（换到 Python / LangGraph），原有的 21 项断言必须全部覆盖并保证通过。

## 4. 流程规范：MLOps 与 DevOps
- **DevOps (持续集成/持续部署)**：
  - 强制执行代码审查原则。
  - 所有新功能必须先做 **Dry-run (试运行)** 与单元测试拦截，确保服务能平稳启动，才能挂载到主流程。
- **MLOps (大模型与强化学习运维)**：
  - **模型监控**：强化学习收敛曲线（PPO 概率坍缩等）的分布必须可观测，不能出现梯度爆炸导致死循环。
  - **动态调度**：基于 LangGraph 的大模型调用必须有容错（Fallback）机制、并发超时时间控制以及耗时/Token 分析兜底。

---
> **To AI Agent**: Read these rules carefully before initiating any code refactor. If you understand these instructions, explicitly acknowledge adherence to OOP, TDD, and MLOps principles in your implementation steps.