from abc import ABC, abstractmethod
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

# --- 1. 面向对象的设计：上下文与实体 (Data Models) ---

class AppContext:
    """流转整个责任链的上下文实体"""
    def __init__(self, project_context, tech_temp=1.5, risk_temp=1.1, cost_temp=0.7):
        self.project_context = project_context
        # MLOps: 动态大模型调试参数
        self.tech_temp = tech_temp
        self.risk_temp = risk_temp
        self.cost_temp = cost_temp
        
        # 结果槽位
        self.tech_eval = ""
        self.risk_eval = ""
        self.cost_eval = ""
        self.status = "INITIALIZED"

# --- LangGraph 内部状态定义 (State) ---
# TDD & LangGraph 要求：TypedDict 管理各个节点内的数据流
class GraphState(TypedDict):
    project_context: str
    tech_temp: float
    risk_temp: float
    cost_temp: float
    # 用 Annotations 和 reducers 管理图内更新机制
    tech_eval: str
    risk_eval: str
    cost_eval: str

# --- 2. 责任链设计模式底层基类 (OOP Chain of Responsibility) ---

class BaseHandler(ABC):
    """责任链节点基类：消除冗余，统一数据流转。不允许写单体面面俱到函数。"""
    def __init__(self):
        self.next_handler = None
        
    def set_next(self, handler):
        self.next_handler = handler
        return handler
        
    async def handle(self, ctx: AppContext) -> AppContext:
        ctx = await self.process(ctx)
        if self.next_handler:
            return await self.next_handler.handle(ctx)
        return ctx
        
    @abstractmethod
    async def process(self, ctx: AppContext) -> AppContext:
        """子类覆盖此方法完成具体的业务"""
        pass

# --- 3. 具体的责任链环节 ---

class DataIngestionHandler(BaseHandler):
    """节点 A: 数据注入"""
    async def process(self, ctx: AppContext) -> AppContext:
        ctx.status = "INGESTED"
        # 此处可以拓展数据清洗、历史记录加载等
        return ctx

# --- LangGraph 专家节点定义 (Nodes) --- 
def tech_expert_node(state: GraphState):
    t = state["tech_temp"]
    # 此处模拟调用了 LLM (如 DeepSeek) 可以根据 t (temperature) 做动态调整
    return {"tech_eval": f"Tech(T={t}): 严重警告，代码如山倒，不建议救。"}

def risk_expert_node(state: GraphState):
    t = state["risk_temp"]
    return {"risk_eval": f"Risk(T={t}): 合规审查失败，负责人逃跑了不能接盘。"}

def cost_expert_node(state: GraphState):
    t = state["cost_temp"]
    return {"cost_eval": f"Cost(T={t}): 48工时收益为负，ROI爆冷。"}

class LangGraphRoutingHandler(BaseHandler):
    """节点 B: 基于 LangGraph 的多专家并发路由系统"""
    def __init__(self):
        super().__init__()
        # 构建 LangGraph 状态机 - 遵循 MLOps 任务流观测
        builder = StateGraph(GraphState)
        
        # 添加代理 Expert 节点
        builder.add_node("tech_expert", tech_expert_node)
        builder.add_node("risk_expert", risk_expert_node)
        builder.add_node("cost_expert", cost_expert_node)
        
        # Fan-out 并行启动多专家点评
        builder.add_edge(START, "tech_expert")
        builder.add_edge(START, "risk_expert")
        builder.add_edge(START, "cost_expert")
        
        # Fan-in 聚合到终点
        builder.add_edge("tech_expert", END)
        builder.add_edge("risk_expert", END)
        builder.add_edge("cost_expert", END)
        
        self.graph = builder.compile()

    async def process(self, ctx: AppContext) -> AppContext:
        # Step 1: 适配器模式，将 Context 转为 GraphState
        initial_state = {
            "project_context": ctx.project_context,
            "tech_temp": ctx.tech_temp,
            "risk_temp": ctx.risk_temp,
            "cost_temp": ctx.cost_temp,
            "tech_eval": "",
            "risk_eval": "",
            "cost_eval": ""
        }
        
        # Step 2: ainvoke 触发整个并发图，并自动聚合结果 (MLOps/并行调度)
        result = await self.graph.ainvoke(initial_state)
        
        # Step 3: 更新下文数据
        ctx.tech_eval = result["tech_eval"]
        ctx.risk_eval = result["risk_eval"]
        ctx.cost_eval = result["cost_eval"]
        ctx.status = "ROUTING_COMPLETE"
        
        return ctx