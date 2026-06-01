import pytest
from langgraph_pipeline import AppContext, DataIngestionHandler, LangGraphRoutingHandler

@pytest.mark.asyncio
async def test_chain_of_responsibility_and_langgraph():
    """
    TDD Test 1:
    确认在面向对象的设计下，责任链模式能正确挂载和传递状态。
    并测试 LangGraph 能正确的做多 Agent 节点的内部调度（并发/并行扇出和温度注入）。
    """
    
    # 1. 建立初始数据
    ctx = AppContext(
        project_context="烂尾项目求抢救",
        tech_temp=1.5,
        risk_temp=1.1,
        cost_temp=0.7
    )
    
    # 2. 从对象实例化节点并搭建单向传输责任链
    ingest_handler = DataIngestionHandler()
    routing_handler = LangGraphRoutingHandler()
    ingest_handler.set_next(routing_handler)
    
    # 3. 运行执行链
    final_ctx = await ingest_handler.handle(ctx)
    
    # 4. 严苛断言（测试驱动准则）
    assert final_ctx.status == "ROUTING_COMPLETE", "最终状态必须是评估完成"
    
    # 确认动态温度和评委被精准路由进去了
    assert "Tech(T=1.5)" in final_ctx.tech_eval, "缺少技术专家及对应的动态温度设定"
    assert "Risk(T=1.1)" in final_ctx.risk_eval, "缺少风控专家及对应的动态温度设定"
    assert "Cost(T=0.7)" in final_ctx.cost_eval, "缺少成本专家及对应的动态温度设定"
