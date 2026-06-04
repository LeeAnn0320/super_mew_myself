from typing import Dict,Optional



_LAST_RAG_CONTEXT=None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue

def _set_last_rag_context(context:Dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT=context



def get_last_rag_context(clear:bool=True)->Optional[Dict]:
    "获取最近一次检索上下文，读取后清空"
    global _LAST_RAG_CONTEXT
    context=_LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT=None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数"""

    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN=0



def set_rag_step_queue(queue):
    """设置RAG步骤队列，并捕获当前事件循环以便跨线程调度"""
    global _RAG_STEP_LOOP,_RAG_STEP_QUEUE

    _RAG_STEP_QUEUE=queue

    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP=asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP=asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP=None

def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass