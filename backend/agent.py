from dotenv import load_dotenv
import os
import json
import asyncio
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,AIMessageChunk
from typing import List,Dict
from database import SessionLocal
from cache import cache
from datetime import datetime,timezone
from models import User,ChatSession,ChatMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools import get_last_rag_context,_set_last_rag_context,reset_tool_call_guards,set_rag_step_queue
load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")


class ConversationStorage:
    '''对话存储'''
    @staticmethod
    def _messages_cache_key(user_id:str,session_id:str)->str:
        return f"chat_messages:{user_id}:{session_id}"
    

    @staticmethod
    def _session_cache_key(user_id:str)->str:
        return f"chat_session:{user_id}"
    
    @staticmethod
    def _to_langchain_messages(records:List[Dict])->List:
        messages=[]

        for msg_data in records:
            msg_type=msg_data.get("type")
            content=msg_data.get("content","")
            if msg_type =="human":
                messages.append(HumanMessage(content=content))
            elif msg_type =="ai":
                messages.append(AIMessage(content=content))
            elif msg_type=="system":
                messages.append(SystemMessage(content=content))
        return messages
    
    def save(self,user_id:str,session_id:str,messages:List,metadata:Dict=None,extra_message_data:List=None):
        '''保存对话'''
        db=SessionLocal()

        try:

            user=db.query(User).filter(User.username==user_id).first()

            if not user:
                return
            session=(
                db.query(ChatSession).filter(ChatSession.user_id==user_id,ChatSession.session_id==session_id)
                .first()
            )
            if not session:
                session=ChatSession(user_id=user.id,session_id=session_id,metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json=metadata or {}
            db.query(ChatMessage).filter(ChatMessage.session_ref_id==session_id).delete(synchronize_session=False)
            
            serialized=[]
            now=datetime.utcnow()
            # now=datetime.now(timezone.utc)
            # 2026-06-04 04:36:15.420008+00:00

            for idx,msg in enumerate(messages):
                rag_trace=None
                if extra_message_data and idx < len(extra_message_data):
                    extra=extra_message_data[idx]or {}

                    rag_trace=extra.get("rag_trace")
                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace
                    )
                )
                serialized.append(
                    {
                        "type":msg.type,
                        "content":str(msg.content),
                        "timestamp":now.isoformat(),
                        "rag_trace":rag_trace
                    }
                )
            session.updated_at=now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id,session_id),serialized)
            cache.delete(self._session_cache_key(user_id))
        
        finally:
            db.close()
    def load(self,user_id:str,session_id:str)->List:
        """加载对话"""
        cached=cache.get_json(self._messages_cache_key(user_id,session_id))

        if cached is not None:
            return self._to_langchain_messages(cached)
        
        records=self.get_session_messages(user_id,session_id)
        cache.set_json(self._messages_cache_key(user_id,session_id),records)
        return self._to_langchain_messages(records)


    def list_sessions(self,user_id:str)->List:
        """列出用户的所有对话"""
    
        return [item["session_id"]for item in self.list_session_infos(user_id)]


    def list_session_infos(self,user_id:str)->List[Dict]:
        cached=cache.get_json(self._session_cache_key(user_id))

        if cached is not None:
            return cached
        
        db=SessionLocal()
        try:
            user=db.query(User).filter(User.username==user_id).first()
            if not user:
                return []
            
            sessions=(
                db.query(ChatSession).filter(
                    ChatSession.user_id==user_id
                ).order_by(ChatSession.updated_at.desc()).all()
            )

            result=[]

            for s in sessions:
                count=db.query(ChatMessage).filter(ChatMessage.session_ref_id==s.id).count()
                result.append(
                    {
                        "session_id":s.session_id,
                        "updated_at":s.updated_at.isoformat(),
                        "messsage_count":count
                    }
                )
            cache.set_json(self._session_cache_key(user_id),result)
            return result
        
        finally:
            db.close()

    def get_session_messages(self,user_id:str,session_id:str)->List[Dict]:
        cached=cache.get_json(self._messages_cache_key(user_id,session_id))
        if cached is not None:
            return cached
        
        db=SessionLocal()

        try:
            user=db.query(User).filter(User.username==user_id).first()
            if not user:
                return []
            
            session=(
                db.query(ChatSession).filter(ChatSession.user_id==user_id,ChatSession.session_id==session_id).first()
            )
            if not session:
                return []
            rows=(
                db.query(ChatMessage).filter(ChatMessage.session_ref_id==session.id).order_by(ChatMessage.id.asc()).all()
            )
            result=[
                {
                    "type":row.message_type,
                    "content":row.content,
                    "timestamp":row.timestamp.isoformat(),
                    "rag_trace":row.rag_trace
                }for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id,session_id),result)
            return result
        finally:
            db.close()
        

    def delete_session(self,user_id:str,session_id:str)->bool:
        '''删除指定用户的会话返回是否删除成功'''
        db=SessionLocal()

        try:
            user=db.query(User).filter(User.username==user_id).first()
            if not user:
                return False
            
            session=(
                db.query(ChatSession).filter(ChatSession.user_id==user_id,ChatSession.session_id==session_id).first()
            )

            if not session:
                return False
            db.delete(session)

            db.commit()
            cache.delete(self._messages_cache_key(user_id,session_id))
            cache.delete(self._session_cache_key(user_id))
            return True
        finally:
            db.close()


def create_agent_instance():
    model=init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True
    )
    
    agent=create_agent(
        model=model,tools=[],system_prompt=(
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        )
    )
    return agent,model


agent,model=create_agent()

storage=ConversationStorage()

def summarize_old_messages(model,messages:List)->str:
    """将旧消息总结为摘要"""
    old_conversation="\n".join([
        f"{'用户'if msg.type=='human'else'AI'}:{msg.content}"for msg in messages
    ])
    #生成摘要

    summary_prompt=f"""请总结以下对话的关键信息：{old_conversation}总结（包含用户信息，重要事实，待办事项）:"""

    summary=model.invoke(summary_prompt).content
    return summary
    


def chat_with_agent(user_text:str,user_id:str="default_user",session_id:str="default_session"):
    """使用agent处理消息并返回响应"""

    messages=storage.load(user_id,session_id)
    #清理可能残留的RAG上下文
    get_last_rag_context()
    reset_tool_call_guards()

    if len(messages)>50:
        summary=summarize_old_messages(model,messages[:40])
        messages=[
            SystemMessage(content=f"之前的对话摘要:\n{summary}")
        ]+messages[:40]
    messages.append(HumanMessage(content=user_text))
    result=agent.invoke(
        {"messages":messages},
        config={"recursion_limit":8},
        )
    response_content=""
    if isinstance(result,dict):
        if "output" in result:
            response_content=result["output"]
        elif "messages" in result  and result["messages"]:
            msg=result["messages"][-1]
            response_content=getattr(msg,"content",str(msg))
        else:
            response_content=str(result)
    elif hasattr(result,"content"):
        response_content=result.content
    else:
        response_content=str(result)
    messages.append(AIMessage(content=response_content))
    rag_context=get_last_rag_context(clear=True)
    rag_trace=rag_context.get("rag_trace") if rag_context else None

    extra_message_data=[None]*(len(messages)-1)+[{"rag_trace":rag_trace}]
    storage.save(
        user_id,session_id,messages,extra_message_data=extra_message_data
    )
    return {
        "response":response_content,
        "rag_trace":rag_trace
    }


async def chat_with_agent_stream(user_text:str,user_id:str="default_user",session_id:str="default_session"):
    """使用agent 处理用户消息并流式响应"""

    messages=storage.load(user_id,session_id)

    #清理可能残留的RAG上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()


    #统一输出队列：所有事件都汇入这里

    output_queue=asyncio.Queue()

    class _RAGStepProxy:
        """代理对象:将emit_rag_step的原始step_dict包装后放入统一输出队列"""

        def put_nowait(self,step):
            output_queue.put_nowait({"type":"rag_step","step":step})
    set_rag_step_queue(_RAGStepProxy())

    if len(messages)>50:
        summary=summarize_old_messages(model,messages[:40])
        messages=[
            SystemMessage(content=f"之前的对话摘要:\n{summary}")
        ]+messages[40:]

    messages.append(HumanMessage(content=user_text))

    full_response=""
    async def _agent_worker():
        """后台任务：运行agent并将内容chunk推入输出队列"""
        nonlocal full_response
        try:
            async for msg,metadata in agent.astream(
                {"messages":messages},
                stream_mode="messages",
                config={"recursion_limit":8}
            ):
                if not isinstance(msg,AIMessageChunk):
                    continue
                if getattr(msg,"tool_call_chunks",None):
                    continue
                content=""

                if isinstance(msg.content,str):
                    content=msg.content
                elif isinstance(msg.content,list):
                    for block in msg.content:
                        if isinstance(block,str):
                            content+=block
                        elif isinstance(block,dict) and block.get("type")=="text":
                            content+=block.get("text","")
                if content:
                    full_response+=content
                    await output_queue.put({"type":"content","content":content})
        except Exception as e:
            await output_queue.put({"type":"error","content":str(e)})
        finally:
            await output_queue.put(None)
    #启动后台任务
    agent_task=asyncio.create_task(_agent_worker())
    
    try:
        #主循环：持续从统一队列取事件并yield SSE
        #RAG步骤从工具执行期间通过call_soon_threadsafe实时入队，不需要等agent 产出chunk
        while True:
            event=await output_queue.get()
            if event is None:
                break
            yield f"data:{json.dumps(event)}\n\n"

    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
    
            
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()
    #获取RAG trace

    rag_context=get_last_rag_context(clear=True)

    rag_trace=rag_context.get("rag_trace")if rag_trace else None

    #发送trace信息

    if rag_trace:
        yield f"data:{json.dumps({'type':'trace','rag_trace':rag_trace})}\n\n"
    yield "data:[Done]\n\n"


    #保存对话
    messages.append(AIMessage(content=full_response))

    extra_message_data=[None]*(len(messages)-1)+[{"rag_trace":rag_trace}]

    storage.save(user_id,session_id,messages,extra_message_data=extra_message_data)