
import re
import json
from models import User
from pathlib import Path
from parent_chunk_store import ParentChunkStore
from document_loader import DocumentLoader
from agent import storage,chat_with_agent_stream,chat_with_agent
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from sqlalchemy.orm import Session
from embedding import embedding_service
import os
from auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, require_admin, resolve_role
from fastapi import APIRouter,Depends,HTTPException,UploadFile,File,BackgroundTasks
from fastapi.responses import StreamingResponse
from upload_jobs import upload_job_manager,delete_job_manager,DELETE_STEPS
from schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    CurrentUserResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
    LoginRequest,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR.parent/"data"
UPLOAD_DIR=DATA_DIR/"documents"


loader=DocumentLoader()

parent_chunk_store=ParentChunkStore()

milvus_manager=MilvusManager()
milvus_writer=MilvusWriter(embedding_service=embedding_service,milvus_manager=milvus_manager)


router=APIRouter()

def _remove_bm25_stats_for_filename(filename:str)->None:
    """删除Milvus中该文件对应chunk前，先从持久化BM25统计中扣减"""

    rows=milvus_manager.query_all(
        filter_expr=f"filename=='{filename}'",
        output_fields=["text"]
    )
    texts=[r.get("text")or ""for r in rows]
    embedding_service.increment_remove_documents(texts)


#用户注册
@router.post("auth/register",response_model=AuthResponse)
async def register(request:RegisterRequest,db:Session=Depends(get_db)):
    username=(request.username or "").strip()
    password=(request.password or "").strip()
    if not username or password:
        raise HTTPException(status_code=409,detail="用户名已存在")

    exists=db.query(User).filter(User.username==username).first()

    if exists:
        raise HTTPException(status_code=409,detail="用户已存在")
    

    role=resolve_role(requested_role=request.role,admin_code=request.admin_code)

    user=User(username=username,password_hash=get_password_hash(password=password),role=role)    

    db.add(user)

    db.commit()

    token=create_access_token(username=username,role=role)
    return AuthResponse(access_token=token,username=username,role=role)

#用户登录
@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}",response_model=SessionMessagesResponse)
async def get_session_messages(session_id:str,current_user:User=Depends(get_current_user)):
    "获取指定会话的所有消息"
    try:
        messages=[
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg["rag_trace"]
            )for msg in storage.get_session_messages(current_user.username,session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

@router.get("/sessions",response_model=SessionListResponse)
async def list_sessions(current_user:User=Depends(get_current_user)):
    """获取当前用户的所有会话列表"""

    try:
        sessions=[SessionInfo(**item)for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x:x.updated_at,reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

@router.delete("/sessions/{session_id}",response_model=SessionDeleteResponse)
async def delete_session(session_id:str,current_user:User=Depends(get_current_user)):
    """删除当前用户的指定会话"""

    try:
        deleted=storage.delete_session(user_id=current_user.username,session_id=session_id)
        if not deleted:
            raise HTTPException(status_code=404,detail="会话不存在")
        
        return SessionDeleteResponse(session_id=session_id,message="成功删除会话")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

@router.post("/chat",response_model=ChatResponse)
async def chat_endpoint(request:ChatRequest,current_user:User=Depends(get_current_user)):
    try:
        session_id=request.session_id or "default_session"
        resp=chat_with_agent(user_text=request.message,user_id=current_user.username,session_id=session_id)
    
        if isinstance(resp,dict):
            return ChatResponse(**resp)
        '''
        class ChatResponse(BaseModel):
            response: str
            rag_trace: Optional[RagTrace] = None
        因为这里ChatResonse中的rag_trace是可能为空的
        如果既有response和rag_trace那就是字典类型,从上面返回
        '''
        return ChatResponse(response=resp)
    except Exception as e:
        message=str(e)
        match=re.search(r"Error code:\s*(\d{3})",message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)



@router.post("/chat/stream")
async def chat_stream_endpoint(request:ChatRequest,current_user:User=Depends(get_current_user)):
    """流式对话"""

    async def event_generator():
        try:
            session_id=request.session_id or "default_session"

            async for chunk in chat_with_agent_stream(user_text=request.message,user_id=current_user.username,session_id=session_id):
                yield chunk                
        except Exception as e:
            error_data={"type":"error","content":str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no-cache,no-store,must-revalidate",
            "Connection":"keep-alive",
            "X-Accel-Buffering":"no"
        }
    )



def _is_supported_document(filename:str)->bool:
    file_lower=filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx","doc"))
        or file_lower.endswith((".xlsx",".xls"))
    )


async def _save_upload_file(file:UploadFile,file_path:str)->None:
    """按块写入上传文件 避免大文件一次性读入内存"""
    with open(file_path,'wb') as f :
        while True:
            chunk=await file.read(1024*1024)
            if not chunk:
                break
            f.write(chunk)


def _process_upload_job(job_id:str,file_path:str,filename:str)->None:
    """后台执行耗时的解析 分块 向量化入库并持续更新任务进展"""
    failed_step="cleanup"
    try:
        upload_job_manager.complete_step(job_id=job_id,step_key="upload",message="文件已保存到服务器")
        failed_step="cleanup"

        upload_job_manager.update_step(job_id=job_id,
                                       step_key="cleanup",
                                       percent=10,
                                       status="running",
                                       message="正在清理同名旧文档")
        milvus_manager.init_collection()

        delete_expr=f'filename=="{filename}'
        try:
            _remove_bm25_stats_for_filename(filename=filename)
        except:
            pass
        try:
            milvus_manager.delete(filter_expr=delete_expr)
        except Exception:
            pass

        try:
            parent_chunk_store.delete_by_filename(filename=filename)
        except Exception:
            pass

        upload_job_manager.update_step(job_id=job_id,step_key="parse",percent=5,status="running",message="正在解析文档并执行三级分块")

        new_docs=loader.load_document(file_path=file_path,filename=filename)

        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")
        parent_docs=[doc for doc in new_docs if int(doc.get("chunk_level",0)or 0)in(1,2)]
        leaf_docs=[doc for doc in new_docs if int(doc.get('chunk_level',0)or 0)==3]
        if not leaf_docs:
            raise HTTPException(status_code=500,detail="文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id=job_id,
            step_key="parse",
            message=f"解析完成:父级分块{len(parent_docs)}个，叶子分块:{len(leaf_docs)}个"
        )

        failed_step="parent_store"
        upload_job_manager.update_step(job_id=job_id,
                                       step_key="parent_store",
                                       percent=20,
                                       status="running",
                                       message="正在写入父级分块")

        parent_chunk_store.upsert_document(parent_docs)

        upload_job_manager.complete_step(
            job_id=job_id,
            step_key="parent_store",
            message=f"父级分块已入库:{len(parent_docs)}个"
        )
        failed_step="vectore_store"
        total_leaf=len(leaf_docs)

        upload_job_manager.update_step(
            job_id=job_id,
            step_key="vector_store",
            percent=0,
            status="running",
            message=f"正在向量化入库：0/{total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0
        )
        def _on_vector_progress(processed:int,total:int)->None:
            percent=round(processed *100/total)if total else 100
            upload_job_manager.update_step(
                job_id=job_id,
                step_key="vector_store",
                percent=percent,
                status="running",
                message=f"正在向量化入库:{processed}/{total}",
                total_chunks=total,
                processed_chunks=processed
            )

        milvus_writer.writer_document(documents=leaf_docs,progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id=job_id,step_key="vectore_store",message=f"向量化入库完成:{total_leaf}个叶子分块")
        upload_job_manager.complete_job(job_id=job_id,message=f"成功上传并处理{filename}")
    except Exception as e:
        upload_job_manager.fail_job(job_id=job_id,step_key=failed_step,error=str(e))



def _process_delete_job(job_id:str,filename:str)->None:
    """后台执行文档删除，并把每个删除阶段同步给前端行内进度卡片"""

    failed_step="prepare"

    try:
        failed_step="prepare"
        delete_job_manager.update_step(job_id=job_id,
                                       step_key="prepare",
                                       percent=20,
                                       status="running",
                                       message="正在初始化Milvus集合")
        milvus_manager.init_collection()
        delete_expr=f'filename=="{filename}"'
        delete_job_manager.complete_step(job_id=job_id,step_key="prepare",message="删除任务已创建")



        failed_step="bm25"
        delete_job_manager.update_step(job_id=job_id,step_key="bm25",percent=20,status="正在同步BM25统计")
        _remove_bm25_stats_for_filename(filename=filename)
        delete_job_manager.complete_step(job_id=job_id,step_key="prepare",message="正在通同步BM25统计")



        failed_step="milvus"
        delete_job_manager.update_step(job_id=job_id,step_key="milvus",percent=30,status="running",message="正在删除Milvus向量数据")
        result=milvus_manager.delete(delete_expr)
        deleted_count=result.get("deleted_count",0)if isinstance(result,dict) else 0
        delete_job_manager.complete_step(job_id=job_id,step_key="milvus",message=f"向量数据已删除：{deleted_count}条")


        failed_step="parent_store"
        delete_job_manager.update_step(job_id=job_id,step_key="parent_store",percent=30,status="running",message="正在删除PostgreSQL父级分块")
        parent_chunk_store.delete_by_filename(filename=filename)
        delete_job_manager.complete_step(job_id=job_id,step_key="parent_store",message="父级分块已删除")



        delete_job_manager.complete_job(job_id=job_id,message=f"已删除{filename},向量数据{deleted_count}条")
    except Exception as e:
        delete_job_manager.fail_job(job_id=job_id,step_key=failed_step,error=str(e))


@router.get("/documents",response_model=DocumentListResponse)
async def list_document(_:User=Depends(require_admin)):
    """获取已上传的文档列表（管理员）"""
    try:    
        milvus_manager.init_collection()

        results=milvus_manager.query(
            output_fields=["filename","file_type"],
            limit=10000
        )

        file_stats={}

        for item in results:
            filename=item.get("filename","")
            file_type=item.get("file_type","")

            if filename not in file_stats:
                file_stats[filename]={
                    "filename":filename,
                    "file_type":file_type,
                    "chunk_count":0
                }
            file_stats[filename]["chunk_count"]+=1

        documents=[DocumentInfo(**stats)for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"获取文档列表失败：{str(e)}")
    

@router.post("/documents/upload/async",response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks:BackgroundTasks,
    file:UploadFile=File(...),
    _:User=Depends(require_admin)
):
    """轻量版异步上传:文件罗盘后立即返回job_id 后台继续解析和向量化"""

    filename=file.filename or ""
    if not filename:
        raise HTTPException(status_code=400,detail="文件名不能为空")
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400,detail="仅支持PDF,Word,和excel文档")
   
    os.makedirs(UPLOAD_DIR,exist_ok=True)
    job=upload_job_manager.create_job(filename=filename)

    file_path=UPLOAD_DIR/ filename
    try:
        upload_job_manager.update_step(job["job_id"],"upload",1,"running","正在保存文件到服务器")
        await _save_upload_file(file,file_path)

        upload_job_manager.complete_step(job["job_id"],"upload","文件已上传，等待后台处理")
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"],"upload",f"文件上传失败:{e}")
        raise HTTPException(status_code=500,detail=f"文件上传失败:{str(e)}")
    

    background_tasks.add_task(_process_upload_job,job["job_id"],str(file_path),filename)

    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库"
    )

'''
查询上传任务状态
'''
@router.get("/documents/upload/jobs/{job_id}",response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id:str,_:User=Depends(require_admin)):
    job=upload_job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404,detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)
    

@router.get("/documents/upload/jobs",response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_:User=Depends(require_admin)):
    jobs=upload_job_manager.list_job()
    jobs.sort(key=lambda item:item.get("created_at",""),reverse=True)
    return [DocumentUploadJobResponse(**job)for job in jobs]


@router.delete("/documents/delete/async/{filename}",response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename:str,
    background_tasks:BackgroundTasks,
    _:User=Depends(require_admin)
):
    """轻量版异地删除 立即返回job_id，实际删除在后台执行"""

    job=delete_job_manager.create_job(
        filename=filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        messages="等待删除",
        completion_step="parent_step"
    )
    delete_job_manager.update_step(job["job_id"],"prepare",1,"running","删除任务已提交")
    background_tasks.add_task(_process_delete_job,job["job_id"],filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        messge=f"正在删除{filename}"
    )

@router.get("/documents/delete/jobs/{job_id}",response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id:str,_:User=Depends(require_admin)):
    job=delete_job_manager.get_job(job_id=job_id)
    if not job:
        raise HTTPException(status_code=4004,detail="删除不存在或过期")
    
    return DocumentDeleteJobResponse(
        **job
    )


@router.post("/documents/upload",response_model=DocumentUploadResponse)
async def upload_document(file:UploadFile=File(...),_:User=Depends(require_admin)):
    """上传文档并进行embedding  管理员"""

    try:
        filename=file.filename or ""
        file_lower=filename.lower()

        if not filename:
            raise HTTPException(status_code=400,detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx","doc"))
            or file_lower.endswith((".xlsx",".xls"))
        ):
            raise HTTPException(status_code=400,detail="仅支持PDF,Word,excel文件")

        os.makedirs(UPLOAD_DIR,exist_ok=True)

        milvus_manager.init_collection()

        delete_expr=f"filename=='{filename}'"
        try:
            _remove_bm25_stats_for_filename(filename=filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(filter_expr=delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename=filename)
        except Exception:
            pass    

        file_path=UPLOAD_DIR/filename

        with open(file_path,'wb') as f:
            content=await f.read()
            f.write(content)

        try:
            new_docs=loader.load_document(str(file_path),filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500,detail=f"文档处理失败:{doc_err}")
        
        if not new_docs:
            raise HTTPException(status_code=500,detail="文档处理失败，未能提取内容")
        

        parent_docs=[doc for doc in new_docs if int(doc.get("chunk_level",0)or 0)in(1,2)]
        leaf_docs=[doc for doc in new_docs if int(doc.get('chunk_level',0)or 0)==3]
        if not leaf_docs:
            raise HTTPException(status_code=500,detail="文档处理失败，未生成可检索叶子分块")
        parent_chunk_store.upsert_document(parent_docs)
        milvus_writer.writer_document(leaf_docs)
        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            messages=(
                f"成功上传并处理{filename}，叶子分块{len(leaf_docs)}个"
                f"父级分块{len(parent_docs)}个（存入PostgreSQL）"
            )
        )
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"文档上传失败:{str(e)}")

@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件，管理员）"""
    try:
        milvus_manager.init_collection()

        delete_expr = f'filename == "{filename}"'
        _remove_bm25_stats_for_filename(filename)
        result = milvus_manager.delete(delete_expr)
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")