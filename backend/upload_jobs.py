"""
上传任务进度管理
"""

from __future__ import annotations

from copy import deepcopy

from datetime import UTC,datetime

from threading import Lock

from typing import Literal,List,Dict,Any,Tuple,Optional

from uuid import uuid4


StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed"]


DEFAULT_STEPS = [
    ("upload", "文档上传"),
    ("cleanup", "清理旧版本"),
    ("parse", "解析与分块"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "向量化入库"),
]

DELETE_STEPS = [
    ("prepare", "准备删除"),
    ("bm25", "同步 BM25 统计"),
    ("milvus", "删除向量数据"),
    ("parent_store", "删除父级分块"),
]


def _now_iso()->str:
    return datetime.now(UTC).isoformat()

class UploadJobManager:
    def __init__(self):
        self._jobs:Dict[str,Dict]={}
        self._lock=Lock()


    def create_job(
        self,
        filename:str,
        *,#关键字参数分隔符，它后面的参数必须以关键字参数传入
        steps:List[Tuple[str,str]]|None=None,
        current_step:str="upload",
        message:str="等待上传",
        completion_step:str="vector_store",
    )->Dict:
        
        steps=steps or DEFAULT_STEPS
        job_id=uuid4().hex
        now=_now_iso()
        job={
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "current_step": current_step,
            "message": message,
            # 完成节点用于区分上传和删除，避免 complete_job 写死最后一步。
            "completion_step": completion_step,
            "total_chunks": 0,
            "processed_chunks": 0,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "steps":[
                {
                    "key":key,
                    "label":label,
                    "percent":0,
                    "status":"pending",
                    "message":""
                }for key,label in steps
            ]
        }

        with self._lock:
            self._jobs[job_id]=job
            return deepcopy(job) #深拷贝 所有对象都复 包括嵌套对象
        

    def get_job(self,job_id:str)->Dict | None:
        with self._lock:
            job=self._jobs.get(job_id)
            return deepcopy(job) if job else None


    def update_step(
        self,
        job_id:str,
        step_key:str,
        percent:int,
        status:StepStatus="running",
        message:str="",
        *,
        total_chunks:int |None=None,
        processed_chunks:int|None=None
    )->Dict | None:
        percent=max(0,min(100,int(percent)))

        with self._lock:
            job=self._jobs.get(job_id)
            if not job:
                return None
            step=self._find_step(job,step_key)
            
            if not step:
                return None
            

            step["percent"]=percent
            step["status"]=status
            step["message"]=message
            job["status"]="failed"if status =="failed"else "running"
            job["current_step"]=step_key
            job["message"]=message
            job["updated_at"]=_now_iso()


            if total_chunks is not None:
                job["total_chunks"]=int(total_chunks)
            if processed_chunks is not None:
                job["processed_chunks"]=int(processed_chunks)

            return deepcopy(job) 

    def complete_step(self,job_id:str,step_key:str,message:str="")->Dict|None:
        return self.update_step(job_id=job_id,step_key=step_key,percent=100,status="completed",message=message)
    

    def complete_job(self,job_id:str,message:str="文档入库完成")->Dict |None:
        with self._lock:
            job=self._jobs.get(job_id)
            if not job:
                return None
            for step in job["steps"]:
                if step["status"]!="failed":
                    step["percent"]=100
                    step["status"]="completed"

            job["status"]="completed"
            job["current_step"]=job.get("completion_step")or job["current_step"]
            job["message"]=message
            job["error"]=None
            job["updated_at"]=_now_iso()
            return deepcopy(job)


    def fail_job(self,job_id:str,step_key:str,error:str)->Dict |None:
        with self._lock:
            job=self._jobs.get(job_id)
            if not job:
                return None
            step=self._find_step(job,step_key)
            if step:
                step["status"]="failed"
                step["messages"]=error
            
            job["status"]="failed"
            job["current_step"]=step_key
            job["message"]=error
            job["error"]=error
            job["updated_at"]=_now_iso()
            return deepcopy(job)


    def list_job(self)->List[Dict]:
        with self._lock:
            return [deepcopy(job)for job in self._jobs.values()]


    @staticmethod
    def _find_step(job:Dict,step_key:str)->Dict |None:
        for step in job["steps"]:
            if step["key"]==step_key:
                return step
        return None
    

upload_job_manager = UploadJobManager()
delete_job_manager = UploadJobManager()