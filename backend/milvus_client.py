"""
Milvus 客户端  支持密集向量与稀疏向量检索
"""


from pymilvus import MilvusClient,DataType,AnnSearchRequest,RRFRanker
import threading

import os
from typing import Callable,TypeVar,List,Dict
from dotenv import load_dotenv
load_dotenv()


QUERY_MAX_LIMIT=16384

#声明一个类型变量
T=TypeVar("T")


class MilvusManager:
    def __init__(self):
        self.host=os.getenv("MILVUS_HOST","localhost")
        self.port=os.getenv("MILVUS_PORT","19530")
        self.collection_name=os.getenv("MILVUS_COLLECTION","embedding_collection")
        self.url=f"http://{self.host}:{self.port}"
        self.client=None
        self._client_lock=threading.RLock()


    def _get_client(self)->MilvusClient:
        with self._client_lock:
            if self.client is None:
                self.client=MilvusClient(self.url)
            return self.client
        
    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "closed channel" in str(exc).lower()

    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass
    def _reset_client(self, failed_client=None) -> None:
        with self._client_lock:
            if self.client is None:
                return
            if failed_client is not None and self.client is not failed_client:
                return
            client = self.client
            self.client = None

        self._close_client(client)
    
    def _run_with_reconnect(self,operation:Callable[[MilvusClient],T])->T:
        '''
         Callable[[参数类型, ...], 返回类型]
        
        '''
        client=self._get_client()
        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise
            self._reset_client(client)
            return operation(self._get_client())
        

    def init_collection(self,dense_dim:int|None=None):
        if dense_dim is None:
            dense_dim=int(os.getenv("DENSE_EMBEDDING_DIM","1024"))
        def _init(client:MilvusClient)->None:
            exists = client.has_collection(self.collection_name)

            # 如果 collection 已存在但缺少索引，则删掉重建（索引是 load 的前提）
            if exists:
                indexes = client.list_indexes(self.collection_name)
                required_fields = {"dense_embedding", "sparse_embedding"}
                if not required_fields.issubset(set(indexes)):
                    client.drop_collection(self.collection_name)
                    exists = False

            if not exists:
                schema=client.create_schema(auto_id=True,enable_dynamic_field=True)
                # 主键
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

                # 密集向量（来自 embedding 模型）
                schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)

                # 稀疏向量（来自 BM25）
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

                # 文本和元数据字段
                schema.add_field("text", DataType.VARCHAR, max_length=2000)
                schema.add_field("filename", DataType.VARCHAR, max_length=255)
                schema.add_field("file_type", DataType.VARCHAR, max_length=50)
                schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
                schema.add_field("page_number", DataType.INT64)
                schema.add_field("chunk_idx", DataType.INT64)

                # Auto-merging 所需层级字段
                schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("chunk_level", DataType.INT64)

                index_params=client.prepare_index_params()

                #密集向量索引
                index_params.add_index(
                    field_name="dense_embedding",
                    index_type="HNSW",
                    metric_type="IP",
                    params={"M":16,"efConstruction":256}
                )
                '''
                M:每个节点可连接的最大邻居数量。
                efConstruction:索引构建过程中考虑连接的候选邻居数量。
                '''

                #稀疏向量索引
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                    params={"drop_ratio_build":0.2}
                )

                client.create_collection(collection_name=self.collection_name,schema=schema,index_params=index_params)
            # 确保 collection 已加载到内存
            client.load_collection(self.collection_name)
        self._run_with_reconnect(_init)

    def insert(self,data:List[Dict]):
        '''插入数据到Milvus'''
        return self._run_with_reconnect(lambda client:client.insert(self.collection_name,data))
    
    def query(
            self,
            filter_expr:str="",
            output_fields:List[str]=None,
            limit:int=10000,
            offset:int=0
    ):
        '''
        查询数据  注意是查询 表示搜索
        '''
        return self._run_with_reconnect(
            lambda client:client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["filename","file_type"],
                limit=min(limit,QUERY_MAX_LIMIT),
                offset=offset
            )
        )
    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """分页拉取匹配 filter 的全部行，避免单次 limit 超过服务端窗口。"""
        fields = output_fields or ["filename", "file_type"]
        out: list = []
        offset = 0
        while True:
            batch = self._run_with_reconnect(
                lambda client: client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
            )
            if not batch:
                break
            out.extend(batch)
            if len(batch) < QUERY_MAX_LIMIT:
                break
            offset += len(batch)
        return out
    
    def get_chunks_by_ids(self,chunk_ids:List[str])->List[str]:
        ids=[item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids=",".join([f'"{item}"'for item in ids])

        filter_expr=f"chunk_id in [{quoted_ids}]"

        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids)
        )
    
    def hybrid_retrieve(self,dense_embedding:List[float],sparse_embedding:dict,top_k:int=5,rrf_k:int=60,filter_expr:str="")->List[Dict]:
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]

        dense_search=AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            limit=top_k*2,
            expr=filter_expr,
            param={"metric_type": "IP", "params": {"ef": 64}},
        )

        # 稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        #使用RRF排序算法
        reranker=RRFRanker(k=rrf_k)
        results=self._run_with_reconnect(
            lambda client:client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search,sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields
            )
        )
        #格式化返回结果
        formatted_results=[]
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })
        return formatted_results
    
    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        仅使用密集向量检索（降级模式，用于稀疏向量不可用时）
        """
        results = self._run_with_reconnect(
            lambda client: client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
            )
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results
    
    def delete(self,filter_expr:str):
        return self._run_with_reconnect(
            lambda client:client.delete(
                collection_name=self.collection_name,filter=filter_expr
            )
        )
    
    def has_collection(self)->bool:
        '''检查集合是否存在'''
        return self._run_with_reconnect(lambda client:client.has_collection(self.collection_name))
    
    
    def drop_collection(self):
        def _drop(client:MilvusClient)->None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run_with_reconnect(_drop)