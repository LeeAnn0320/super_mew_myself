from embedding import EmbeddingService,embedding_service as _default_embedding_service
from milvus_client import MilvusManager

class MilvusWriter:
    '''文档写入Milvus'''

    def __init__(self,embedding_service:EmbeddingService=None,milvus_manager:MilvusManager=None):
        self.embedding_service=embedding_service or _default_embedding_service
        self.milvus_manager=milvus_manager or MilvusManager()
        

    def writer_document(self,documents:list[dict],batch_size:int=50,progress_callback=None):
        if not documents:
            return
        self.milvus_manager.init_collection()

        all_texts=[doc["text"]for doc in documents]
        self.embedding_service.increment_add_documents(all_texts)

        total=len(documents)

        for i in range(0,total,batch_size):
            batch=documents[i:i+batch_size]
            texts=[doc["text"]for doc in batch]

            #同时生成密集向量与稀疏向量
            dense_embeddings,sparse_embeddings=self.embedding_service.get_all_embeddings(texts)
            insert_data=[{
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }for doc,dense_emb,sparse_emb in zip(batch,dense_embeddings,sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)
            #每个批次写入后更新进度，前端据此展示 向量化入库
            if progress_callback:
                processed=min(i,batch_size,total)
                progress_callback(processed,total)