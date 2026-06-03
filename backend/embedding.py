"""文本向量化服务 - 支持密集向量和稀疏向量（BM25），词表与 df 持久化 + 增量更新"""
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 默认状态文件路径: backend/../data/bm25_state.json
_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    """
    创建密集向量嵌入器（基于 HuggingFace 本地模型）
    - EMBEDDING_MODEL: 模型名称，默认 BAAI/bge-m3（支持中英文）
    - EMBEDDING_DEVICE: 运行设备，默认 cpu（可选 cuda）
    """
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # L2 归一化，利于向量相似度计算
    )


class EmbeddingService:
    """文本向量化服务 - 密集向量本地模型 + BM25 稀疏向量（持久化统计）"""

    def __init__(self, state_path: Path | str | None = None):
        self._embedder = _create_dense_embedder()  # 密集向量嵌入器
        # 状态文件路径：优先参数 > 环境变量 > 默认路径
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()  # 线程锁，保证并发安全

        # BM25 算法参数（经典经验值）
        self.k1 = 1.5   # 词频饱和参数：控制词频增长对得分的影响
        self.b = 0.75   # 文档长度归一化参数：0 完全忽略长度，1 完全归一化

        # 词表：token -> 索引ID（BM25 稀疏向量的维度）
        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0  # 下一个可用的词表索引

        # 文档频率统计：token -> 包含该词的文档数
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0           # 总文档数
        self._sum_token_len = 0        # 所有文档的 token 总数
        self._avg_doc_len = 1.0       # 平均文档长度（用于 BM25 归一化）

        # 启动时加载已持久化的状态（词表、df 等）
        self._load_state()

    def _recompute_avg_len(self) -> None:
        """重新计算平均文档长度"""
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        """
        从磁盘加载 BM25 状态（词表、文档频率等）
        仅加载 version=1 的格式，兼容未来格式升级
        """
        path = self._state_path
        if not path.is_file():
            return  # 文件不存在则跳过（首次运行）

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # 解析失败或文件读取错误则跳过

        if raw.get("version") != 1:
            return  # 不支持的版本则跳过

        # 恢复各项统计
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))

        # 恢复词表计数器（最大索引 + 1，确保不冲突）
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0

        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        """
        将当前状态写入磁盘（先写临时文件再原子替换，避免写入中断导致损坏）
        调用方需持有 _lock
        """
        self._state_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)  # 原子替换，避免文件损坏

    def _persist(self) -> None:
        """线程安全的持久化封装"""
        with self._lock:
            self._persist_unlocked()

    def increment_add_documents(self, texts: list[str]) -> None:
        """
        增量添加文档到 BM25 统计（用于索引新文档）
        - 更新词表（OOV token 分配新索引）
        - 更新文档频率 df
        - 更新总文档数、平均长度
        - 持久化到磁盘
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                # set(tokens) 去重，每个 token 仅计数一次（df 统计的是文档频率，不是词频）
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """
        从 BM25 统计中移除文档（与 increment_add_documents 对称，用于删除场景）
        - 词表索引不回收（避免与 Milvus 中已存在的旧稀疏向量维度冲突）
        - 仅更新 df、文档数、平均长度
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]  # df 为 0 时移除，节省内存
            self._recompute_avg_len()
            self._persist_unlocked()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        获取密集向量嵌入（HuggingFace 本地模型）
        返回: list[向量] 每项向量已 L2 归一化
        """
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e

    def tokenize(self, text: str) -> list[str]:
        """
        分词器：支持中文（单字）和英文（单词）
        - 中文: 每个汉字作为一个 token（如 "你好" -> ["你", "好"]）
        - 英文: 提取连续字母序列（如 "hello world" -> ["hello", "world"]）
        - 其他字符（数字、标点等）直接丢弃
        """
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")    # Unicode 中文字符范围
        english_pattern = re.compile(r"[a-zA-Z]+")           # 英文字母序列

        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                # 中文字符：单字作为 token
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                # 英文字符：提取完整单词
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                # 其他字符（标点、数字等）：跳过
                i += 1
        return tokens

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        """
        计算单个文本的 BM25 稀疏向量（需外部加锁）
        返回: (稀疏向量 dict, 词表是否有变化)
        稀疏向量格式: {token_index: bm25_score}
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)  # token -> 词频
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            # 动态扩展词表（首次出现分配新索引）
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)

            # BM25 IDF 计算（使用 Lucene 公式变体，避免 df=0 时的奇异性）
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # BM25 得分公式: IDF * (tf * (k1+1)) / (tf + k1 * (1-b + b*dl/avg))
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator

            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        """
        获取单个文本的 BM25 稀疏向量（线程安全）
        """
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()  # 词表扩展时自动持久化
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """
        批量获取 BM25 稀疏向量（线程安全，批量持久化）
        """
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """
        同时获取密集向量和稀疏向量（BM25）
        用于同时需要两种向量的混合检索场景
        """
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings


# 全局单例：整个进程共用同一份 BM25 状态（写入和检索共用）
# 确保并发安全，避免多实例导致状态不一致
embedding_service = EmbeddingService()