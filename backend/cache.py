import redis
import json
import os
from typing import Optional, Any


class RedisCache:
    """
    Redis 缓存管理器
    用于存储、读取、删除 JSON 格式的缓存数据
    """

    def __init__(self):
        """
        初始化配置参数（从环境变量读取）
        - REDIS_URL: Redis 连接地址，默认 redis://localhost:6379/0
        - REDIS_KEY_PREFIX: 缓存键前缀，避免多项目键冲突，默认 'supermew'
        - REDIS_DEFAULT_TTL: 默认过期时间（秒），默认 3600（1小时）
        """
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.key_prefix = os.getenv('REDIS_KEY_PREFIX', 'supermew')
        self.default_ttl = int(os.getenv('REDIS_DEFAULT_TTL', '3600'))
        self._client = None  # Redis 客户端实例（延迟初始化）  
        '''
        懒加载   先设为None 第一次调用_get_client时才创建连接，后续复用同一连接
        '''

    def _get_client(self):
        """
        获取 Redis 客户端（懒加载单例模式）
        首次调用时创建连接，后续复用同一连接
        """
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        """
        生成带前缀的缓存键
        格式: {key_prefix}:{key}
        防止不同项目或模块的键名冲突
        """
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        """
        从 Redis 读取 JSON 数据并自动反序列化
        Args:
            key: 缓存键名
        Returns:
            反序列化后的数据，键不存在或出错时返回 None
        """
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception as e:
            print(f"Error getting key {key} from Redis: {e}")
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        将数据序列化为 JSON 后存入 Redis
        Args:
            key: 缓存键名
            value: 要存储的任意 Python 对象
            ttl: 可选过期时间（秒），默认使用 default_ttl
        """
        try:
            payload = json.dumps(value, ensure_ascii=False)  #python 对象转换为json字符串
            self._get_client().setex(self._key(key), ttl or self.default_ttl, payload)
        except Exception as e:
            print(f"Error setting key {key} in Redis: {e}")
            return

    def delete(self, key: str) -> None:
        """
        删除指定的缓存键
        Args:
            key: 要删除的缓存键名
        """
        try:
            self._get_client().delete(self._key(key))
        except Exception as e:
            print(f"Error deleting key {key} from Redis: {e}")
            return

    def delete_pattern(self, pattern: str) -> None:
        """
        批量删除所有匹配通配符模式的键
        Args:
            pattern: 通配符模式，如 'user_*' 删除所有以 'user_' 开头的键
        """
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                self._get_client().delete(*keys)
        except Exception as e:
            print(f"Error deleting keys with pattern {pattern} from Redis: {e}")
            return


# 模块级单例实例，直接 import cache 即可使用
cache = RedisCache()