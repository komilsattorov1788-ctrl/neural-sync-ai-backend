import json, hashlib, logging, time, asyncio
from typing import Optional
import redis.asyncio as aioredis

logger = logging.getLogger("apex.data_plane.semantic_cache")
EXACT_CACHE_PREFIX = "apex:cache:exact:"
SEMANTIC_CACHE_PREFIX = "apex:cache:semantic:"
STATS_KEY = "apex:cache:stats"
EXACT_TTL_SECONDS = 3600
SEMANTIC_TTL_SECONDS = 21600

class SemanticCache:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._hits = 0
        self._misses = 0

    def _exact_key(self, model: str, message: str) -> str:
        raw = f"{model}::{message.strip().lower()}"
        return EXACT_CACHE_PREFIX + hashlib.sha256(raw.encode()).hexdigest()

    async def get_exact(self, model: str, message: str) -> Optional[dict]:
        key = self._exact_key(model, message)
        try:
            raw = await self.redis.get(key)
            if raw:
                self._hits += 1
                data = json.loads(raw)
                data["_cache_tier"] = "exact"
                data["_cache_hit"] = True
                return data
        except Exception as e:
            logger.warning(f"[SemanticCache] exact get error: {e}")
        self._misses += 1
        return None

    async def set_exact(self, model: str, message: str, response: dict) -> None:
        key = self._exact_key(model, message)
        try:
            payload = {**response, "_cached_at": time.time()}
            await self.redis.setex(key, EXACT_TTL_SECONDS, json.dumps(payload))
        except Exception as e:
            logger.warning(f"[SemanticCache] exact set error: {e}")

    def _simhash(self, text: str) -> str:
        text = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in text.strip().lower())
        words = text.split()
        features = set()
        for i in range(len(words)):
            features.add(words[i])
            if i + 1 < len(words):
                features.add(f"{words[i]}_{words[i+1]}")
        if not features:
            return hashlib.md5(text.encode()).hexdigest()[:16]
        vector = [0] * 64
        for feature in features:
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            for i in range(64):
                vector[i] += 1 if h & (1 << i) else -1
        simhash = sum((1 << i) for i in range(64) if vector[i] > 0)
        bucket = (simhash >> 16) & 0xFFFFFFFFFFFF
        return f"{bucket:012x}"

    def _semantic_key(self, model: str, message: str) -> str:
        return SEMANTIC_CACHE_PREFIX + f"{model}:{self._simhash(message)}"

    async def get_semantic(self, model: str, message: str) -> Optional[dict]:
        key = self._semantic_key(model, message)
        try:
            raw = await self.redis.get(key)
            if raw:
                self._hits += 1
                data = json.loads(raw)
                data["_cache_tier"] = "semantic"
                data["_cache_hit"] = True
                return data
        except Exception as e:
            logger.warning(f"[SemanticCache] semantic get error: {e}")
        self._misses += 1
        return None

    async def set_semantic(self, model: str, message: str, response: dict) -> None:
        key = self._semantic_key(model, message)
        try:
            payload = {**response, "_cached_at": time.time()}
            await self.redis.setex(key, SEMANTIC_TTL_SECONDS, json.dumps(payload))
        except Exception as e:
            logger.warning(f"[SemanticCache] semantic set error: {e}")

    async def get(self, model: str, message: str) -> Optional[dict]:
        result = await self.get_exact(model, message)
        if result:
            return result
        return await self.get_semantic(model, message)

    async def set(self, model: str, message: str, response: dict) -> None:
        await asyncio.gather(
            self.set_exact(model, message, response),
            self.set_semantic(model, message, response),
            return_exceptions=True
        )

    async def get_stats(self) -> dict:
        try:
            raw = await self.redis.hgetall(STATS_KEY)
            stats = {k.decode(): int(v) for k, v in raw.items()}
        except Exception:
            stats = {}
        total = self._hits + self._misses
        return {
            "session_hits": self._hits,
            "session_misses": self._misses,
            "hit_rate_pct": round(self._hits / total * 100, 2) if total > 0 else 0.0,
            "cost_saved_usd": round(self._hits * 0.005, 4),
            "redis_stats": stats,
        }

_cache_instance: Optional[SemanticCache] = None

async def get_semantic_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        from security.redis_limiter import get_redis
        redis = await get_redis()
        _cache_instance = SemanticCache(redis)
    return _cache_instance
