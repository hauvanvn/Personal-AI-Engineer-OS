"""Redis-based response cache for the Orchestrator API.

Stores and retrieves cached JSON payloads keyed by a normalized SHA-256
hash of the user query.  All keys live under the ``ai_os:chat_cache:``
prefix and expire after a configurable TTL (default 3600 seconds).

The module is designed to fail gracefully: every public method catches
connection errors so that a temporarily unreachable Redis instance never
crashes the API server.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "ai_os:chat_cache:"
DEFAULT_TTL_SECONDS = 3600


class RedisCacheManager:
    """Thin wrapper around a Redis connection for caching chat responses."""

    def __init__(self) -> None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self._client: redis.Redis = redis.from_url(
                redis_url, decode_responses=True
            )
            # Verify connectivity on startup (best-effort)
            self._client.ping()
            logger.info("RedisCacheManager connected to %s", redis_url)
        except redis.RedisError as exc:
            logger.warning(
                "RedisCacheManager failed to connect to Redis at %s: %s. "
                "Caching will be disabled until Redis becomes reachable.",
                redis_url,
                exc,
            )
            self._client = redis.from_url(redis_url, decode_responses=True)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Return a deterministic SHA-256 hex digest for *query*."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _make_key(self, query: str) -> str:
        return f"{CACHE_KEY_PREFIX}{self._normalize_query(query)}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Return the cached payload for *query*, or ``None`` on miss."""
        try:
            raw = self._client.get(self._make_key(query))
            if raw is not None:
                return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.warning("Redis GET failed: %s", exc)
        return None

    def store_response(
        self,
        query: str,
        response: str,
        tool_traces: list,
        citations: list,
        ttl: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Persist a JSON payload in Redis with the given TTL."""
        payload = {
            "response": response,
            "tool_traces": tool_traces,
            "citations": citations,
            "cached": True,
        }
        try:
            self._client.setex(
                self._make_key(query),
                ttl,
                json.dumps(payload, ensure_ascii=False),
            )
        except redis.RedisError as exc:
            logger.warning("Redis SETEX failed: %s", exc)

    def invalidate_all_chat_cache(self) -> int:
        """Delete every key matching ``ai_os:chat_cache:*``.

        Uses SCAN to avoid blocking Redis with a single KEYS command.
        Returns the number of keys deleted.
        """
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor=cursor, match=f"{CACHE_KEY_PREFIX}*", count=200
                )
                if keys:
                    deleted += self._client.delete(*keys)
                if cursor == 0:
                    break
            if deleted:
                logger.info(
                    "Invalidated %d cached chat response(s).", deleted
                )
        except redis.RedisError as exc:
            logger.warning("Redis cache invalidation failed: %s", exc)
        return deleted
