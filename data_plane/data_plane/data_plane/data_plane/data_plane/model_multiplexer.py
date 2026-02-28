import asyncio, logging, os, time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger("apex.data_plane.model_multiplexer")

class ProviderStatus(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DOWN     = "down"

@dataclass
class ProviderHealth:
    provider: str
    status: ProviderStatus = ProviderStatus.HEALTHY
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0

    def error_rate(self) -> float:
        return self.total_errors / self.total_calls if self.total_calls > 0 else 0.0

    def record_success(self, latency_ms: float):
        self.total_calls += 1
        self.consecutive_errors = 0
        self.avg_latency_ms = 0.2 * latency_ms + 0.8 * self.avg_latency_ms
        if self.status == ProviderStatus.DOWN:
            self.status = ProviderStatus.HEALTHY
            logger.info(f"[Multiplexer] {self.provider} recovered!")

    def record_error(self, error: str):
        self.total_calls += 1
        self.total_errors += 1
        self.consecutive_errors += 1
        self.last_error = error
        if self.consecutive_errors >= 5:
            self.status = ProviderStatus.DOWN
            logger.warning(f"[Multiplexer] {self.provider} MARKED DOWN!")
        elif self.consecutive_errors >= 2:
            self.status = ProviderStatus.DEGRADED

@dataclass
class MultiplexedResponse:
    content: str
    provider_used: str
    model_used: str
    latency_ms: float
    failover_count: int = 0
    error_chain: list = field(default_factory=list)

async def _call_openai(message: str, model: str, max_tokens: int) -> str:
    import openai
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens, temperature=0.7,
    )
    return resp.choices[0].message.content or ""

async def _call_anthropic(message: str, model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY", ""))
    resp = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
    )
    return resp.content[0].text if resp.content else ""

async def _call_local(message: str, model: str, max_tokens: int) -> str:
    import httpx
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{url}/api/chat", json={
            "model": model.replace("ollama/", ""),
            "messages": [{"role": "user", "content": message}],
            "stream": False, "options": {"num_predict": max_tokens},
        })
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

FAILOVER_CHAINS = {
    "gpt-4o-mini": [("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307"), ("local", "ollama/llama3")],
    "gpt-4o":      [("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-20241022"), ("openai", "gpt-4o-mini"), ("local", "ollama/llama3")],
    "claude-3-5-sonnet-20241022": [("anthropic", "claude-3-5-sonnet-20241022"), ("openai", "gpt-4o"), ("local", "ollama/llama3")],
    "claude-3-haiku-20240307":    [("anthropic", "claude-3-haiku-20240307"), ("openai", "gpt-4o-mini"), ("local", "ollama/llama3")],
}
CALLERS = {"openai": _call_openai, "anthropic": _call_anthropic, "local": _call_local}

class ModelMultiplexer:
    def __init__(self):
        self._health = {
            "openai":    ProviderHealth("openai"),
            "anthropic": ProviderHealth("anthropic"),
            "local":     ProviderHealth("local"),
        }

    async def complete(self, message: str, primary_model: str,
                       max_tokens: int = 1024, timeout_per_provider_s: float = 15.0) -> MultiplexedResponse:
        chain = FAILOVER_CHAINS.get(primary_model, FAILOVER_CHAINS["gpt-4o-mini"])
        error_chain = []
        failover_count = 0

        for provider, model_id in chain:
            health = self._health.get(provider)
            if health and health.status == ProviderStatus.DOWN:
                error_chain.append(f"{provider}:CIRCUIT_OPEN")
                failover_count += 1
                continue

            caller = CALLERS.get(provider)
            if not caller: continue
            start = time.time()
            try:
                content = await asyncio.wait_for(
                    caller(message, model_id, max_tokens), timeout=timeout_per_provider_s)
                latency_ms = (time.time() - start) * 1000
                self._health[provider].record_success(latency_ms)
                logger.info(f"[Multiplexer] {provider}/{model_id} OK {latency_ms:.0f}ms failovers={failover_count}")
                return MultiplexedResponse(content, provider, model_id, latency_ms, failover_count, error_chain)
            except asyncio.TimeoutError:
                self._health[provider].record_error("timeout")
                error_chain.append(f"{provider}:TIMEOUT")
            except Exception as e:
                self._health[provider].record_error(str(e))
                error_chain.append(f"{provider}:{type(e).__name__}")
                logger.warning(f"[Multiplexer] {provider}/{model_id} FAILED: {e}")
            failover_count += 1

        raise RuntimeError(f"All AI providers unavailable. Errors: {error_chain}")

    def get_health_report(self) -> dict:
        return {p: {"status": h.status.value, "error_rate_pct": round(h.error_rate() * 100, 1),
                    "avg_latency_ms": round(h.avg_latency_ms, 1), "total_calls": h.total_calls}
                for p, h in self._health.items()}

    def force_reset_provider(self, provider: str):
        if provider in self._health:
            self._health[provider].status = ProviderStatus.HEALTHY
            self._health[provider].consecutive_errors = 0

_mux: Optional[ModelMultiplexer] = None

def get_multiplexer() -> ModelMultiplexer:
    global _mux
    if _mux is None:
        _mux = ModelMultiplexer()
        logger.info("[Multiplexer] Initialized. OpenAI → Anthropic → Local failover ready.")
    return _mux
