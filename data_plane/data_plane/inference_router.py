import logging, re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger("apex.data_plane.inference_router")

class ModelTier(str, Enum):
    MINI = "mini"
    PRO  = "pro"
    LOCAL = "local"

@dataclass
class ModelSpec:
    model_id: str
    provider: str
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context_tokens: int
    avg_latency_ms: int
    supports_streaming: bool = True

MODEL_REGISTRY = {
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "openai", ModelTier.MINI, 0.00015, 0.00060, 128000, 400),
    "gpt-4o":      ModelSpec("gpt-4o",      "openai", ModelTier.PRO,  0.005,   0.015,   128000, 800),
    "claude-3-haiku-20240307":     ModelSpec("claude-3-haiku-20240307",     "anthropic", ModelTier.MINI, 0.00025, 0.00125, 200000, 500),
    "claude-3-5-sonnet-20241022":  ModelSpec("claude-3-5-sonnet-20241022",  "anthropic", ModelTier.PRO,  0.003,   0.015,   200000, 900),
    "ollama/llama3": ModelSpec("ollama/llama3", "local", ModelTier.LOCAL, 0.0, 0.0, 8000, 2000, supports_streaming=False),
}

HIGH_COMPLEXITY = [r'\banalyze\b', r'\brefactor\b', r'\barchitecture\b', r'\boptimize\b',
                   r'\bcompare\b', r'\bdebug\b', r'\bcomprehensive\b', r'\bprove\b']
LOW_COMPLEXITY  = [r'^hi\b', r'^hello\b', r'^hey\b', r'\bwhat is\b.{1,30}\?$']

INTENT_TIER_MAP = {
    "chat": ModelTier.MINI, "knowledge": ModelTier.MINI,
    "code": ModelTier.PRO,  "math": ModelTier.PRO,
    "image": ModelTier.MINI, "video": ModelTier.PRO,
    "translation": ModelTier.MINI,
}

@dataclass
class RoutingDecision:
    selected_model: str
    provider: str
    tier: ModelTier
    reason: str
    estimated_cost_usd: float
    fallback_chain: list = field(default_factory=list)
    complexity_score: int = 0

def _tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _complexity(message: str) -> int:
    score = 0
    msg = message.lower()
    t = _tokens(message)
    if t > 500: score += 25
    elif t > 200: score += 15
    elif t > 100: score += 8
    if "```" in message or "def " in message or "class " in message: score += 30
    if message.count('\n') > 3: score += 15
    for p in HIGH_COMPLEXITY:
        if re.search(p, msg): score += 12; break
    for p in LOW_COMPLEXITY:
        if re.search(p, msg): score -= 20; break
    return max(0, min(100, score))

def _pick_model(tier: ModelTier, intent: str, unhealthy: set) -> Optional[str]:
    choices = {
        ModelTier.MINI:  ["gpt-4o-mini", "claude-3-haiku-20240307"],
        ModelTier.PRO:   ["claude-3-5-sonnet-20241022", "gpt-4o"] if intent == "code" else ["gpt-4o", "claude-3-5-sonnet-20241022"],
        ModelTier.LOCAL: ["ollama/llama3"],
    }
    for m in choices.get(tier, ["gpt-4o-mini"]):
        if m not in unhealthy and m in MODEL_REGISTRY:
            return m
    return None

def _fallback(primary: str, unhealthy: set) -> list:
    all_m = ["gpt-4o-mini", "claude-3-haiku-20240307", "gpt-4o", "claude-3-5-sonnet-20241022", "ollama/llama3"]
    chain = [m for m in all_m if m != primary and m not in unhealthy][:3]
    if "ollama/llama3" not in chain: chain.append("ollama/llama3")
    return chain

def _cost(spec: ModelSpec, tokens: int) -> float:
    out = min(tokens * 2, 2000)
    return round(tokens / 1000 * spec.cost_per_1k_input + out / 1000 * spec.cost_per_1k_output, 6)

async def route_inference(
    message: str,
    intent: str = "chat",
    user_tier: str = "free",
    requested_model: Optional[str] = None,
    unhealthy_models: Optional[set] = None,
) -> RoutingDecision:
    unhealthy = unhealthy_models or set()
    score = _complexity(message)
    tokens = _tokens(message)

    if requested_model and requested_model in MODEL_REGISTRY and requested_model not in unhealthy:
        spec = MODEL_REGISTRY[requested_model]
        return RoutingDecision(requested_model, spec.provider, spec.tier,
                               "User requested", _cost(spec, tokens),
                               _fallback(requested_model, unhealthy), score)

    intent_tier = INTENT_TIER_MAP.get(intent, ModelTier.MINI)
    if score >= 60 or intent_tier == ModelTier.PRO:
        tier = ModelTier.PRO
        reason = f"Complex (score={score})"
    else:
        tier = ModelTier.MINI
        reason = f"Simple (score={score})"

    if user_tier == "free" and tier == ModelTier.PRO:
        tier = ModelTier.MINI
        reason += " [free tier]"

    selected = _pick_model(tier, intent, unhealthy) or "ollama/llama3"
    spec = MODEL_REGISTRY[selected]

    return RoutingDecision(selected, spec.provider, spec.tier, reason,
                           _cost(spec, tokens), _fallback(selected, unhealthy), score)
