import asyncio, json, logging, time, uuid
from typing import AsyncGenerator, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("apex.data_plane.stream_manager")

class StreamEventType(str, Enum):
    TOKEN     = "token"
    DONE      = "done"
    ERROR     = "error"
    METADATA  = "metadata"
    HEARTBEAT = "heartbeat"

@dataclass
class StreamEvent:
    event_type: StreamEventType
    data: Any
    stream_id: str
    sequence: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_sse(self) -> str:
        payload = {
            "type": self.event_type.value,
            "data": self.data,
            "stream_id": self.stream_id,
            "seq": self.sequence,
            "ts": round(self.timestamp, 3),
        }
        return f"event: {self.event_type.value}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def to_ws(self) -> str:
        return json.dumps({"t": self.event_type.value, "d": self.data,
                           "sid": self.stream_id, "seq": self.sequence}, ensure_ascii=False)


async def stream_openai(
    message: str,
    model: str = "gpt-4o-mini",
    stream_id: Optional[str] = None,
    system_prompt: str = "You are Apex AI, a helpful and intelligent assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[StreamEvent, None]:
    import openai, os
    sid = stream_id or str(uuid.uuid4())
    seq = 0
    full_text = []
    start_time = time.time()

    yield StreamEvent(StreamEventType.METADATA, {"model": model, "stream_id": sid}, sid, seq)
    seq += 1

    try:
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        async with client.chat.completions.stream(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": message}],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_text.append(delta.content)
                    yield StreamEvent(StreamEventType.TOKEN, {"text": delta.content, "index": seq}, sid, seq)
                    seq += 1

        total = "".join(full_text)
        latency_ms = (time.time() - start_time) * 1000
        yield StreamEvent(StreamEventType.DONE, {"full_text": total,
            "latency_ms": round(latency_ms, 1), "model": model}, sid, seq)
        logger.info(f"[StreamManager] {sid} done in {latency_ms:.0f}ms")

    except Exception as e:
        logger.error(f"[StreamManager] {sid} ERROR: {e}")
        yield StreamEvent(StreamEventType.ERROR, {"error": str(e)}, sid, seq)


async def stream_anthropic(
    message: str,
    model: str = "claude-3-5-sonnet-20241022",
    stream_id: Optional[str] = None,
    system_prompt: str = "You are Apex AI, a helpful and intelligent assistant.",
    max_tokens: int = 2048,
) -> AsyncGenerator[StreamEvent, None]:
    import anthropic, os
    sid = stream_id or str(uuid.uuid4())
    seq = 0
    full_text = []
    start_time = time.time()

    yield StreamEvent(StreamEventType.METADATA, {"model": model, "provider": "anthropic"}, sid, seq)
    seq += 1

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY", ""))
        async with client.messages.stream(
            model=model,
            messages=[{"role": "user", "content": message}],
            system=system_prompt,
            max_tokens=max_tokens,
        ) as stream:
            async for text_chunk in stream.text_stream:
                full_text.append(text_chunk)
                yield StreamEvent(StreamEventType.TOKEN, {"text": text_chunk, "index": seq}, sid, seq)
                seq += 1

        total = "".join(full_text)
        latency_ms = (time.time() - start_time) * 1000
        yield StreamEvent(StreamEventType.DONE, {"full_text": total,
            "latency_ms": round(latency_ms, 1), "model": model}, sid, seq)

    except Exception as e:
        logger.error(f"[StreamManager:Anthropic] {sid} ERROR: {e}")
        yield StreamEvent(StreamEventType.ERROR, {"error": str(e)}, sid, seq)


async def dispatch_stream(
    message: str,
    model: str,
    stream_id: Optional[str] = None,
    **kwargs,
) -> AsyncGenerator[StreamEvent, None]:
    if model.startswith("claude"):
        async for event in stream_anthropic(message, model, stream_id, **kwargs):
            yield event
    else:
        async for event in stream_openai(message, model, stream_id, **kwargs):
            yield event


async def sse_response_generator(
    source_generator: AsyncGenerator[StreamEvent, None],
    heartbeat_interval_s: float = 15.0,
) -> AsyncGenerator[str, None]:
    source_queue: asyncio.Queue = asyncio.Queue()

    async def _feed():
        async for event in source_generator:
            await source_queue.put(event.to_sse())
        await source_queue.put(None)

    asyncio.create_task(_feed())
    hb_seq = 0
    while True:
        try:
            item = await asyncio.wait_for(source_queue.get(), timeout=heartbeat_interval_s)
            if item is None:
                break
            yield item
        except asyncio.TimeoutError:
            hb_seq += 1
            hb = StreamEvent(StreamEventType.HEARTBEAT, {"ping": hb_seq}, "hb", hb_seq)
            yield hb.to_sse()
