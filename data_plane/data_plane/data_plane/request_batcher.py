import asyncio, logging, time, uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Awaitable

logger = logging.getLogger("apex.data_plane.request_batcher")

MAX_BATCH_SIZE  = 10
BATCH_WINDOW_MS = 50
MAX_QUEUE_SIZE  = 500
BATCH_TIMEOUT_S = 30.0

@dataclass
class BatchRequest:
    request_id: str
    payload: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)
    priority: int = 0

@dataclass
class BatchResult:
    request_id: str
    result: Optional[Any] = None
    error: Optional[Exception] = None
    latency_ms: float = 0.0
    batch_size: int = 1
    was_batched: bool = False

class BatchQueue:
    def __init__(self, model_id: str, handler: Callable, max_batch_size: int = MAX_BATCH_SIZE, batch_window_ms: int = BATCH_WINDOW_MS):
        self.model_id = model_id
        self.handler = handler
        self.max_batch_size = max_batch_size
        self.batch_window_s = batch_window_ms / 1000.0
        self._queue: list = []
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._total_requests = 0
        self._total_batches = 0
        self._total_batched = 0

    async def start(self):
        if self._running: return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop(), name=f"batcher-{self.model_id}")
        logger.info(f"[BatchQueue:{self.model_id}] Started.")

    async def stop(self):
        self._running = False
        self._event.set()
        if self._task:
            try: await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError: self._task.cancel()

    async def enqueue(self, payload: dict, priority: int = 0) -> Any:
        if len(self._queue) >= MAX_QUEUE_SIZE:
            raise RuntimeError(f"Queue full ({MAX_QUEUE_SIZE}). Backpressure active.")
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        req = BatchRequest(str(uuid.uuid4()), payload, future, priority=priority)
        async with self._lock:
            self._queue.append(req)
            self._total_requests += 1
            if len(self._queue) >= self.max_batch_size:
                self._event.set()
        return await future

    async def _batch_loop(self):
        while self._running or self._queue:
            try:
                await asyncio.wait_for(self._wait_for_item(), timeout=self.batch_window_s)
            except asyncio.TimeoutError:
                pass
            async with self._lock:
                if not self._queue: continue
                self._queue.sort(key=lambda r: (-r.priority, r.enqueued_at))
                batch = self._queue[:self.max_batch_size]
                self._queue = self._queue[self.max_batch_size:]
                self._event.clear()
            if batch:
                asyncio.create_task(self._process_batch(batch))

    async def _wait_for_item(self):
        while not self._queue:
            self._event.clear()
            await self._event.wait()

    async def _process_batch(self, batch: list):
        start = time.time()
        self._total_batches += 1
        self._total_batched += len(batch)
        try:
            results = await asyncio.wait_for(self.handler(batch), timeout=BATCH_TIMEOUT_S)
            latency_ms = (time.time() - start) * 1000
            for i, req in enumerate(batch):
                if not req.future.done():
                    val = results[i] if i < len(results) else None
                    req.future.set_result(BatchResult(req.request_id, val,
                        latency_ms=latency_ms / len(batch), batch_size=len(batch), was_batched=len(batch) > 1))
        except Exception as e:
            logger.error(f"[BatchQueue:{self.model_id}] Batch FAILED: {e}")
            for req in batch:
                if not req.future.done(): req.future.set_exception(e)

    def get_stats(self) -> dict:
        avg = self._total_batched / max(1, self._total_batches)
        return {"model_id": self.model_id, "total_requests": self._total_requests,
                "avg_batch_size": round(avg, 2), "queue_depth": len(self._queue)}

class RequestBatcherManager:
    def __init__(self):
        self._queues: dict[str, BatchQueue] = {}

    def register(self, model_id: str, handler: Callable, max_batch_size: int = MAX_BATCH_SIZE, batch_window_ms: int = BATCH_WINDOW_MS) -> BatchQueue:
        if model_id not in self._queues:
            self._queues[model_id] = BatchQueue(model_id, handler, max_batch_size, batch_window_ms)
        return self._queues[model_id]

    async def start_all(self):
        await asyncio.gather(*[q.start() for q in self._queues.values()])

    async def stop_all(self):
        await asyncio.gather(*[q.stop() for q in self._queues.values()])

    async def submit(self, model_id: str, payload: dict, priority: int = 0) -> BatchResult:
        q = self._queues.get(model_id)
        if not q: raise ValueError(f"No queue for: {model_id}")
        return await q.enqueue(payload, priority=priority)

    def get_all_stats(self) -> list:
        return [q.get_stats() for q in self._queues.values()]

async def openai_batch_handler(batch: list) -> list:
    import openai, os
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    async def call(req):
        r = await client.chat.completions.create(
            model=req.payload.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": req.payload.get("message", "")}],
            max_tokens=req.payload.get("max_tokens", 1024),
        )
        return r.choices[0].message.content or ""
    return list(await asyncio.gather(*[call(r) for r in batch], return_exceptions=True))

_manager: Optional[RequestBatcherManager] = None

async def get_batcher_manager() -> RequestBatcherManager:
    global _manager
    if _manager is None:
        _manager = RequestBatcherManager()
        _manager.register("gpt-4o-mini", openai_batch_handler, 10, 50)
        _manager.register("gpt-4o",      openai_batch_handler, 5,  30)
        await _manager.start_all()
    return _manager
