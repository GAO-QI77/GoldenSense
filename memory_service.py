from contextlib import asynccontextmanager
import asyncio
import os
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from memory_retriever import MemoryRetriever

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_event_text: str = Field(..., description="当前突发新闻的文本")
    top_k: int = Field(default=3, description="返回最相似的历史事件数量", ge=1, le=10)

class SearchResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_date: str
    headline: str
    context_summary: str
    gold_t1_return: float
    gold_t7_return: float
    similarity: float

class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int
    storage: str
    status: Literal["ok", "degraded", "unavailable"]
    degraded_reason: Optional[str] = None
    source_freshness_seconds: Optional[int] = None
    timing_ms: Dict[str, int]
    results: List[SearchResultItem]

retriever = None
retriever_status: Literal["not_started", "loading", "ready", "unavailable"] = "not_started"
retriever_error: Optional[str] = None


def _status_response(
    req: SearchRequest,
    *,
    storage: str,
    status: Literal["ok", "degraded", "unavailable"],
    degraded_reason: Optional[str],
) -> SearchResponse:
    return SearchResponse(
        query=req.current_event_text,
        top_k=req.top_k,
        storage=storage,
        status=status,
        degraded_reason=degraded_reason,
        source_freshness_seconds=None,
        timing_ms={"embed": 0, "connect_and_detect": 0, "db_query": 0, "total": 0},
        results=[],
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, retriever_status, retriever_error
    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/postgres")
    model_id = os.environ.get("MEMORY_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    start_background_load = os.environ.get("MEMORY_START_BACKGROUND_LOAD", "0") == "1"
    retriever = None
    retriever_status = "not_started"
    retriever_error = None
    app.state.database_url = db_url
    app.state.model_id = model_id
    task = None
    if start_background_load:
        task = asyncio.create_task(_load_retriever(db_url, model_id))
    yield
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _load_retriever(db_url: str, model_id: str) -> None:
    global retriever, retriever_status, retriever_error
    retriever_status = "loading"
    try:
        retriever = await asyncio.to_thread(MemoryRetriever, database_url=db_url, model_id=model_id)
        retriever_status = "ready"
        retriever_error = None
    except Exception as exc:
        retriever = None
        retriever_status = "unavailable"
        retriever_error = f"{type(exc).__name__}:{exc}"
        print(f"Failed to initialize MemoryRetriever: {retriever_error}")


app = FastAPI(title="GoldenSense Memory API", version="1.0.0", lifespan=lifespan)

@app.post("/api/v1/memory/search", response_model=SearchResponse)
async def search_memory(req: SearchRequest):
    if retriever is None:
        return _status_response(
            req,
            storage="unavailable",
            status="unavailable",
            degraded_reason=f"memory_retriever_{retriever_status}:{retriever_error or 'not_loaded'}",
        )
    
    try:
        res = retriever.search(req.current_event_text, req.top_k)
        return SearchResponse(**res)
    except Exception as e:
        print(f"Memory search degraded: {e}")
        return _status_response(
            req,
            storage="degraded",
            status="degraded",
            degraded_reason=f"memory_search_failed:{type(e).__name__}:{e}",
        )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "memory_search",
        "retriever_status": retriever_status,
        "retriever_error": retriever_error,
    }


@app.get("/health/live")
def health_live():
    return {"status": "ok", "service": "memory_search"}


@app.get("/health/ready")
def health_ready():
    ready = retriever is not None and retriever_status == "ready"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "unavailable",
            "retriever_status": retriever_status,
            "retriever_error": retriever_error,
            "errors": [] if ready else ["memory_retriever_not_ready"],
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory_service:app", host="0.0.0.0", port=8012, reload=False)
