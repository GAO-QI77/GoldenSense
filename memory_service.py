from contextlib import asynccontextmanager
import os
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI
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
    global retriever
    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/postgres")
    try:
        retriever = MemoryRetriever(database_url=db_url)
    except Exception as e:
        print(f"Failed to initialize MemoryRetriever: {e}")
        retriever = None
    yield


app = FastAPI(title="GoldenSense Memory API", version="1.0.0", lifespan=lifespan)

@app.post("/api/v1/memory/search", response_model=SearchResponse)
async def search_memory(req: SearchRequest):
    if retriever is None:
        return _status_response(
            req,
            storage="unavailable",
            status="unavailable",
            degraded_reason="memory_retriever_not_initialized",
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
    return {"status": "ok", "service": "memory_search"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory_service:app", host="0.0.0.0", port=8012, reload=False)
