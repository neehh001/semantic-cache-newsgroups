import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.semantic_cache import SemanticCache
from src.model_loader import search

app = FastAPI()

cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_api(request: QueryRequest):

    start_time = time.time()

    embedding, cluster, result = search(request.query)

    cached = cache.lookup(embedding, cluster)

    if cached:

        entry, score = cached

        return {
            "query": request.query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"],
            "dominant_cluster": int(cluster),
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

    cache.insert(request.query, embedding, result, cluster)

    return {
        "query": request.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": int(cluster),
        "latency_ms": round((time.time() - start_time) * 1000, 2)
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"status": "cache cleared"}