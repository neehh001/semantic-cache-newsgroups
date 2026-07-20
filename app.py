"""
Semantic Search with Cluster-Aware Semantic Caching — Interactive Demo
Run locally with:  streamlit run app.py

Expects these 4 artifacts (saved from the training notebook) in the same folder:
  pca.pkl, gmm.pkl, docs.pkl, faiss.index
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Semantic Search + Cluster-Aware Caching", layout="wide")

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "models") if os.path.isdir(os.path.join(APP_DIR, "models")) else APP_DIR

# ---------------------------------------------------------------------------
# Load model + saved artifacts once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(os.path.join(MODELS_DIR, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "gmm.pkl"), "rb") as f:
        gmm = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(os.path.join(MODELS_DIR, "faiss.index"))

    return model, pca, gmm, docs, index

missing = [f for f in ["pca.pkl", "gmm.pkl", "docs.pkl", "faiss.index"]
           if not os.path.exists(os.path.join(MODELS_DIR, f))]

if missing:
    st.error(
        "Missing artifact file(s) in the models/ folder: " + ", ".join(missing) +
        ".\n\nThese are the files your training notebook saves and downloads at the end "
        "(pca.pkl, gmm.pkl, docs.pkl, faiss.index). Put them in a 'models' subfolder "
        "next to app.py (or directly next to app.py) and restart the app."
    )
    st.stop()

model, pca, gmm, docs, index = load_artifacts()

# ---------------------------------------------------------------------------
# Semantic cache (same logic as the notebook)
# ---------------------------------------------------------------------------
class SemanticCache:
    def __init__(self, threshold=0.85):
        self.cache = {}
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, embedding, cluster):
        if cluster not in self.cache:
            return None
        best_match, best_score = None, 0
        for entry in self.cache[cluster]:
            sim = cosine_similarity([embedding], [entry["embedding"]])[0][0]
            if sim > best_score:
                best_score, best_match = sim, entry
        if best_score >= self.threshold:
            self.hit_count += 1
            return best_match, best_score
        return None

    def insert(self, query, embedding, result, cluster):
        self.cache.setdefault(cluster, []).append(
            {"query": query, "embedding": embedding, "result": result}
        )
        self.miss_count += 1

    def stats(self):
        total = self.hit_count + self.miss_count
        return {
            "total_entries": sum(len(v) for v in self.cache.values()),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0,
        }

if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache(threshold=0.85)
if "log" not in st.session_state:
    st.session_state.log = []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Semantic Search with Cluster-Aware Semantic Caching")
st.caption(
    "SentenceTransformer + FAISS similarity search over 20 Newsgroups, with a "
    "cluster-aware cache (GMM clustering) that skips a full FAISS search on repeat/similar queries."
)

with st.sidebar:
    st.header("Cache settings")
    threshold = st.slider("Cache similarity threshold", 0.50, 0.99,
                           st.session_state.cache.threshold, 0.01)
    if threshold != st.session_state.cache.threshold:
        st.session_state.cache.threshold = threshold

    if st.button("Reset cache"):
        st.session_state.cache = SemanticCache(threshold=threshold)
        st.session_state.log = []
        st.rerun()

    stats = st.session_state.cache.stats()
    st.metric("Cache entries", stats["total_entries"])
    st.metric("Hit rate", f"{stats['hit_rate']:.1%}")
    st.metric("Hits / Misses", f"{stats['hit_count']} / {stats['miss_count']}")

query = st.text_input("Enter a search query", "space shuttle launch technology")
run = st.button("Search", type="primary")

if run and query.strip():
    t0 = time.perf_counter()

    q_emb = model.encode([query])
    q_emb_reduced = pca.transform(q_emb)
    cluster = int(gmm.predict(q_emb_reduced)[0])

    cached = st.session_state.cache.lookup(q_emb_reduced[0], cluster)

    if cached:
        entry, score = cached
        elapsed_ms = (time.perf_counter() - t0) * 1000
        st.success(f"Cache HIT — similarity {score:.3f} — cluster {cluster} — {elapsed_ms:.1f} ms")
        st.write(f"**Matched cached query:** {entry['query']}")
        st.write(entry["result"][:500] + "...")
    else:
        D, I = index.search(q_emb, k=3)
        result = docs[I[0][0]]
        st.session_state.cache.insert(query, q_emb_reduced[0], result, cluster)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        st.warning(f"Cache MISS (ran full FAISS search) — cluster {cluster} — {elapsed_ms:.1f} ms")
        st.write("**Top match:**")
        st.write(result[:500] + "...")

    st.session_state.log.append({
        "query": query, "cluster": cluster,
        "result": "HIT" if cached else "MISS", "latency_ms": round(elapsed_ms, 2),
    })

if st.session_state.log:
    st.subheader("Query log (this session)")
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True)

    hit_lat = log_df.loc[log_df["result"] == "HIT", "latency_ms"]
    miss_lat = log_df.loc[log_df["result"] == "MISS", "latency_ms"]
    c1, c2 = st.columns(2)
    if len(hit_lat):
        c1.metric("Avg cache-hit latency", f"{hit_lat.mean():.1f} ms")
    if len(miss_lat):
        c2.metric("Avg cache-miss latency", f"{miss_lat.mean():.1f} ms")

st.divider()
st.caption(
    "Try: search 'space shuttle launch technology', then 'NASA shuttle launch system' — "
    "semantically close enough to hit the cache instead of re-running FAISS."
)
