# Semantic Cache System for 20 Newsgroups

This project implements a lightweight semantic search system using vector embeddings, fuzzy clustering, and a custom semantic cache.

The system avoids redundant computation by recognizing semantically similar queries, even when they are phrased differently. It combines modern NLP embeddings with a vector database and an API service for real-time querying.

## Live Demo
https://semantic-cache-newsgroups.streamlit.app/

## Project Overview

The goal of this system is to build a semantic search pipeline that:

- Embeds documents and queries using a transformer model
- Stores embeddings in a vector database
- Discovers latent semantic structure using fuzzy clustering
- Uses a semantic cache to avoid recomputing similar queries
- Exposes the system through a FastAPI service

This demonstrates how machine learning systems can improve efficiency in real-world search applications.

## System Architecture

User Query  
↓  
SentenceTransformer Embedding  
↓  
PCA Dimensionality Reduction  
↓  
Gaussian Mixture Model (Fuzzy Clustering)  
↓  
Semantic Cache Lookup

Cache Hit → Return Cached Result  
Cache Miss → FAISS Vector Search → Store Result in Cache

## Technologies Used

- Python
- SentenceTransformers
- FAISS (Vector Database)
- Scikit-learn
- FastAPI
- NumPy

## Dataset

This project uses the **20 Newsgroups dataset** from the UCI Machine Learning Repository.

The dataset contains approximately **20,000 news articles across 20 topic categories**.  
Preprocessing removes noisy elements such as email headers, metadata, and quoted replies to ensure embeddings capture meaningful semantic content.

---

## Key Components

### Embeddings

We use the `all-MiniLM-L6-v2` SentenceTransformer model to generate dense semantic embeddings for documents and queries.

### Vector Database

FAISS is used to perform fast similarity search across document embeddings.

### Fuzzy Clustering

A **Gaussian Mixture Model (GMM)** is used instead of hard clustering.  
This allows documents to belong to multiple topics with different probabilities, reflecting the real semantic structure of the dataset.

### Semantic Cache

The semantic cache stores previously processed queries and their results.

Instead of exact matching, cosine similarity between query embeddings determines whether a new query is similar enough to reuse a cached result.

The cache is also **cluster-aware**, meaning queries are stored within clusters to improve lookup efficiency.

### FastAPI Service

The system exposes a REST API that allows queries to be processed and cache statistics to be inspected.

FastAPI automatically generates interactive documentation using Swagger UI.

---

## Running the API

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

```bash
uvicorn src.api:app --reload
```

### Open API documentation

http://127.0.0.1:8000/docs

This will open the **Swagger UI interface** where the API can be tested interactively.

## API Endpoints

### POST `/query`

Processes a natural language query and returns the most relevant document.

Example request:

{
"query": "space shuttle launch technology"
}

Response includes:

- cache_hit
- similarity_score
- dominant_cluster
- latency_ms

### GET `/cache/stats`

Returns statistics about the semantic cache.

Example output:

{
"total_entries": 3,
"hit_count": 2,
"miss_count": 1,
"hit_rate": 0.66
}

### DELETE `/cache`

Clears the cache and resets statistics.

## Performance Demonstration

Cache hits significantly reduce query latency because the system avoids recomputing vector search operations.

Example:

Query Type - Latency

Cache Miss ~150 ms
Cache Hit ~5 ms

## NOTEBOOK LINK
https://colab.research.google.com/drive/1xLTXDX-W1QcEZeYObbjScRxwdc8XOZV-?usp=sharing

## Author

Neha Khadeeja
