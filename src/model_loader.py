import pickle
import faiss
from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load saved models
with open("models/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("models/gmm.pkl", "rb") as f:
    gmm = pickle.load(f)

with open("models/docs.pkl", "rb") as f:
    clean_docs = pickle.load(f)

# load FAISS index
index = faiss.read_index("models/faiss.index")


def search(query):

    # embed query
    q_emb = model.encode([query])

    # PCA transform
    q_emb_reduced = pca.transform(q_emb)

    # predict cluster
    cluster = gmm.predict(q_emb_reduced)[0]

    # vector search
    D, I = index.search(q_emb, k=3)

    result = clean_docs[I[0][0]]

    return q_emb_reduced[0], cluster, result