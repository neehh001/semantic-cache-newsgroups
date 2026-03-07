from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):
        self.cache = {}
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, embedding, cluster):

        if cluster not in self.cache:
            return None

        best_match = None
        best_score = 0

        for entry in self.cache[cluster]:

            sim = cosine_similarity(
                [embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > best_score:
                best_score = sim
                best_match = entry

        if best_score >= self.threshold:
            self.hit_count += 1
            return best_match, best_score

        return None

    def insert(self, query, embedding, result, cluster):

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

        self.miss_count += 1

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = self.hit_count / total if total else 0

        return {
            "total_entries": sum(len(v) for v in self.cache.values()),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0