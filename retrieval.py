import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRetriever:
    def __init__(self, catalog_path="catalog.json"):
        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)
        
        # Combine text fields for matching
        self.documents = [
            f"{item['name']} {item.get('description', '')} {item.get('test_type', '')}"
            for item in self.catalog
        ]
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matching indices
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.0:  # Only return if there's some match
                results.append(self.catalog[idx])
                
        # Fallback if no keywords matched: return top default items
        if not results:
            results = self.catalog[:top_k]
            
        return results