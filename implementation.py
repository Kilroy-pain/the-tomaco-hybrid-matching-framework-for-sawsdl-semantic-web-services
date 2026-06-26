import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

class TomacoMatchingFramework:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def syntactic_similarity(self, text1, text2):
        """Compute syntactic similarity using TF-IDF and cosine similarity."""
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    def semantic_similarity(self, embedding1, embedding2):
        """Compute semantic similarity using cosine similarity on embeddings."""
        embedding1_tensor = torch.tensor(embedding1, dtype=torch.float32)
        embedding2_tensor = torch.tensor(embedding2, dtype=torch.float32)
        similarity = cosine_similarity(embedding1_tensor.unsqueeze(0), embedding2_tensor.unsqueeze(0))
        return similarity.item()

    def hybrid_similarity(self, text1, text2, embedding1, embedding2, alpha=0.5):
        """Combine syntactic and semantic similarity using a weighted average."""
        syntactic_sim = self.syntactic_similarity(text1, text2)
        semantic_sim = self.semantic_similarity(embedding1, embedding2)
        hybrid_sim = alpha * syntactic_sim + (1 - alpha) * semantic_sim
        return hybrid_sim

if __name__ == '__main__':
    # Dummy data for testing
    text1 = "Find weather forecast services"
    text2 = "Search for services providing weather predictions"
    embedding1 = np.random.rand(300)  # Simulating a 300-dimensional embedding
    embedding2 = np.random.rand(300)  # Simulating a 300-dimensional embedding

    # Instantiate the framework
    tomaco = TomacoMatchingFramework()

    # Compute syntactic similarity
    syntactic_sim = tomaco.syntactic_similarity(text1, text2)
    print(f"Syntactic Similarity: {syntactic_sim}")

    # Compute semantic similarity
    semantic_sim = tomaco.semantic_similarity(embedding1, embedding2)
    print(f"Semantic Similarity: {semantic_sim}")

    # Compute hybrid similarity
    hybrid_sim = tomaco.hybrid_similarity(text1, text2, embedding1, embedding2, alpha=0.7)
    print(f"Hybrid Similarity: {hybrid_sim}")