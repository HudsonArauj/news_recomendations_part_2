# preprocess.py

import numpy as np
from scripts.vectorizer import get_embedding

def search(query, index, embeddings, similarity_threshold, k=40):
    query_embedding = get_embedding(query).astype('float32')
    query_embedding = np.squeeze(query_embedding)

    D, I = index.search(query_embedding.reshape(1, -1), k)

    similarity_scores = 1 / (1 + D[0]) 
    similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
    
    
    similar_texts = [
        embeddings[i] for score, i in zip(similarity_scores, I[0]) if score >= similarity_threshold
    ]
    # Remover o primeiro elemento e o último elemento da lista de similar_texts e similarity_scores pq é sempre 1 e 0
    similar_texts = similar_texts[1:-1]
    similarity_scores = similarity_scores[1:-1]

    return similar_texts, similarity_scores
