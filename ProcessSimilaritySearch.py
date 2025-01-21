import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilaritySearch:
    def __init__(self, query):
        self.query = query

    def search_cases(self):
        # Load the general vectorizer and vectors
        with open('general_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('general_vectors.pkl', 'rb') as f:
            vectors = pickle.load(f)

        # Vectorize the query
        query_vec = vectorizer.transform([self.query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, vectors).flatten()

        # Get the top 3 matches
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_similarities = similarities[top_indices]

        # Return the top indices and their scores
        results = [(index, score) for index, score in zip(top_indices, top_similarities)]
        return results

    def search_case_number(self):
        # Load the case number vectorizer and vectors
        with open('case_number_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('case_number_vectors.pkl', 'rb') as f:
            vectors = pickle.load(f)

        # Vectorize the query
        query_vec = vectorizer.transform([self.query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, vectors).flatten()
        
         # Check for exact match
        exact_match_index = np.where(similarities == 1.0)[0]
        if exact_match_index.size > 0:
            return [(exact_match_index[0], 1.0)]
        
        # Get the top 3 matches
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_similarities = similarities[top_indices]

        # Return the top indices and their scores
        cases_results = [(index, score) for index, score in zip(top_indices, top_similarities)]
        return cases_results
    
    def search_kcs(self):
        # Load the general vectorizer and vectors
        with open('kcs_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('kcs_vectors.pkl', 'rb') as f:
            vectors = pickle.load(f)

        # Vectorize the query
        query_vec = vectorizer.transform([self.query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, vectors).flatten()

        print("KCS Similarities --> ", similarities)

        # Get the top 3 matches
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_similarities = similarities[top_indices]

        print("Top indices:", top_indices)
        print("Top similarities:", top_similarities)

        # Return the top indices and their scores
        kcs_results = [(index, score) for index, score in zip(top_indices, top_similarities)]

        return kcs_results