from langchain_community.vectorstores import Chroma
from datetime import datetime
import time
from pdf_to_vector_store import TurkishBERTEmbeddings
import numpy as np

class VectorStore:
    def __init__(self, persist_directory="vector_store", similarity_threshold=0.3):
        """Initialize vector store with a lower default threshold for better recall"""
        self.embeddings = TurkishBERTEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.similarity_threshold = similarity_threshold
        self.doc_count = self.vectorstore._collection.count()

    def search(self, query, k=5, search_type="hybrid"):
        """Enhanced search function with hybrid approach and adaptive k"""
        try:
            # Adjust k based on available documents
            k = min(k, self.doc_count)
            
            if search_type == "hybrid":
                # Get more candidates than needed for better results
                similarity_results = self.similarity_search(query, k=k*2)
                mmr_results = self.mmr_search(query, k=k)
                
                # Combine and deduplicate results
                all_results = similarity_results + mmr_results
                seen_contents = set()
                unique_results = []
                
                for doc, score in all_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        unique_results.append((doc, score))
                
                # Sort by score and take top k
                results = sorted(unique_results, key=lambda x: x[1], reverse=True)[:k]
            else:
                results = self.similarity_search(query, k=k)

            # Apply threshold but always return at least one result
            filtered_results = [(doc, score) for doc, score in results 
                              if (score + 1) / 2 >= self.similarity_threshold]
            
            if not filtered_results and results:
                # If no results meet threshold, return the best match
                filtered_results = [results[0]]
                
            return filtered_results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def similarity_search(self, query, k=5):
        """Similarity search with proper cosine similarity"""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search using Chroma's raw score
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Convert scores to cosine similarity (0-1 range)
            normalized_results = []
            for doc, score in results:
                # Chroma returns L2 distance, convert to cosine similarity
                cosine_sim = 1 / (1 + score)  # Convert L2 to cosine similarity
                normalized_results.append((doc, cosine_sim))
                
            return normalized_results
        
        except Exception as e:
            print(f"Similarity search error: {str(e)}")
            return []

    def mmr_search(self, query, k=5, lambda_mult=0.5):
        """MMR search with normalized scoring"""
        try:
            docs = self.vectorstore.max_marginal_relevance_search(query, k=k, lambda_mult=lambda_mult)
            
            # Calculate relevance scores
            query_embedding = self.embeddings.embed_query(query)
            scores = []
            
            for doc in docs:
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                # Normalize vectors before calculating dot product
                query_norm = np.linalg.norm(query_embedding)
                doc_norm = np.linalg.norm(doc_embedding)
                
                if query_norm > 0 and doc_norm > 0:
                    # Calculate cosine similarity (will be between -1 and 1)
                    score = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                    # Normalize to 0-1 range
                    score = (score + 1) / 2
                else:
                    score = 0
                    
                scores.append(score)
            
            return list(zip(docs, scores))
        except Exception as e:
            print(f"MMR search error: {str(e)}")
            return []