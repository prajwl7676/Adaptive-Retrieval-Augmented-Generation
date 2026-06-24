import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional

import pipeline_loggers as logger

# Base TF-IDF directory (lives inside the BERT directory)
TF_IDF_BASE_DIR = "./chroma_db_bert/tf_idf"

# Subdirectories
TF_IDF_CHUNK_DIR = os.path.join(TF_IDF_BASE_DIR, "chunks")
TF_IDF_FULL_DIR = os.path.join(TF_IDF_BASE_DIR, "full")
TF_IDF_SUMMARY_DIR = os.path.join(TF_IDF_BASE_DIR, "summaries")

# Ensure all directories exist
os.makedirs(TF_IDF_CHUNK_DIR, exist_ok=True)
os.makedirs(TF_IDF_FULL_DIR, exist_ok=True)
os.makedirs(TF_IDF_SUMMARY_DIR, exist_ok=True)


def _tfidf_cache_path(base_name: str, identifier: str, collection_type: str):
    if collection_type == "chunk":
        folder = TF_IDF_CHUNK_DIR
    elif collection_type == "full_doc":
        folder = TF_IDF_FULL_DIR
    elif collection_type == "summary":
        folder = TF_IDF_SUMMARY_DIR
    else:
        raise ValueError(f"Invalid TF-IDF collection type: {collection_type}")

    return os.path.join(folder, f"{base_name}_{identifier}.pkl")




class DocumentRetrieverTFIDF:
   

    def __init__(self):
        logger.info("Initializing TF-IDF Vectorizers...")

        # Create vectorizers but content is loaded per-cache
        self.chunk_vectorizer = TfidfVectorizer()
        self.full_doc_vectorizer = TfidfVectorizer()
        self.summary_vectorizer = TfidfVectorizer()

        # Cached data holders
        self.chunk_tfidf_matrix = None
        self.chunked_docs = []
        self.chunk_metadata = []

        self.full_doc_tfidf_matrix = None
        self.full_documents = []
        self.full_document_metadata = []

        self.summary_tfidf_matrix = None
        self.summaries_text = []
        self.summaries_full_data = []

        # Base names to match BERT naming style
        self.chunk_cache_base = "rag_tfidf_chunks"
        self.full_cache_base = "rag_tfidf_full_documents"
        self.summary_cache_base = "rag_tfidf_summaries"

        logger.debug("TF-IDF Vectorizers initialized successfully.")


   
    def _load_cache(self, base_name: str, identifier: str, collection_type: str):
        path = _tfidf_cache_path(base_name, identifier, collection_type)
        if os.path.exists(path):
            logger.debug(f"Loading TF-IDF cache from: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    
    
    def _save_cache(self, base_name: str, identifier: str, collection_type: str, data: dict):
        path = _tfidf_cache_path(base_name, identifier, collection_type)
        logger.debug(f"Saving TF-IDF cache to: {path}")
        with open(path, "wb") as f:
            pickle.dump(data, f)



    def create_chunk_embeddings(self, chunked_docs, doc_mapping, chunk_metadata, cache_identifier: str):

        #cache = self._load_cache(self.chunk_cache_base, cache_identifier)
        cache = self._load_cache(self.chunk_cache_base, cache_identifier, "chunk")

        # Cache hit if same doc count
        if cache and cache["count"] == len(chunked_docs):
            logger.debug("TF-IDF chunk cache hit — loading from disk, skipping recomputation.")
            self.chunk_vectorizer = cache["vectorizer"]
            self.chunk_tfidf_matrix = cache["matrix"]
            self.chunked_docs = cache["docs"]
            self.chunk_metadata = cache["metadata"]
            return

        logger.info(f"Creating TF-IDF matrix for {len(chunked_docs)} chunks...")

        self.chunked_docs = chunked_docs
        self.chunk_metadata = chunk_metadata

        self.chunk_tfidf_matrix = self.chunk_vectorizer.fit_transform(chunked_docs)

        # Save cache
        self._save_cache(self.chunk_cache_base, cache_identifier, "chunk", {
            "vectorizer": self.chunk_vectorizer,
            "matrix": self.chunk_tfidf_matrix,
            "docs": self.chunked_docs,
            "metadata": self.chunk_metadata,
            "count": len(chunked_docs)
        })

        logger.debug(f"TF-IDF chunk matrix shape: {self.chunk_tfidf_matrix.shape}")


   
    def create_full_document_embeddings(self, documents, document_metadata, cache_identifier: str):

        cache = self._load_cache(self.full_cache_base, cache_identifier, "full_doc")

        if cache and cache["count"] == len(documents):
            logger.debug("TF-IDF full-doc cache hit — loading from disk.")
            self.full_doc_vectorizer = cache["vectorizer"]
            self.full_doc_tfidf_matrix = cache["matrix"]
            self.full_documents = cache["docs"]
            self.full_document_metadata = cache["metadata"]
            return

        logger.info(f"Creating TF-IDF matrix for {len(documents)} full documents...")

        self.full_documents = documents
        self.full_document_metadata = document_metadata

        self.full_doc_tfidf_matrix = self.full_doc_vectorizer.fit_transform(documents)

        self._save_cache(self.full_cache_base, cache_identifier, "full_doc", {
            "vectorizer": self.full_doc_vectorizer,
            "matrix": self.full_doc_tfidf_matrix,
            "docs": self.full_documents,
            "metadata": self.full_document_metadata,
            "count": len(documents)
        })

        logger.debug(f"TF-IDF full-doc matrix shape: {self.full_doc_tfidf_matrix.shape}")


 
    def create_summary_embeddings(self, summaries, cache_identifier: str):

        cache = self._load_cache(self.summary_cache_base, cache_identifier, "summary")

        summary_texts = [s["summary"] for s in summaries]

        if cache and cache["count"] == len(summaries):
            logger.debug("TF-IDF summary cache hit — loading from disk.")
            self.summary_vectorizer = cache["vectorizer"]
            self.summary_tfidf_matrix = cache["matrix"]
            self.summaries_text = cache["texts"]
            self.summaries_full_data = cache["metadata"]
            return

        logger.info(f"Creating TF-IDF matrix for {len(summary_texts)} summaries...")

        self.summaries_text = summary_texts
        self.summaries_full_data = summaries

        self.summary_tfidf_matrix = self.summary_vectorizer.fit_transform(summary_texts)

        self._save_cache(self.summary_cache_base, cache_identifier, "summary", {
            "vectorizer": self.summary_vectorizer,
            "matrix": self.summary_tfidf_matrix,
            "texts": self.summaries_text,
            "metadata": self.summaries_full_data,
            "count": len(summaries)
        })

        logger.debug(f"TF-IDF summary matrix shape: {self.summary_tfidf_matrix.shape}")


    
    def retrieve_relevant_docs(self, query, k=5, collection_type="chunk"):
        vectorizer = None
        tfidf_matrix = None
        items_content = []
        items_metadata = []

        if collection_type == "chunk":
            vectorizer = self.chunk_vectorizer
            tfidf_matrix = self.chunk_tfidf_matrix
            items_content = self.chunked_docs
            items_metadata = self.chunk_metadata

        elif collection_type == "full_doc":
            vectorizer = self.full_doc_vectorizer
            tfidf_matrix = self.full_doc_tfidf_matrix
            items_content = self.full_documents
            items_metadata = self.full_document_metadata

        elif collection_type == "summary":
            vectorizer = self.summary_vectorizer
            tfidf_matrix = self.summary_tfidf_matrix
            items_content = self.summaries_text
            items_metadata = self.summaries_full_data

        else:
            logger.info(f"Invalid collection_type: {collection_type}")
            return [], []

        if tfidf_matrix is None or not items_content:
            logger.info(f"TF-IDF matrix for '{collection_type}' not initialized. Skipping retrieval.")
            return [], []

        logger.info(f"Retrieving relevant {collection_type}s from TF-IDF (k={k})")
        logger.debug(f"TF-IDF retrieval query='{query}' on collection='{collection_type}' (k={k}).")

        query_vector = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        top_indices = np.argsort(similarity_scores)[::-1][:k]

        retrieved_items = []
        retrieved_scores = []

        for rank, idx in enumerate(top_indices):
            score = float(similarity_scores[idx])

            if score <= 0:
                continue

            if collection_type == "summary":
                meta = items_metadata[idx].get("metadata", {})
                content = items_metadata[idx].get("summary", items_content[idx])
            else:
                meta = items_metadata[idx]
                content = items_content[idx]

            retrieved_items.append({
                "content": content,
                "metadata": meta,
                "score": score
            })
            retrieved_scores.append(score)

            # DEBUG detailed preview
            preview = content[:120].replace("\n", " ")
            title = meta.get("title", f"{collection_type} {rank}")
            source = meta.get("source", "N/A")

            logger.debug(
                f"[TF-IDF {collection_type}] Rank {rank+1} — Score={score:.4f} — "
                f"Title='{title}' Source='{os.path.basename(source)}' Preview='{preview}...'"
            )

        return retrieved_items, retrieved_scores
