import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
import torch

import pipeline_loggers as logger


CHROMA_DB_BERT_BASE = "./chroma_db_bert/bert"

CHROMA_DB_BERT_CHUNKS = os.path.join(CHROMA_DB_BERT_BASE, "chunks")
CHROMA_DB_BERT_FULL = os.path.join(CHROMA_DB_BERT_BASE, "full")
CHROMA_DB_BERT_SUMMARIES = os.path.join(CHROMA_DB_BERT_BASE, "summaries")

# Ensure folders exist
os.makedirs(CHROMA_DB_BERT_CHUNKS, exist_ok=True)
os.makedirs(CHROMA_DB_BERT_FULL, exist_ok=True)
os.makedirs(CHROMA_DB_BERT_SUMMARIES, exist_ok=True)


class DocumentRetrieverBERT:
    """
    BERT-based retriever (MPNet) using ChromaDB with separated folder structure:
        ./chroma_db_bert/bert/chunks
        ./chroma_db_bert/bert/full
        ./chroma_db_bert/bert/summaries
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        logger.info(f"Initializing ChromaDB BERT (MPNet) — model: {model_name}")

        # Embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # Collections
        self.chunk_collection = None
        self.full_doc_collection = None
        self.summary_collection = None

        self.chunk_collection_base_name = "rag_bert_chunks"
        self.full_doc_collection_base_name = "rag_bert_full_documents"
        self.summary_collection_base_name = "rag_bert_summaries"

        logger.info("BERT retriever initialized with folder-segmented ChromaDB storage.")

   
    def _get_chroma_path(self, collection_type: str) -> str:
        if collection_type == "chunk":
            return CHROMA_DB_BERT_CHUNKS
        if collection_type == "full_doc":
            return CHROMA_DB_BERT_FULL
        if collection_type == "summary":
            return CHROMA_DB_BERT_SUMMARIES
        raise ValueError(f"Invalid BERT collection type: {collection_type}")

 
    def _get_or_create_collection(self, base_name: str, identifier: str, collection_type: str):
        path = self._get_chroma_path(collection_type)
        collection_name = f"{base_name}_{identifier}"

        logger.debug(f"Using BERT ChromaDB directory: {path}")
        logger.debug(f"Creating/connecting to collection: {collection_name}")

        chroma_client = chromadb.PersistentClient(path=path)

        return chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

   
    def create_chunk_embeddings(self, chunked_docs, doc_mapping, chunk_metadata, cache_identifier: str):
        self.chunk_collection = self._get_or_create_collection(
            self.chunk_collection_base_name, cache_identifier, "chunk"
        )

        if self.chunk_collection.count() == len(chunked_docs):
            logger.debug(f"Chunk collection already contains {len(chunked_docs)} documents — skipping embedding.")
            return

        logger.debug(f"Adding {len(chunked_docs)} chunks to BERT ChromaDB...")

        ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
        self.chunk_collection.add(
            documents=chunked_docs,
            metadatas=chunk_metadata,
            ids=ids
        )

        logger.debug(f"Added {len(chunked_docs)} BERT chunks to ChromaDB.")

    def create_full_document_embeddings(self, documents, metadatas, cache_identifier: str):
        self.full_doc_collection = self._get_or_create_collection(
            self.full_doc_collection_base_name, cache_identifier, "full_doc"
        )

        if self.full_doc_collection.count() == len(documents):
            logger.debug(f"Full-doc collection already contains {len(documents)} docs — skipping embedding.")
            return

        logger.debug(f"Adding {len(documents)} full documents to BERT ChromaDB...")

        ids = [f"doc_{i}" for i in range(len(documents))]
        self.full_doc_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.debug(f"Added {len(documents)} BERT full documents to ChromaDB.")

    def create_summary_embeddings(self, summaries, cache_identifier: str):
        self.summary_collection = self._get_or_create_collection(
            self.summary_collection_base_name, cache_identifier, "summary"
        )

        summary_texts = [s["summary"] for s in summaries]

        if self.summary_collection.count() == len(summaries):
            logger.debug(f"Summary collection already contains {len(summaries)} items — skipping embedding.")
            return

        metadatas = []
        for i, s in enumerate(summaries):
            md = s.get("metadata", {}).copy()
            md.setdefault("original_doc_idx", i)
            metadatas.append(md)

        ids = [f"summary_{i}" for i in range(len(summaries))]

        self.summary_collection.add(
            documents=summary_texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.debug(f"Added {len(summaries)} BERT summaries to ChromaDB.")

   
    def retrieve_relevant_docs(self, query: str, k: int = 5, collection_type: str = "chunk", where_clause=None):

        if collection_type == "chunk":
            collection = self.chunk_collection
        elif collection_type == "full_doc":
            collection = self.full_doc_collection
        elif collection_type == "summary":
            collection = self.summary_collection
        else:
            logger.info(f"Invalid collection type for BERT retriever: {collection_type}")
            return [], []

        if collection is None:
            logger.info(f"No BERT collection initialized for type: {collection_type}")
            return [], []

        logger.debug(f"Querying BERT collection '{collection_type}', top-k={k}")

        try:
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_items = []
            retrieved_scores = []

            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    content = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    score = results["distances"][0][i]

                    retrieved_items.append({
                        "content": content,
                        "metadata": meta,
                        "score": score,
                    })
                    retrieved_scores.append(score)

            return retrieved_items, retrieved_scores

        except Exception as e:
            logger.info(f"BERT retrieval error: {e}")
            return [], []
