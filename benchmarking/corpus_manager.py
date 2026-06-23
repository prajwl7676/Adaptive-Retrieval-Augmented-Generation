

import os
import sys
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.hybrid_pipeline import HybridRAGPipeline
from src.adaptive.adpt_rag_pipeline import AdaptiveRAGPipeline


class CorpusManager:
    
    
    def __init__(
        self,
        cache_dir: str = "../corpus_cache",
        cache_enabled: bool = True,
        force_reload: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.force_reload = force_reload
        
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, dataset_name: str, config_hash: str) -> Path:
        """Get path for cached corpus."""
        return self.cache_dir / f"{dataset_name}_{config_hash}.pkl"
    
    def _compute_config_hash(self, config: dict) -> str:
        """Compute hash of configuration for cache key."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _save_to_cache(self, data: dict, cache_path: Path):
        """Save processed corpus to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Corpus cached to: {cache_path}")
        except Exception as e:
            print(f"⚠️  Failed to cache corpus: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[dict]:
        """Load corpus from cache."""
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded corpus from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load from cache: {e}")
            return None
    
    def load_hotpot_corpus(
        self,
        max_samples: Optional[int] = None,
        include_distractors: bool = True,
        split: str = "validation"
    ) -> Tuple[List[Dict], List[Dict]]:
        
        # Create cache key
        config = {
            "dataset": "hotpot_qa",
            "max_samples": max_samples,
            "include_distractors": include_distractors,
            "split": split
        }
        config_hash = self._compute_config_hash(config)
        cache_path = self._get_cache_path("hotpot", config_hash)
        
        # Try to load from cache
        if self.cache_enabled and not self.force_reload:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data["questions"], cached_data["documents"]
        
        # Load dataset
        print(f"\n📥 Loading HotpotQA dataset (split: {split})...")
        if max_samples:
            dataset = load_dataset(
                "hotpot_qa",
                "distractor" if include_distractors else "fullwiki",
                split=f"{split}[:{max_samples}]",
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                "hotpot_qa",
                "distractor" if include_distractors else "fullwiki",
                split=split,
                trust_remote_code=True
            )
        
        print(f"✅ Loaded {len(dataset)} questions")
        
        # Extract questions and documents
        questions = []
        documents = []
        seen_docs = set()  # Track unique documents
        
        print("\n📚 Processing documents...")
        for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing")):
            # Store question info
            question_data = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample["type"],  # bridge, comparison
                "level": sample["level"],  # easy, medium, hard
                "supporting_facts": sample.get("supporting_facts", {})
            }
            questions.append(question_data)
            
            # Extract documents from context
            context_titles = sample["context"]["title"]
            context_sentences_list = sample["context"]["sentences"]
            
            for doc_title, doc_sentences in zip(context_titles, context_sentences_list):
                if not isinstance(doc_sentences, list):
                    continue
                
                # Join sentences to form document
                content = doc_title + "\n\n" + " ".join(doc_sentences)
                
                # Create unique document ID
                doc_id = hashlib.md5(content.encode()).hexdigest()
                
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    doc_entry = {
                        "id": doc_id,
                        "content": content,
                        "metadata": {
                            "source": "hotpotqa",
                            "title": doc_title,
                            "question_id": sample["id"],
                            "length": len(content),
                            "original_doc_idx": len(documents)  # Index in documents list
                        }
                    }
                    documents.append(doc_entry)
        
        print(f"✅ Extracted {len(documents)} unique documents from {len(questions)} questions")
        
        # Cache the results
        if self.cache_enabled:
            cache_data = {
                "questions": questions,
                "documents": documents,
                "config": config
            }
            self._save_to_cache(cache_data, cache_path)
        
        return questions, documents
    
    def initialize_pipeline_with_hotpot(
        self,
        pipeline: RAGPipeline,
        max_samples: Optional[int] = None,
        include_distractors: bool = True
    ):
        
        print("\n🚀 Initializing pipeline with HotpotQA corpus...")
        
        # Load corpus
        questions, documents = self.load_hotpot_corpus(
            max_samples=max_samples,
            include_distractors=include_distractors
        )
        
        # Initialize corpus in pipeline
        print(f"\n📊 Initializing corpus with {len(documents)} documents...")
        pipeline.corpus_initializer.initialize_documents_hotpot(
            pipeline.retriever,
            documents
        )
        
        # If hybrid pipeline, initialize TF-IDF and BERT
        if isinstance(pipeline, (HybridRAGPipeline, AdaptiveRAGPipeline)):
            print("\n🔀 Initializing hybrid retrieval (TF-IDF + BERT)...")
            
            # TF-IDF
            if hasattr(pipeline, 'tfidf_retriever'):
                print("  Initializing TF-IDF...")
                pipeline.tfidf_retriever.create_full_document_embeddings(
                    pipeline.corpus_initializer.documents,
                    pipeline.corpus_initializer.document_metadata,
                    pipeline.corpus_initializer.doc_identifier
                )
                pipeline.tfidf_retriever.create_chunk_embeddings(
                    pipeline.corpus_initializer.chunked_docs,
                    pipeline.corpus_initializer.doc_mapping,
                    pipeline.corpus_initializer.chunk_metadata,
                    pipeline.corpus_initializer.doc_identifier
                )
            
            # BERT
            if hasattr(pipeline, 'bert_retriever'):
                print("  Initializing BERT (MPNet)...")
                pipeline.bert_retriever.create_full_document_embeddings(
                    pipeline.corpus_initializer.documents,
                    pipeline.corpus_initializer.document_metadata,
                    pipeline.corpus_initializer.doc_identifier
                )
                pipeline.bert_retriever.create_chunk_embeddings(
                    pipeline.corpus_initializer.chunked_docs,
                    pipeline.corpus_initializer.doc_mapping,
                    pipeline.corpus_initializer.chunk_metadata,
                    pipeline.corpus_initializer.doc_identifier
                )
        
        pipeline.is_initialized = True
        print("✅ Pipeline initialization complete!")
        
        return questions
    
    def get_corpus_stats(self, documents: List[Dict]) -> dict:
        """Get statistics about the corpus."""
        if not documents:
            return {}
        
        total_chars = sum(len(doc["content"]) for doc in documents)
        total_words = sum(len(doc["content"].split()) for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_doc_length_chars": total_chars / len(documents),
            "avg_doc_length_words": total_words / len(documents),
            "min_doc_length": min(len(doc["content"]) for doc in documents),
            "max_doc_length": max(len(doc["content"]) for doc in documents)
        }


# Example usage
if __name__ == "__main__":
    from benchmark_config import BenchmarkConfig, BenchmarkDataset
    
    # Test corpus manager
    config = BenchmarkConfig(
        dataset=BenchmarkDataset.HOTPOT,
        sample_size=100
    )
    
    manager = CorpusManager(cache_enabled=True)
    
    # Load corpus
    questions, documents = manager.load_hotpot_corpus(
        max_samples=config.sample_size,
        include_distractors=True
    )
    
    # Print stats
    print("\n📊 Corpus Statistics:")
    stats = manager.get_corpus_stats(documents)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test with pipeline
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("\n🧪 Testing pipeline initialization...")
        pipeline = RAGPipeline(api_key=api_key)
        questions = manager.initialize_pipeline_with_hotpot(
            pipeline,
            max_samples=10
        )
        print(f"✅ Pipeline ready with {len(questions)} questions!")