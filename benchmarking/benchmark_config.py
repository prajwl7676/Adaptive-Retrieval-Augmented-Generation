

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RetrievalMethod(Enum):
    """Supported retrieval methods."""
    CHROMADB = "chromadb"
    TFIDF = "tfidf"
    BERT = "bert"
    HYBRID_CHROMA_TFIDF = "hybrid_chroma_tfidf"
    HYBRID_ALL = "hybrid_all"  # ChromaDB + TF-IDF + BERT


class BenchmarkDataset(Enum):
    """Supported benchmark datasets."""
    SQUAD = "squad"
    HOTPOT = "hotpot_qa"
    MSMARCO = "msmarco"


@dataclass
class APIRateLimitConfig:
    """Configuration for API rate limiting."""
    requests_per_minute: int = 15  # Gemini Flash free tier limit
    max_retries: int = 5
    initial_backoff: float = 2.0  # seconds
    max_backoff: float = 60.0  # seconds
    exponential_base: float = 2.0
    
    # Circuit breaker pattern
    failure_threshold: int = 3  # consecutive failures before circuit opens
    reset_timeout: float = 120.0  # seconds to wait before trying again


@dataclass
class CorpusConfig:
    """Configuration for corpus initialization."""
    # HotpotQA specific
    hotpot_load_full_corpus: bool = True  # Load all supporting documents
    hotpot_max_samples: Optional[int] = None  # None = all samples
    hotpot_include_distractors: bool = True
    
    # SQuAD specific
    squad_load_full_corpus: bool = False  # SQuAD has its own corpus
    
    # General
    cache_corpus: bool = True  # Cache processed corpus to disk
    cache_dir: str = "../corpus_cache"
    force_reload: bool = False  # Force reload even if cached


@dataclass
class RetrievalConfig:
    """Configuration for retrieval methods."""
    # ChromaDB
    chromadb_enabled: bool = True
    chromadb_k: int = 20
    chromadb_collection_name: str = "rag_benchmark"
    
    # TF-IDF
    tfidf_enabled: bool = False
    tfidf_k: int = 20
    tfidf_max_features: int = 10000
    
    # BERT (MPNet)
    bert_enabled: bool = False
    bert_k: int = 20
    bert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    bert_batch_size: int = 32
    
    # Hybrid (RRF)
    rrf_k: int = 60  # RRF constant
    
    # Reranking
    rerank_enabled: bool = True
    rerank_top_k: int = 5
    
    # Document processing
    max_chunks_per_doc: int = 3
    adaptive_k: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # BERT scoring
    enable_bert_scoring: bool = True
    bert_threshold: float = 0.85  # F1 threshold for BERT match
    bert_model: str = "distilbert-base-uncased"  # Fast model for benchmarking
    
    # Other metrics
    calculate_rouge: bool = True
    calculate_exact_match: bool = True


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    # Dataset
    dataset: BenchmarkDataset = BenchmarkDataset.HOTPOT
    sample_size: Optional[int] = None  # None = all samples
    
    # Retrieval methods to test
    retrieval_methods: List[RetrievalMethod] = field(
        default_factory=lambda: [RetrievalMethod.CHROMADB]
    )
    
    # Sub-configs
    api_config: APIRateLimitConfig = field(default_factory=APIRateLimitConfig)
    corpus_config: CorpusConfig = field(default_factory=CorpusConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Pipeline settings
    direct_retrieval: bool = True  # Skip full doc check for speed
    use_chat_format: bool = True
    disable_graph_generation: bool = True  # Disable graph generation for benchmarking
    
    # Output settings
    save_results: bool = True
    results_dir: str = "../benchmarking/results/raw"
    generate_report: bool = True
    
    # Logging
    log_dir: str = "../benchmark_logs"
    verbose_console: bool = False
    
    # API
    api_key: Optional[str] = None
    model_name: str = "qwen3:8b"
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Enable retrieval methods based on config
        if RetrievalMethod.HYBRID_CHROMA_TFIDF in self.retrieval_methods:
            self.retrieval_config.chromadb_enabled = True
            self.retrieval_config.tfidf_enabled = True
        
        if RetrievalMethod.HYBRID_ALL in self.retrieval_methods:
            self.retrieval_config.chromadb_enabled = True
            self.retrieval_config.tfidf_enabled = True
            self.retrieval_config.bert_enabled = True
        
        # Adjust rate limits for paid tier
        if "paid" in self.model_name.lower() or "pro" in self.model_name.lower():
            self.api_config.requests_per_minute = 60


# Predefined configurations
QUICK_TEST_CONFIG = BenchmarkConfig(
    dataset=BenchmarkDataset.HOTPOT,
    sample_size=5,
    retrieval_methods=[RetrievalMethod.TFIDF],
    corpus_config=CorpusConfig(hotpot_load_full_corpus=False)
)

FULL_HOTPOT_CHROMADB_CONFIG = BenchmarkConfig(
    dataset=BenchmarkDataset.HOTPOT,
    sample_size=1000,  # or None for all
    retrieval_methods=[RetrievalMethod.CHROMADB],
    corpus_config=CorpusConfig(hotpot_load_full_corpus=True)
)

FULL_HOTPOT_HYBRID_CONFIG = BenchmarkConfig(
    dataset=BenchmarkDataset.HOTPOT,
    sample_size=1000,
    retrieval_methods=[RetrievalMethod.HYBRID_ALL],
    corpus_config=CorpusConfig(hotpot_load_full_corpus=True),
    retrieval_config=RetrievalConfig(
        chromadb_enabled=True,
        tfidf_enabled=True,
        bert_enabled=True
    )
)

ABLATION_STUDY_CONFIG = BenchmarkConfig(
    dataset=BenchmarkDataset.HOTPOT,
    sample_size=100,
    retrieval_methods=[
        RetrievalMethod.CHROMADB,
        RetrievalMethod.TFIDF,
        RetrievalMethod.BERT,
        RetrievalMethod.HYBRID_ALL
    ],
    corpus_config=CorpusConfig(hotpot_load_full_corpus=True)
)


def get_config(config_name: str = "quick_test") -> BenchmarkConfig:
    
    configs = {
        "quick_test": QUICK_TEST_CONFIG,
        "full_hotpot_chromadb": FULL_HOTPOT_CHROMADB_CONFIG,
        "full_hotpot_hybrid": FULL_HOTPOT_HYBRID_CONFIG,
        "ablation_study": ABLATION_STUDY_CONFIG
    }
    
    return configs.get(config_name, QUICK_TEST_CONFIG)


if __name__ == "__main__":
    # Example usage
    config = get_config("full_hotpot_hybrid")
    print(f"Dataset: {config.dataset.value}")
    print(f"Sample Size: {config.sample_size}")
    print(f"Retrieval Methods: {[m.value for m in config.retrieval_methods]}")
    print(f"API Rate Limit: {config.api_config.requests_per_minute} req/min")
    print(f"BERT Threshold: {config.evaluation_config.bert_threshold}")
    print(f"Graph Generation: {'Disabled' if config.disable_graph_generation else 'Enabled'}")
