# 🌟 Adaptive Retrieval-Augmented Generation (Adaptive-RAG)
This repository reproduces **my specific core contributions** to a larger 8-member project at **University of Paderborn** (Grade: **1.7**). The files are structured according to the contributions and does not represent the original repository structure of the project from university. All the contributions from other members are excluded.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-black.svg)](https://ollama.com/)


An advanced, self-tuning **Adaptive RAG Pipeline** featuring hybrid multi-source retrieval, intelligent query classification, interchangeable local/cloud generation backends, and an evolutionary genome-based parameter optimization algorithm.



---

## 🎯 Key Contributions & Impact
* **Hybrid Retrieval & RRF**: Integrated semantic searches (BERT, ChromaDB) and lexical searches (TF-IDF) using **Reciprocal Rank Fusion (RRF)** to generate highly relevant context.
* **Modular Generation Backends**: Engineered a unified interface allowing hot-swapping between **local LLMs (Ollama)** and **commercial APIs (OpenAI/Gemini)**.
* **Dual-Layer Query Classifier**: Developed a fast machine learning query classifier backed by a robust rule-based fallback mechanism to route queries efficiently.
* **Evolutionary Optimization**: Architected an evolutionary self-tuning optimizer tuning **30+ hyper-parameters** across the entire RAG pipeline.
* **Benchmarking & Evaluation Framework**: Built evaluation pipelines using **SQuAD** and **HotpotQA** datasets. Evaluated **1,000+ test samples**, achieving a baseline **0.68 BERTScore**.

---

## 📁 Repository Structure
```text
adaptive-rag-portfolio/
├── .gitignore                   # Excludes caches, virtual environments, and datasets
├── README.md                    # Premium project documentation & contribution map
├── requirements.txt             # Project dependencies (ChromaDB, scikit-learn, transformers, etc.)
├── run_demo.py                  # End-to-end demo script running the RAG pipeline
│
├── src/                         # Core codebase
│   ├── retrieval/               # 🔹 CONTRIBUTED: Hybrid Retrieval & RRF
│   │   ├── __init__.py
│   │   ├── retrieval.py         # Reciprocal Rank Fusion (RRF) & ChromaDB orchestration
│   │   ├── retrieval_tfidf.py   # TF-IDF lexical retrieval implementation
│   │   └── retrieval_bert.py    # BERT-based embedding semantic retrieval
│   │
│   ├── category/                # 🔹 CONTRIBUTED: Query Classifier
│   │   ├── __init__.py
│   │   ├── ml_query_classifier.py          # ML-based intent & complexity classifier
│   │   └── enhanced_category_classifier.py # Rule-based fallback classifier
│   │
│   ├── generation/              # 🔹 CONTRIBUTED: Interchangeable Backends
│   │   ├── __init__.py
│   │   ├── generation.py        # Abstract interface for generation (OpenAI, Gemini, etc.)
│   │   └── generation_ollama.py # Local LLM integration via Ollama
│   │
│   └── evolution/               # 🔹 CONTRIBUTED: Evolutionary Optimization Algorithm
│       ├── __init__.py
│       └── evo_optimizer.py     # Parameter optimizer tuning a 30+ parameter genome
│
├── benchmarking/                # 🔹 CONTRIBUTED: Evaluation & Benchmarking
│   ├── metrics_calculator.py    # ROUGE & BERTScore computation
│   ├── hotpot_benchmark_adaptive.py # Performance runner for HotpotQA dataset
│   ├── squad_benchmark.py       # Performance runner for SQuAD dataset
│   ├── benchmark_logger.py      # Custom logging system for evaluation runs
│   ├── corpus_manager.py        # Loading and managing dataset corpora
│   ├── rate_limiter.py          # API rate-limiting helper for cloud providers
│   ├── benchmark_config.py      # Configuration setup for benchmark evaluation
│   └── data/                    # Tiny mock/sample datasets (10-20 samples) for validation
│
├── benchmark_logs/              # 🔹 Output directory for evaluation logs
│   └── adaptive_rag_benchmark.log # Sample log file showcasing RAG evaluation metrics
│
├── genomes/                     # Output directory for evolutionary parameters
│   └── sample_genome.json       # Pre-optimized 30+ parameter genome sample
│
└── notebooks/                   # Jupyter notebooks showcasing results
    └── adaptive_rag_demo.ipynb  # Interactive notebook with step-by-step visualizations & plots
```

---

## 🛠️ Contribution Deep-Dive

### 1. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Combines the precision of keyword searching with the conceptual understanding of semantic embeddings.
* **Lexical Match (TF-IDF)**: Extracts keyword-heavy context.
* **Semantic Match (BERT & ChromaDB)**: Extracts conceptual/contextual relationships.
* **Reciprocal Rank Fusion (RRF)**: Re-ranks document results based on their position in both search methods, preventing bias towards any single model.

### 2. Intelligent Query Classifier
Optimizes computing resources by determining if a query needs full retrieval, simple retrieval, or direct LLM completion.
* **ML Classifier**: Standard classification model to label query complexity/type.
* **Rule-Based Fallback**: Fast regex/heuristic model that steps in if the ML model is uncertain.

### 3. Evolutionary Optimization (Genome)
The pipeline's behavior changes dynamically based on a **30+ parameter genome** stored in a JSON configuration. The evolutionary algorithm mutates and optimizes:
* **Retrieval Weights**: Balance between TF-IDF and BERT scores.
* **Thresholds**: Minimal similarity scores required to feed context to the LLM.
* **LLM Prompts & Parameters**: Temperature, top-p, and context windows dynamically adjusted depending on query type.

### 4. Interchangeable LLM Generation
Provides unified methods to send prompts and receive responses across providers:
* **Cloud APIs**: OpenAI GPT, Google Gemini.
* **Local Run**: Ollama hosting models like Llama 3.

---

## 📊 Benchmarking & Results
Tested on a subset of **HotpotQA** and **SQuAD** datasets, evaluating output quality against ground truth.

| Metric | Target / Score | Dataset | Description |
|---|---|---|---|
| **BERTScore** | **0.68** | HotpotQA (1k samples) | Measure of semantic similarity |


---

