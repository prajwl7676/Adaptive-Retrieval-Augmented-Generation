
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import json
import os

# Optional imports (install if using ML features)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError: 
    ML_AVAILABLE = False


class MLQueryClassifier:
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_ml: bool = True,
        cache_dir: str = "./classifier_cache"
    ):
        self.categories = ["qa", "summarization", "reasoning", "extraction", "coding"]
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx_to_category = {i: cat for cat, i in self.category_to_idx.items()}
        
        self.use_ml = use_ml and ML_AVAILABLE and EMBEDDINGS_AVAILABLE
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE and self.use_ml:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_model = None
            self.embedding_dim = 0
        
        # Training data storage
        self.training_queries = []
        self.training_labels = []
        self.training_embeddings = []
        
        # ML models
        if ML_AVAILABLE and self.use_ml:
            self.models = {
                "rf": RandomForestClassifier(n_estimators=100, random_state=42),
                "gb": GradientBoostingClassifier(n_estimators=50, random_state=42),
                "lr": LogisticRegression(max_iter=1000, random_state=42)
            }
            self.scaler = StandardScaler()
            self.is_trained = False
        else:
            self.models = {}
            self.scaler = None
            self.is_trained = False
        
        # Performance tracking
        self.performance_history = []
        self.category_performance = {cat: [] for cat in self.categories}
        
        # Load cached data if available
        self._load_cache()
    
    def classify_query(self, query: str, use_ml: bool = None) -> str:
        
        if use_ml is None:
            use_ml = self.use_ml and self.is_trained
        
        if use_ml and self.is_trained:
            return self._classify_ml(query)
        else:
            # Fallback to enhanced rule-based classifier
            from category_classifier import AdaptiveQueryClassifier
            fallback = AdaptiveQueryClassifier()
            return fallback.classify_query(query)
    
    def classify_with_confidence(
        self, 
        query: str,
        use_ml: bool = None
    ) -> Tuple[str, float, Dict[str, float]]:
        
        if use_ml is None:
            use_ml = self.use_ml and self.is_trained
        
        if use_ml and self.is_trained:
            probs = self._get_ml_probabilities(query)
            best_cat = max(probs, key=probs.get)
            confidence = probs[best_cat]
            return best_cat, confidence, probs
        else:
            # Use rule-based with confidence
            from category_classifier import classify_query_with_confidence
            return classify_query_with_confidence(query)
    
    def _classify_ml(self, query: str) -> str:
        
        if not self.is_trained:
            raise ValueError("ML models not trained yet")
        
        # Get embedding
        embedding = self.embedding_model.encode([query])[0]
        embedding_scaled = self.scaler.transform([embedding])
        
        # Ensemble voting
        votes = []
        for model_name, model in self.models.items():
            pred = model.predict(embedding_scaled)[0]
            votes.append(pred)
        
        # Majority vote
        vote_counts = Counter(votes)
        best_idx = vote_counts.most_common(1)[0][0]
        
        return self.idx_to_category[best_idx]
    
    def _get_ml_probabilities(self, query: str) -> Dict[str, float]:
        
        if not self.is_trained:
            return {cat: 1.0/len(self.categories) for cat in self.categories}
        
        embedding = self.embedding_model.encode([query])[0]
        embedding_scaled = self.scaler.transform([embedding])
        
        # Average probabilities from all models
        all_probs = []
        for model in self.models.values():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(embedding_scaled)[0]
                all_probs.append(probs)
        
        if not all_probs:
            return {cat: 1.0/len(self.categories) for cat in self.categories}
        
        # Average ensemble
        avg_probs = np.mean(all_probs, axis=0)
        
        return {self.idx_to_category[i]: float(p) for i, p in enumerate(avg_probs)}
    
    def add_training_example(self, query: str, category: str, feedback_score: float = 1.0):
       
        if category not in self.categories:
            raise ValueError(f"Invalid category: {category}")
        
        self.training_queries.append(query)
        self.training_labels.append(self.category_to_idx[category])
        
        # Store embedding
        if self.embedding_model:
            embedding = self.embedding_model.encode([query])[0]
            self.training_embeddings.append(embedding)
        
        # Track performance
        self.category_performance[category].append(feedback_score)
    
    def train_models(self, min_samples: int = 20):
        
        if not self.use_ml:
            print("ML not available or disabled")
            return False
        
        if len(self.training_queries) < min_samples * len(self.categories):
            print(f"Not enough training data: {len(self.training_queries)} samples")
            return False
        
        # Check class balance
        label_counts = Counter(self.training_labels)
        if min(label_counts.values()) < min_samples:
            print(f"Insufficient samples for some categories: {label_counts}")
            return False
        
        # Prepare training data
        X = np.array(self.training_embeddings)
        y = np.array(self.training_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X_scaled, y)
        
        self.is_trained = True
        self._save_cache()
        
        print(f"✅ Models trained on {len(self.training_queries)} samples")
        return True
    
    def evaluate_models(self, test_queries: List[str], test_labels: List[str]) -> Dict:
       
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        predictions = [self.classify_query(q, use_ml=True) for q in test_queries]
        
        correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_queries)
        
        # Per-category accuracy
        category_accuracy = {}
        for cat in self.categories:
            cat_indices = [i for i, label in enumerate(test_labels) if label == cat]
            if cat_indices:
                cat_correct = sum(1 for i in cat_indices if predictions[i] == test_labels[i])
                category_accuracy[cat] = cat_correct / len(cat_indices)
        
        return {
            "overall_accuracy": accuracy,
            "category_accuracy": category_accuracy,
            "num_samples": len(test_queries)
        }
    
    def bootstrap_from_rules(self, num_samples: int = 100):
        
        from category_classifier import AdaptiveQueryClassifier
        rule_classifier = AdaptiveQueryClassifier()
        
        # Category-specific query templates
        templates = {
            "qa": [
                "What is {}?",
                "Who is {}?",
                "When did {} happen?",
                "Where is {} located?",
                "Define {}"
            ],
            "summarization": [
                "Summarize {}",
                "Give me a brief overview of {}",
                "What are the main points of {}?",
                "TL;DR for {}",
                "Recap {}"
            ],
            "reasoning": [
                "Why does {} happen?",
                "How does {} work?",
                "Explain the logic behind {}",
                "What causes {}?",
                "Derive {}"
            ],
            "extraction": [
                "List all {} in the document",
                "Extract {} from the text",
                "Find all instances of {}",
                "Pull out {}",
                "Identify all {}"
            ],
            "coding": [
                "Write a function to {}",
                "Debug this {} code",
                "Implement {} algorithm",
                "Fix the bug in {}",
                "Create a class for {}"
            ]
        }
        
        # Common topics for filling templates
        topics = [
            "machine learning", "climate change", "quantum computing",
            "DNA replication", "neural networks", "black holes",
            "photosynthesis", "blockchain", "natural selection",
            "protein folding", "market dynamics", "cell division"
        ]
        
        samples_per_category = num_samples // len(self.categories)
        
        for category in self.categories:
            for _ in range(samples_per_category):
                template = np.random.choice(templates[category])
                topic = np.random.choice(topics)
                query = template.format(topic)
                
                self.add_training_example(query, category, feedback_score=0.8)
        
        print(f"✅ Generated {len(self.training_queries)} bootstrap samples")
    
    def _save_cache(self):

        cache_file = os.path.join(self.cache_dir, "classifier_cache.json")
        
        cache_data = {
            "training_queries": self.training_queries,
            "training_labels": self.training_labels,
            "training_embeddings": [emb.tolist() for emb in self.training_embeddings],
            "category_performance": self.category_performance,
            "is_trained": self.is_trained
        }
        
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        
        # Save sklearn models separately
        if ML_AVAILABLE and self.is_trained:
            import pickle
            models_file = os.path.join(self.cache_dir, "models.pkl")
            with open(models_file, "wb") as f:
                pickle.dump({
                    "models": self.models,
                    "scaler": self.scaler
                }, f)
    
    def _load_cache(self):
        
        cache_file = os.path.join(self.cache_dir, "classifier_cache.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            self.training_queries = cache_data.get("training_queries", [])
            self.training_labels = cache_data.get("training_labels", [])
            self.training_embeddings = [
                np.array(emb) for emb in cache_data.get("training_embeddings", [])
            ]
            self.category_performance = cache_data.get("category_performance", 
                                                       {cat: [] for cat in self.categories})
            
            # Load sklearn models
            if ML_AVAILABLE:
                models_file = os.path.join(self.cache_dir, "models.pkl")
                if os.path.exists(models_file):
                    import pickle
                    with open(models_file, "rb") as f:
                        model_data = pickle.load(f)
                        self.models = model_data["models"]
                        self.scaler = model_data["scaler"]
                        self.is_trained = True
                    
                    print(f"✅ Loaded classifier cache: {len(self.training_queries)} samples")




class HybridClassifier:
    
    
    def __init__(self, use_ml: bool = True):
        self.ml_classifier = MLQueryClassifier(use_ml=use_ml)
        
        # Bootstrap with synthetic data if no training data exists
        if len(self.ml_classifier.training_queries) < 50:
            self.ml_classifier.bootstrap_from_rules(num_samples=100)
            self.ml_classifier.train_models(min_samples=10)
    
    def __call__(self, query: str) -> str:
       
        category, confidence, _ = self.ml_classifier.classify_with_confidence(query)
        
        # If low confidence, could trigger multiple retrieval strategies
        if confidence < 0.6:
            print(f"⚠️ Low confidence ({confidence:.2f}) for query classification")
        
        return category
    
    def classify_with_feedback(
        self, 
        query: str, 
        pipeline_performance: float
    ) -> Tuple[str, float]:
        
        category, confidence, _ = self.ml_classifier.classify_with_confidence(query)
        
        # Add to training data if performance is conclusive
        if pipeline_performance > 0.7 or pipeline_performance < 0.3:
            self.ml_classifier.add_training_example(
                query, category, pipeline_performance
            )
        
        # Retrain periodically
        if len(self.ml_classifier.training_queries) % 50 == 0:
            self.ml_classifier.train_models(min_samples=10)
        
        return category, confidence
    


def example_usage():
    
    # Option 1: Use ML classifier directly
    ml_clf = MLQueryClassifier(use_ml=True)
    ml_clf.bootstrap_from_rules(num_samples=200)
    ml_clf.train_models(min_samples=15)
    
    query = "Why does neural network training require backpropagation?"
    category, confidence, probs = ml_clf.classify_with_confidence(query)
    print(f"\nQuery: {query}")
    print(f"Category: {category} (confidence: {confidence:.3f})")
    print(f"All probabilities: {probs}")
    
    # Option 2: Use hybrid classifier (recommended)
    hybrid = HybridClassifier(use_ml=True)
    
    test_queries = [
        "Summarize the key findings",
        "Extract all dates mentioned",
        "Why does this algorithm work?",
        "Write a function to sort the list",
        "What is machine learning?"
    ]
    
    for q in test_queries:
        cat = hybrid(q)
        print(f"'{q}' → {cat}")