import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

class AdaptiveQueryClassifier:
    
    def __init__(self):
        self.categories = ["qa", "summarization", "reasoning", "extraction", "coding"]
        
        # Track category performance for adaptive weighting
        self.category_performance = {cat: {"correct": 0, "total": 0} for cat in self.categories}
        
        # Enhanced keyword patterns with weights
        self.keyword_patterns = {
            "coding": {
                "high": ["function", "class", "method", "bug", "debug", "compile", "syntax error", 
                        "implement", "refactor", "algorithm", "data structure"],
                "medium": ["code", "python", "java", "javascript", "programming", "script"],
                "low": ["program", "software", "development"]
            },
            "summarization": {
                "high": ["summarize", "summary", "tl;dr", "brief overview", "main points", 
                        "key takeaways", "in short"],
                "medium": ["overview", "recap", "condense", "brief"],
                "low": ["highlights", "essence"]
            },
            "extraction": {
                "high": ["extract", "list all", "find all", "pull out", "identify all",
                        "enumerate", "what are the"],
                "medium": ["list", "find", "get", "retrieve", "collect"],
                "low": ["show", "display"]
            },
            "reasoning": {
                "high": ["why", "how does", "explain", "reason", "derive", "prove",
                        "logical", "cause", "justify", "analyze"],
                "medium": ["because", "therefore", "logic", "rationale", "inference"],
                "low": ["understand", "clarify"]
            },
            "qa": {
                "high": ["what is", "who is", "when", "where", "which", "define"],
                "medium": ["tell me", "describe", "information about"],
                "low": ["about", "regarding"]
            }
        }
        
        # Question patterns for structural analysis
        self.question_patterns = {
            "reasoning": [
                r"^(why|how)\s+(does|do|did|is|are|can|could|would|should)",
                r"explain.*why",
                r"what.*cause",
                r"how.*work"
            ],
            "extraction": [
                r"(list|name|identify|find)\s+(all|the|any)",
                r"what\s+are\s+(all\s+)?the",
                r"^(extract|pull|get)\s+"
            ],
            "summarization": [
                r"(summarize|overview|recap|brief).*",
                r"what.*main\s+(point|idea|theme)",
                r"in\s+short"
            ]
        }
    
    def classify_query(self, query: str) -> str:

        query_lower = query.lower().strip()
        
        # Get scores from different classification methods
        keyword_scores = self._score_keywords(query_lower)
        pattern_scores = self._score_patterns(query_lower)
        structure_scores = self._score_query_structure(query_lower)
        complexity_signal = self._analyze_complexity(query_lower)
        
        # Combine scores with adaptive weights
        combined_scores = self._combine_signals(
            keyword_scores, 
            pattern_scores, 
            structure_scores,
            complexity_signal
        )
        
        # Get best category
        best_category = max(combined_scores, key=combined_scores.get)
        
        return best_category
    
    def classify_with_confidence(self, query: str) -> Tuple[str, float, Dict[str, float]]:
       
        query_lower = query.lower().strip()
        
        keyword_scores = self._score_keywords(query_lower)
        pattern_scores = self._score_patterns(query_lower)
        structure_scores = self._score_query_structure(query_lower)
        complexity_signal = self._analyze_complexity(query_lower)
        
        combined_scores = self._combine_signals(
            keyword_scores, 
            pattern_scores, 
            structure_scores,
            complexity_signal
        )
        
        # Normalize scores to get probabilities
        total = sum(combined_scores.values())
        if total > 0:
            probabilities = {k: v/total for k, v in combined_scores.items()}
        else:
            probabilities = {k: 1.0/len(self.categories) for k in self.categories}
        
        best_category = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_category]
        
        return best_category, confidence, probabilities
    
    def _score_keywords(self, query: str) -> Dict[str, float]:
    
        scores = defaultdict(float)
        
        for category, patterns in self.keyword_patterns.items():
            for priority, keywords in patterns.items():
                weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[priority]
                for keyword in keywords:
                    if keyword in query:
                        scores[category] += weight
        
        return dict(scores)
    
    def _score_patterns(self, query: str) -> Dict[str, float]:
        
        scores = defaultdict(float)
        
        for category, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    scores[category] += 2.0
        
        return dict(scores)
    
    def _score_query_structure(self, query: str) -> Dict[str, float]:
        
        scores = defaultdict(float)
        
        # Question word analysis
        question_words = {
            "what": 0.5, "who": 0.5, "when": 0.5, "where": 0.5,
            "why": 2.0, "how": 2.0, "which": 0.3
        }
        
        first_word = query.split()[0] if query.split() else ""
        if first_word in question_words:
            if first_word in ["why", "how"]:
                scores["reasoning"] += question_words[first_word]
            else:
                scores["qa"] += question_words[first_word]
        
        # Length analysis
        word_count = len(query.split())
        if word_count < 5:
            scores["qa"] += 0.5  # Short queries often QA
        elif word_count > 15:
            scores["reasoning"] += 0.5  # Complex queries often reasoning
        
        # Code indicators
        if re.search(r'[{}\[\]()<>]', query) or '```' in query:
            scores["coding"] += 2.0
        
        # List indicators
        if re.search(r'\d+\.|\*|\-', query):  # Numbered or bullet lists
            scores["extraction"] += 1.0
        
        return dict(scores)
    
    def _analyze_complexity(self, query: str) -> Dict[str, float]:

        scores = defaultdict(float)
        
        # Complexity indicators
        complexity_score = 0
        
        # Multi-clause queries
        clause_markers = ["and", "but", "however", "therefore", "because"]
        complexity_score += sum(1 for marker in clause_markers if marker in query)
        
        # Technical terminology
        if len([w for w in query.split() if len(w) > 10]) > 2:
            complexity_score += 1
        
        # Conditional language
        if any(word in query for word in ["if", "when", "assuming", "given"]):
            complexity_score += 1
        
        # Distribute complexity signal
        if complexity_score >= 2:
            scores["reasoning"] += 1.0
        elif complexity_score == 1:
            scores["reasoning"] += 0.5
            scores["qa"] += 0.3
        else:
            scores["qa"] += 0.5
        
        return dict(scores)
    
    def _combine_signals(
        self, 
        keyword_scores: Dict[str, float],
        pattern_scores: Dict[str, float],
        structure_scores: Dict[str, float],
        complexity_scores: Dict[str, float]
    ) -> Dict[str, float]:
        
        combined = defaultdict(float)
        
        # Base weights (can be tuned)
        weights = {
            "keywords": 1.0,
            "patterns": 1.5,  # Regex patterns are reliable
            "structure": 1.0,
            "complexity": 0.8
        }
        
        # Add weighted scores
        for category in self.categories:
            combined[category] += keyword_scores.get(category, 0) * weights["keywords"]
            combined[category] += pattern_scores.get(category, 0) * weights["patterns"]
            combined[category] += structure_scores.get(category, 0) * weights["structure"]
            combined[category] += complexity_scores.get(category, 0) * weights["complexity"]
            
            # Apply performance-based adjustment
            perf = self.category_performance[category]
            if perf["total"] > 10:  # Only after sufficient samples
                accuracy = perf["correct"] / perf["total"]
                # Boost well-performing categories slightly
                combined[category] *= (0.9 + accuracy * 0.2)
        
        # Default to QA if no strong signals
        if max(combined.values()) < 1.0:
            combined["qa"] = 1.0
        
        return dict(combined)
    
    def update_performance(self, category: str, was_correct: bool):
        
        if category in self.category_performance:
            self.category_performance[category]["total"] += 1
            if was_correct:
                self.category_performance[category]["correct"] += 1
    
    def get_multi_category(self, query: str, threshold: float = 0.2) -> List[Tuple[str, float]]:
       
        _, _, probabilities = self.classify_with_confidence(query)
        
        # Filter categories above threshold
        candidates = [
            (cat, prob) for cat, prob in probabilities.items() 
            if prob >= threshold
        ]
        
        # Sort by probability descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates




# Global classifier instance
_global_classifier = AdaptiveQueryClassifier()

def classify_query(query: str) -> str:
   
    return _global_classifier.classify_query(query)

def classify_query_with_confidence(query: str) -> Tuple[str, float, Dict[str, float]]:
    
    return _global_classifier.classify_with_confidence(query)

def classify_query_multi(query: str, threshold: float = 0.2) -> List[Tuple[str, float]]:
   
    return _global_classifier.get_multi_category(query, threshold)



CATEGORIES = ["qa", "summarization", "reasoning", "extraction", "coding"]