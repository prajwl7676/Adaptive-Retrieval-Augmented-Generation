import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

# UPDATED IMPORT - Use adaptive pipeline
from src.adaptive.adpt_rag_pipeline import AdaptiveRAGPipeline
from benchmark_logger import BenchmarkLogger
from rate_limiter import AdaptiveRateLimiter
from corpus_manager import CorpusManager
from benchmark_config import BenchmarkConfig, RetrievalMethod, get_config
from metrics_calculator import MetricsCalculator
from src.generation.generation_ollama import AnswerGenerator


class AdaptiveHotpotBenchmark:
    """Enhanced HotpotQA benchmark with adaptive RAG pipeline."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
        # Initialize logger
        self.logger = BenchmarkLogger(
            "hotpot_adaptive",
            log_dir=config.log_dir,
            verbose_console=config.verbose_console
        )

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            bert_model="distilbert-base-uncased"  # Fast model for benchmarking
        )
        
        # BERT matching threshold
        self.bert_threshold = 0.85

        # Initialize answer generator for evaluation (using Ollama)
        self.answer_generator = AnswerGenerator(
            api_key=config.api_key,
            default_model=config.model_name
        )
        
        # Initialize rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=config.api_config.requests_per_minute,
            max_retries=config.api_config.max_retries,
            initial_backoff=config.api_config.initial_backoff,
            max_backoff=config.api_config.max_backoff,
            failure_threshold=config.api_config.failure_threshold,
            reset_timeout=config.api_config.reset_timeout
        )
        
        # Initialize corpus manager
        self.corpus_manager = CorpusManager(
            cache_dir=config.corpus_config.cache_dir,
            cache_enabled=config.corpus_config.cache_corpus,
            force_reload=config.corpus_config.force_reload
        )
        
        self.pipeline: Optional[AdaptiveRAGPipeline] = None
        self.questions: List[Dict] = []
        self.results: List[Dict] = []
    
    def setup_pipeline(self):
        """Initialize the Adaptive RAG pipeline with HotpotQA corpus."""
        self.logger.log_section("Adaptive Pipeline Setup")
        
        # Initialize adaptive pipeline (always uses hybrid retrieval)
        self.logger.log("Initializing Adaptive RAG Pipeline...", to_console=True)
        self.pipeline = AdaptiveRAGPipeline(
            api_key=self.config.api_key,
            answer_model_name=self.config.model_name,
            full_doc_relevance_threshold=1.5,
            summary_relevance_threshold=1.5,
            bert_model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load and initialize corpus
        self.logger.log("Loading HotpotQA corpus...", to_console=True)
        self.questions = self.corpus_manager.initialize_pipeline_with_hotpot(
            self.pipeline,
            max_samples=self.config.sample_size,
            include_distractors=self.config.corpus_config.hotpot_include_distractors
        )
        
        self.logger.log_success(
            f"Pipeline ready with {len(self.questions)} questions\n"
            f"  • RL Controller initialized for categories: {list(self.pipeline.controller.routers.keys())}\n"
            f"  • Arm Space: {self.pipeline.arm_space}\n"
            f"  • Default Genome params: {len(self.pipeline.tuned_params)} parameters"
        )
    
    def run_single_query(
        self,
        question_data: Dict,
        index: int,
        total: int
    ) -> Dict:
       
        question = question_data["question"]
        expected_answer = question_data["answer"]
        
        # Show progress
        self.logger.log_progress(
            index + 1,
            total,
            f"{question[:60]}..."
        )
        
        start_time = time.time()
        
        try:
            # Execute query with adaptive pipeline using run_with_genome
            # The adaptive pipeline returns: {answer, reward, ops, genome, steps}
            result = self.rate_limiter.execute_with_retry(
                self.pipeline.run_with_genome,
                question,
                self.pipeline.tuned_params  # Pass current genome
            )
            
            generated_answer = result["answer"]
            
            # Extract adaptive pipeline metadata
            selected_ops = result.get("ops", {})
            genome_used = result.get("genome", {})
            pipeline_steps = result.get("steps", {})
            rl_reward = result.get("reward", 0.0)
            category = self.pipeline.category
            
            # Check if answer matches (case-insensitive substring)
            match = self._check_answer_match(expected_answer, generated_answer)
            
            # Calculate BERT scores for this single query
            bert_scores = None
            bert_match = False
            try:
                bert_scores = self.metrics_calculator.bert_score(
                    [generated_answer],
                    [expected_answer],
                    verbose=False
                )
                bert_match = bert_scores.get('f1', 0.0) >= self.bert_threshold
            except Exception as e:
                self.logger.log(f"BERT scoring failed for query {index + 1}: {e}", to_console=False)

            # LLM-based evaluation
            llm_match = False
            try:
                llm_match = self.answer_generator.evaluate_answer_match(
                    question, 
                    expected_answer, 
                    generated_answer
                )
            except Exception as e:
                self.logger.log(f"LLM evaluation failed for query {index + 1}: {e}", to_console=False)
            
            query_time = time.time() - start_time
            
            # Create result entry with adaptive metadata
            result_entry = {
                "question_id": question_data["id"],
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "match": match,
                "llm_match": llm_match,
                "bert_match": bert_match,
                "bert_scores": bert_scores,
                "question_type": question_data.get("type", "unknown"),
                "question_level": question_data.get("level", "unknown"),
                
                # Adaptive pipeline specific metrics
                "category": category,
                "selected_operators": selected_ops,
                "genome_parameters": genome_used,
                "pipeline_steps": pipeline_steps,
                "rl_reward": rl_reward,
                
                "total_time": query_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log result with all metrics
            self.logger.log_result(
                question=question,
                generated=generated_answer,
                expected=expected_answer,
                matched=match,
                bert_scores=bert_scores,
                bert_threshold=self.bert_threshold,
                llm_match=llm_match
            )
            
            # Enhanced console output
            status = "✓" if match else "✗"
            bert_status = f" | BERT: {'✓' if bert_match else '✗'}" if bert_scores else ""
            llm_status = f" | LLM: {'✓' if llm_match else '✗'}"
            ops_info = f" | Ops: R:{selected_ops.get('retrieval', '?')[:3]}, S:{selected_ops.get('summary', '?')[:3]}, G:{selected_ops.get('generation', '?')[:3]}"
            print(f"    Result: {status}{bert_status}{llm_status}{ops_info} ({query_time:.2f}s)")
            
            return result_entry
            
        except Exception as e:
            self.logger.log_error(f"Query {index + 1} failed: {str(e)}")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", to_console=False)
            return {
                "question_id": question_data["id"],
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": f"ERROR: {str(e)}",
                "match": False,
                "bert_match": False,
                "llm_match": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_benchmark(self):
        """Run the complete benchmark with adaptive pipeline."""
        self.logger.log_section("Starting Adaptive HotpotQA Benchmark")
        self.logger.log(f"Sample Size: {len(self.questions)}")
        self.logger.log(f"BERT F1 Threshold: {self.bert_threshold}")
        self.logger.log(f"RL Categories: {list(self.pipeline.controller.routers.keys())}")
        
        # Run queries
        self.results = []
        for i, question_data in enumerate(self.questions):
            result = self.run_single_query(question_data, i, len(self.questions))
            self.results.append(result)
        
        # Calculate standard metrics
        summary = self._calculate_summary()
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics()
        summary['advanced_metrics'] = advanced_metrics
        
        # Calculate adaptive-specific metrics
        adaptive_stats = self._calculate_adaptive_stats()
        summary['adaptive_stats'] = adaptive_stats
        
        self.logger.log_summary(summary)
        
        # Log advanced metrics
        self._log_advanced_metrics(advanced_metrics)
        
        # Log adaptive statistics
        self._log_adaptive_stats(adaptive_stats)
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        # Save RL policies for each category
        self._save_rl_policies()
        
        # Print rate limiter stats
        self._log_rate_limiter_stats()
        
        return summary
    
    def _calculate_adaptive_stats(self) -> Dict:
        
        stats = {
            "category_distribution": {},
            "operator_usage": {
                "retrieval": {},
                "summary": {},
                "rerank": {},
                "generation": {}
            },
            "operator_performance": {},
            "genome_analysis": {},
            "rl_rewards": []
        }
        
        # Category distribution
        for result in self.results:
            category = result.get("category", "unknown")
            if category not in stats["category_distribution"]:
                stats["category_distribution"][category] = {
                    "count": 0,
                    "correct": 0,
                    "bert_correct": 0,
                    "llm_correct": 0
                }
            stats["category_distribution"][category]["count"] += 1
            if result.get("match"):
                stats["category_distribution"][category]["correct"] += 1
            if result.get("bert_match"):
                stats["category_distribution"][category]["bert_correct"] += 1
            if result.get("llm_match"):
                stats["category_distribution"][category]["llm_correct"] += 1
        
        # Operator usage frequency
        for result in self.results:
            ops = result.get("selected_operators", {})
            for stage, op_name in ops.items():
                if stage in stats["operator_usage"]:
                    if op_name not in stats["operator_usage"][stage]:
                        stats["operator_usage"][stage][op_name] = 0
                    stats["operator_usage"][stage][op_name] += 1
        
        # Operator performance (accuracy when using specific operators)
        for result in self.results:
            ops = result.get("selected_operators", {})
            op_key = f"{ops.get('retrieval', '?')}-{ops.get('summary', '?')}-{ops.get('generation', '?')}"
            
            if op_key not in stats["operator_performance"]:
                stats["operator_performance"][op_key] = {
                    "count": 0,
                    "correct": 0,
                    "bert_correct": 0,
                    "llm_correct": 0
                }
            stats["operator_performance"][op_key]["count"] += 1
            if result.get("match"):
                stats["operator_performance"][op_key]["correct"] += 1
            if result.get("bert_match"):
                stats["operator_performance"][op_key]["bert_correct"] += 1
            if result.get("llm_match"):
                stats["operator_performance"][op_key]["llm_correct"] += 1
        
        # RL rewards distribution
        stats["rl_rewards"] = [r.get("rl_reward", 0) for r in self.results if "rl_reward" in r]
        if stats["rl_rewards"]:
            stats["avg_rl_reward"] = sum(stats["rl_rewards"]) / len(stats["rl_rewards"])
            stats["min_rl_reward"] = min(stats["rl_rewards"])
            stats["max_rl_reward"] = max(stats["rl_rewards"])
        
        return stats
    
    def _log_adaptive_stats(self, stats: Dict):
        """Log adaptive pipeline statistics."""
        self.logger.log_section("Adaptive Pipeline Statistics")
        
        # Category distribution
        if stats.get("category_distribution"):
            self.logger.log("\n📊 Query Category Distribution:", to_console=True)
            for category, data in stats["category_distribution"].items():
                acc = data["correct"] / data["count"] if data["count"] > 0 else 0
                bert_acc = data["bert_correct"] / data["count"] if data["count"] > 0 else 0
                llm_acc = data["llm_correct"] / data["count"] if data["count"] > 0 else 0
                self.logger.log(
                    f"  {category:15s}: {data['count']:3d} queries | "
                    f"Acc: {acc:.2%} | BERT: {bert_acc:.2%} | LLM: {llm_acc:.2%}",
                    to_console=True
                )
        
        # Operator usage
        if stats.get("operator_usage"):
            self.logger.log("\n🔧 Operator Usage Frequency:", to_console=True)
            for stage, ops in stats["operator_usage"].items():
                self.logger.log(f"  {stage.capitalize()}:", to_console=True)
                for op_name, count in sorted(ops.items(), key=lambda x: x[1], reverse=True):
                    pct = count / len(self.results) * 100
                    self.logger.log(f"    {op_name:20s}: {count:3d} ({pct:.1f}%)", to_console=True)
        
        # Top performing operator combinations
        if stats.get("operator_performance"):
            self.logger.log("\n🏆 Top Operator Combinations:", to_console=True)
            sorted_combos = sorted(
                stats["operator_performance"].items(),
                key=lambda x: (x[1]["correct"] / x[1]["count"] if x[1]["count"] > 0 else 0),
                reverse=True
            )[:5]
            
            for combo, perf in sorted_combos:
                acc = perf["correct"] / perf["count"] if perf["count"] > 0 else 0
                bert_acc = perf["bert_correct"] / perf["count"] if perf["count"] > 0 else 0
                llm_acc = perf["llm_correct"] / perf["count"] if perf["count"] > 0 else 0
                self.logger.log(
                    f"  {combo:40s}: Acc {acc:.2%} | BERT {bert_acc:.2%} | LLM {llm_acc:.2%} ({perf['count']} queries)",
                    to_console=True
                )
        
        # RL Rewards
        if "avg_rl_reward" in stats:
            self.logger.log("\n🎯 RL Reward Statistics:", to_console=True)
            self.logger.log(f"  Average: {stats['avg_rl_reward']:.4f}", to_console=True)
            self.logger.log(f"  Min: {stats['min_rl_reward']:.4f}", to_console=True)
            self.logger.log(f"  Max: {stats['max_rl_reward']:.4f}", to_console=True)
    
    def _save_rl_policies(self):
        """Save RL policies for all categories."""
        try:
            for category in self.pipeline.controller.routers.keys():
                policy_path = f"rl_policies/{category}.json"
                # The controller already saves to these files automatically
                self.logger.log(f"RL policy for '{category}' saved to {policy_path}", to_console=False)
            
            self.logger.log_success("RL policies saved for all categories")
        except Exception as e:
            self.logger.log_error(f"Failed to save RL policies: {e}")
    
    def _calculate_advanced_metrics(self) -> Dict[str, float]:
        """Calculate advanced evaluation metrics (F1, EM, ROUGE, BERTScore)."""
        self.logger.log_section("Calculating Advanced Metrics")
        
        valid_results = [
            r for r in self.results 
            if "error" not in r and r.get("generated_answer") and r.get("expected_answer")
        ]
        
        if not valid_results:
            self.logger.log("No valid results for metrics calculation")
            return {}
        
        predictions = [r["generated_answer"] for r in valid_results]
        ground_truths = [r["expected_answer"] for r in valid_results]
        
        self.logger.log(f"Calculating metrics for {len(valid_results)} predictions...", to_console=True)
        
        try:
            metrics = self.metrics_calculator.calculate_batch_metrics(
                predictions,
                ground_truths,
                verbose=True
            )
            return metrics
        except Exception as e:
            self.logger.log_error(f"Failed to calculate advanced metrics: {e}")
            return {}

    def _log_advanced_metrics(self, metrics: Dict[str, float]):
        """Log advanced metrics in a formatted way."""
        if not metrics:
            return
        
        self.logger.log_section("Advanced Evaluation Metrics")
        
        basic_metrics = ['exact_match', 'f1', 'precision', 'recall']
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        bert_metrics = ['bert_precision', 'bert_recall', 'bert_f1']
        
        self.logger.log("\n📊 Basic Metrics:", to_console=True)
        for metric in basic_metrics:
            if metric in metrics:
                value = metrics[metric]
                self.logger.log(f"  {metric:15s}: {value:.4f}", to_console=True)
        
        if any(m in metrics for m in rouge_metrics):
            self.logger.log("\n📝 ROUGE Scores:", to_console=True)
            for metric in rouge_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    self.logger.log(f"  {metric:15s}: {value:.4f}", to_console=True)
        
        if any(m in metrics for m in bert_metrics):
            self.logger.log("\n🤖 BERTScore:", to_console=True)
            for metric in bert_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    self.logger.log(f"  {metric:15s}: {value:.4f}", to_console=True)
    
    def _check_answer_match(self, expected: str, generated: str) -> bool:
        
        import re
        
        expected_lower = expected.lower().strip()
        generated_lower = generated.lower().strip()
        
        def normalize_text(text):
            text = ' '.join(text.split())
            text = re.sub(r'[,;:\'\"]', '', text)
            return text
        
        expected_norm = normalize_text(expected_lower)
        generated_norm = normalize_text(generated_lower)
        
        if expected_norm == generated_norm:
            return True
        
        rejection_phrases = [
            "cannot answer", "unable to", "i cannot", "i was unable",
            "no information", "not found", "don't have", "do not have",
            "insufficient information", "not enough information",
            "not mentioned", "does not mention", "doesn't mention"
        ]
        
        first_part = generated_lower[:150]
        if any(phrase in first_part for phrase in rejection_phrases):
            for phrase in rejection_phrases:
                if phrase in first_part:
                    phrase_pos = first_part.find(phrase)
                    if expected_norm in generated_lower[:phrase_pos]:
                        return True
            return False
        
        if expected_norm in ['yes', 'no']:
            first_30 = generated_lower[:30]
            
            if expected_norm == 'yes':
                yes_patterns = [
                    r'\byes\b', r'\byes,', r'\byes\.', r'answer is yes',
                    r'answer: yes', r'\baffirmative\b', r'\bcorrect\b', r'\btrue\b'
                ]
                if any(re.search(pattern, first_30) for pattern in yes_patterns):
                    return True
            
            elif expected_norm == 'no':
                no_patterns = [
                    r'\bno\b', r'\bno,', r'\bno\.', r'answer is no',
                    r'answer: no', r'\bnegative\b', r'\bincorrect\b',
                    r'\bfalse\b', r'\bnot both\b', r'\bare not\b',
                    r'\bdo not\b', r'\bdoes not\b'
                ]
                
                yes_pos = generated_lower.find('yes')
                no_pos = generated_lower.find('no')
                
                if yes_pos != -1 and yes_pos < no_pos:
                    return False
                
                if any(re.search(pattern, first_30) for pattern in no_patterns):
                    return True
            
            return False
        
        expected_numbers = re.findall(r'\d+(?:,\d+)*', expected)
        generated_numbers = re.findall(r'\d+(?:,\d+)*', generated_lower[:200])
        
        if expected_numbers:
            expected_nums_clean = [n.replace(',', '') for n in expected_numbers]
            generated_nums_clean = [n.replace(',', '') for n in generated_numbers]
            
            if all(any(exp_num in gen_num or gen_num in exp_num 
                    for gen_num in generated_nums_clean) 
                for exp_num in expected_nums_clean):
                return True
        
        if expected_norm in generated_norm:
            match_pos = generated_lower.find(expected_lower)
            if match_pos < 200:
                return True
        
        expected_words = set(expected_norm.split())
        generated_words = set(generated_norm.split())
        
        if len(expected_words) > 1:
            common_words = expected_words & generated_words
            stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'and', 'or', 'but'}
            significant_expected = expected_words - stopwords
            significant_common = common_words - stopwords
            
            if not significant_expected:
                return expected_norm in generated_norm
            
            overlap_ratio = len(significant_common) / len(significant_expected)
            
            if overlap_ratio >= 0.75:
                return True
        
        return False
    
    def _calculate_summary(self) -> dict:
        """Calculate benchmark summary statistics."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("match", False))
        accuracy = correct / total if total > 0 else 0
        
        bert_correct = sum(1 for r in self.results if r.get("bert_match", False))
        bert_accuracy = bert_correct / total if total > 0 else 0
        
        llm_correct = sum(1 for r in self.results if r.get("llm_match", False))
        llm_accuracy = llm_correct / total if total > 0 else 0

        type_stats = {}
        for result in self.results:
            q_type = result.get("question_type", "unknown")
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "correct": 0, "bert_correct": 0, "llm_correct": 0}
            type_stats[q_type]["total"] += 1
            if result.get("match", False):
                type_stats[q_type]["correct"] += 1
            if result.get("bert_match", False):
                type_stats[q_type]["bert_correct"] += 1
            if result.get("llm_match", False):
                type_stats[q_type]["llm_correct"] += 1
        
        level_stats = {}
        for result in self.results:
            q_level = result.get("question_level", "unknown")
            if q_level not in level_stats:
                level_stats[q_level] = {"total": 0, "correct": 0, "bert_correct": 0, "llm_correct": 0}
            level_stats[q_level]["total"] += 1
            if result.get("match", False):
                level_stats[q_level]["correct"] += 1
            if result.get("bert_match", False):
                level_stats[q_level]["bert_correct"] += 1
            if result.get("llm_match", False):
                level_stats[q_level]["llm_correct"] += 1
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "bert_correct": bert_correct,
            "bert_accuracy": bert_accuracy,
            "bert_threshold": self.bert_threshold,
            "llm_correct": llm_correct,
            "llm_accuracy": llm_accuracy,
            "type_breakdown": {
                k: {
                    "accuracy": v["correct"] / v["total"],
                    "bert_accuracy": v["bert_correct"] / v["total"],
                    "llm_accuracy": v["llm_correct"] / v["total"],
                    "count": v["total"]
                }
                for k, v in type_stats.items()
            },
            "level_breakdown": {
                k: {
                    "accuracy": v["correct"] / v["total"],
                    "bert_accuracy": v["bert_correct"] / v["total"],
                    "llm_accuracy": v["llm_correct"] / v["total"],
                    "count": v["total"]
                }
                for k, v in level_stats.items()
            }
        }
    
    def _save_results(self):
        """Save benchmark results to JSON."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptive_hotpot_results_{timestamp}.json"
        filepath = results_dir / filename
        
        output_data = {
            "config": {
                "sample_size": len(self.questions),
                "model_name": self.config.model_name,
                "bert_threshold": self.bert_threshold,
                "timestamp": timestamp,
                "arm_space": self.pipeline.arm_space,
                "default_genome": self.pipeline.tuned_params
            },
            "results": self.results,
            "summary": self._calculate_summary(),
            "rate_limiter_stats": self.rate_limiter.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.log_success(f"Results saved to: {filepath}")
    
    def _log_rate_limiter_stats(self):
        """Log rate limiter statistics."""
        stats = self.rate_limiter.get_stats()
        
        self.logger.log_section("Rate Limiter Statistics")
        self.logger.log(f"Total Requests: {stats['total_requests']}", to_console=True)
        self.logger.log(f"Successful: {stats['successful_requests']}", to_console=True)
        self.logger.log(f"Failed: {stats['failed_requests']}", to_console=True)
        self.logger.log(f"Retried: {stats['retried_requests']}", to_console=True)
        self.logger.log(f"Success Rate: {stats['success_rate']:.2f}%", to_console=True)
        self.logger.log(f"Total Wait Time: {stats['total_wait_time']:.2f}s", to_console=True)
        self.logger.log(f"Avg Wait/Request: {stats['avg_wait_per_request']:.2f}s", to_console=True)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive HotpotQA Benchmark")
    parser.add_argument(
        '--config',
        type=str,
        default='quick_test',
        choices=['quick_test', 'full_hotpot_chromadb', 'full_hotpot_hybrid'],
        help='Predefined configuration to use'
    )
    parser.add_argument('--samples', type=int, help='Override sample size')
    parser.add_argument('--bert-threshold', type=float, default=0.85, help='BERT F1 threshold for matching')
    parser.add_argument('--model', type=str, default='qwen3:8b', help='Ollama model name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Apply overrides
    if args.samples:
        config.sample_size = args.samples
    
    if args.model:
        config.model_name = args.model
    
    # Print configuration
    print("\n" + "="*80)
    print("🚀 Adaptive HotpotQA Benchmark")
    print("="*80)
    print(f"\nConfiguration: {args.config}")
    print(f"  • Sample Size: {config.sample_size or 'ALL'}")
    print(f"  • Model: {config.model_name}")
    print(f"  • Rate Limit: {config.api_config.requests_per_minute} req/min")
    print(f"  • Corpus Cache: {config.corpus_config.cache_corpus}")
    print(f"  • BERT Threshold: {args.bert_threshold}")
    print(f"  • Adaptive Features: RL-based operator selection, genome tuning")
    print("\n" + "="*80 + "\n")
    
    # Run benchmark
    benchmark = AdaptiveHotpotBenchmark(config)
    benchmark.bert_threshold = args.bert_threshold  # Set custom threshold
    benchmark.setup_pipeline()
    summary = benchmark.run_benchmark()
    
    # Print final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"\n🎯 Standard Accuracy: {summary['accuracy']:.2%}")
    print(f"   Correct: {summary['correct']}/{summary['total']}")
    
    print(f"\n🤖 BERT Accuracy (≥ {summary['bert_threshold']}): {summary['bert_accuracy']:.2%}")
    print(f"   BERT Matches: {summary['bert_correct']}/{summary['total']}")
    
    print(f"\n💬 LLM Evaluation Accuracy: {summary['llm_accuracy']:.2%}")
    print(f"   LLM Matches: {summary['llm_correct']}/{summary['total']}")
    
    if 'type_breakdown' in summary:
        print("\n📋 By Question Type:")
        for q_type, stats in summary['type_breakdown'].items():
            print(f"  {q_type}:")
            print(f"    Standard: {stats['accuracy']:.2%} ({stats['count']} questions)")
            print(f"    BERT: {stats['bert_accuracy']:.2%}")
            print(f"    LLM: {stats['llm_accuracy']:.2%}")
    
    if 'level_breakdown' in summary:
        print("\n🎚️  By Difficulty:")
        for level, stats in summary['level_breakdown'].items():
            print(f"  {level}:")
            print(f"    Standard: {stats['accuracy']:.2%} ({stats['count']} questions)")
            print(f"    BERT: {stats['bert_accuracy']:.2%}")
            print(f"    LLM: {stats['llm_accuracy']:.2%}")
    
    # Show adaptive stats summary
    if 'adaptive_stats' in summary:
        adaptive = summary['adaptive_stats']
        print("\n🔄 Adaptive Pipeline Summary:")
        if 'category_distribution' in adaptive:
            print("  Category Usage:")
            for cat, data in adaptive['category_distribution'].items():
                print(f"    {cat}: {data['count']} queries")
        if 'avg_rl_reward' in adaptive:
            print(f"  Average RL Reward: {adaptive['avg_rl_reward']:.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()   