

import os
import sys
from datetime import datetime
from typing import Optional, TextIO, Dict
from pathlib import Path

# Add parent directory to path to import pipeline_logger
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BenchmarkLogger:
   
    
    def __init__(self, benchmark_name: str, log_dir: str = "../benchmark_logs", verbose_console: bool = False):
       
        self.benchmark_name = benchmark_name
        # Convert to absolute path if relative
        if not os.path.isabs(log_dir):
            # Relative to the benchmarking folder
            benchmarking_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = Path(os.path.join(benchmarking_dir, log_dir))
        else:
            self.log_dir = Path(log_dir)
        
        self.verbose_console = verbose_console
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{benchmark_name}_{timestamp}.log"
        self.log_path = self.log_dir / self.log_filename
        
        # Open log file
        self.log_file: Optional[TextIO] = open(self.log_path, 'w', encoding='utf-8')
        
        # Write header
        self._write_header()
        
    def _write_header(self):
        """Write initial header to log file."""
        header = f"""
{'='*80}
BENCHMARK LOG: {self.benchmark_name.upper()}
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Log File: {self.log_path}
{'='*80}
"""
        self.log_file.write(header)
        self.log_file.flush()
        
        # Print minimal info to console
        print(f"\n📋 Logging to: {self.log_path}")
        print(f"{'='*80}\n")
    
    def log(self, message: str, to_console: bool = True, level: str = "INFO"):
        
        if self.log_file is None:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        # Always write to file
        self.log_file.write(formatted_msg + "\n")
        self.log_file.flush()
        
        # Conditionally write to console
        if to_console or self.verbose_console:
            print(message)
    
    def log_progress(self, current: int, total: int, message: str = ""):
        
        progress_msg = f"[{current}/{total}] {message}"
        self.log(progress_msg, to_console=True, level="PROGRESS")
    
    def log_section(self, title: str):
        """Log a section header."""
        separator = "=" * 80
        section_msg = f"\n{separator}\n{title}\n{separator}"
        self.log(section_msg, to_console=self.verbose_console, level="SECTION")
    
    def log_subsection(self, title: str):
        """Log a subsection header."""
        separator = "-" * 80
        subsection_msg = f"\n{separator}\n{title}\n{separator}"
        self.log(subsection_msg, to_console=False, level="SUBSECTION")
    
    def log_metric(self, name: str, value: float, to_console: bool = True):
        """Log a metric value."""
        metric_msg = f"  {name}: {value:.4f}"
        self.log(metric_msg, to_console=to_console, level="METRIC")
    
    def log_error(self, message: str):
        """Log an error message (always shown in console)."""
        self.log(f"❌ ERROR: {message}", to_console=True, level="ERROR")
    
    def log_success(self, message: str):
        """Log a success message (always shown in console)."""
        self.log(f"✅ {message}", to_console=True, level="SUCCESS")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(f"⚠️  WARNING: {message}", to_console=True, level="WARNING")
    
    def log_result(
        self, 
        question: str, 
        generated: str, 
        expected: str, 
        matched: bool,
        bert_scores: Optional[Dict[str, float]] = None,
        bert_threshold: float = 0.85,
        llm_match: Optional[bool] = None
    ):
        
        # Truncate generated answer for display
        generated_display = generated[:200] + ('...' if len(generated) > 200 else '')
        
        # Base result block
        result_block = f"""
Query: {question}
Generated: {generated_display}
Expected: {expected}
Match: {'✓' if matched else '✗'}"""
        
        # Add LLM Judge Result
        if llm_match is not None:
             result_block += f"\nLLM Judge:    {'✓' if llm_match else '✗'}"
        
        # Add BERT scores if provided
        if bert_scores:
            bert_match = bert_scores.get('f1', 0.0) >= bert_threshold
            result_block += f"""
BERT Precision: {bert_scores.get('precision', 0.0):.4f}
BERT Recall: {bert_scores.get('recall', 0.0):.4f}
BERT F1: {bert_scores.get('f1', 0.0):.4f}
BERT Match (≥ {bert_threshold}): {'✓' if bert_match else '✗'}"""
        
        self.log(result_block, to_console=False, level="RESULT")
    
    def log_summary(self, results: dict):
            """
            Log benchmark summary statistics.
            """
            summary_msg = f"""
    {'='*80}
    BENCHMARK SUMMARY
    {'='*80}
    Total Samples: {results.get('total', 0)}

    1. String/Rule Match:
    Correct: {results.get('correct', 0)}
    Accuracy: {results.get('accuracy', 0):.2%}
    """
            
            # Add LLM Judge statistics
            if 'llm_correct' in results:
                summary_msg += f"""
    2. LLM Judge Match (Ollama):
    Correct: {results.get('llm_correct', 0)}
    Accuracy: {results.get('llm_accuracy', 0):.2%}
    """

            # Add BERT match statistics
            if 'bert_correct' in results:
                summary_msg += f"""
    3. BERT Semantic Match (≥ {results.get('bert_threshold', 0.85)}):
    Correct: {results.get('bert_correct', 0)}
    Accuracy: {results.get('bert_accuracy', 0):.2%}
    """
            
            self.log(summary_msg, to_console=True, level="SUMMARY")
            
            # Log timing metrics if available
            if 'avg_metrics' in results:
                self.log("\nAverage Pipeline Metrics:", to_console=True)
                for key, value in results['avg_metrics'].items():
                    self.log_metric(key, value, to_console=True)
    
    def close(self):
        """Close the log file."""
        if self.log_file is not None:
            footer = f"""
{'='*80}
Benchmark Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.close()
            self.log_file = None
            print(f"\n✅ Log saved to: {self.log_path}\n")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Convenience function to patch pipeline_logger for benchmarks
def redirect_pipeline_logger_to_file(log_file_path: str):
    
    import pipeline_logger
    
    # Create a file handler
    log_file = open(log_file_path, 'a', encoding='utf-8')
    
    # Monkey-patch all logging functions to write to file
    original_functions = {}
    
    for attr_name in dir(pipeline_logger):
        if attr_name.startswith('log_'):
            original_func = getattr(pipeline_logger, attr_name)
            if callable(original_func):
                original_functions[attr_name] = original_func
                
                # Create wrapper that writes to file
                def make_wrapper(func, file_obj):
                    def wrapper(*args, **kwargs):
                        # Redirect stdout temporarily
                        old_stdout = sys.stdout
                        sys.stdout = file_obj
                        try:
                            result = func(*args, **kwargs)
                            file_obj.flush()
                            return result
                        finally:
                            sys.stdout = old_stdout
                    return wrapper
                
                setattr(pipeline_logger, attr_name, make_wrapper(original_func, log_file))
    
    return log_file, original_functions


# Example usage demonstration
if __name__ == "__main__":
    # Example 1: Basic usage with BERT scores
    with BenchmarkLogger("test_benchmark") as logger:
        logger.log_section("Starting Test Benchmark")
        
        for i in range(5):
            logger.log_progress(i+1, 5, f"Processing query {i+1}")
            
            # Simulate BERT scores
            bert_scores = {
                'precision': 0.90 + i * 0.01,
                'recall': 0.92 + i * 0.01,
                'f1': 0.91 + i * 0.01
            }
            
            logger.log_result(
                question=f"What is {i}+{i}?",
                generated=f"The answer is {i*2}",
                expected=f"{i*2}",
                matched=True,
                bert_scores=bert_scores
            )
        
        logger.log_summary({
            'total': 5,
            'correct': 4,
            'accuracy': 0.8,
            'bert_correct': 5,
            'bert_accuracy': 1.0,
            'bert_threshold': 0.85,
            'avg_metrics': {
                'retrieval_time': 0.234,
                'generation_time': 0.567
            }
        })