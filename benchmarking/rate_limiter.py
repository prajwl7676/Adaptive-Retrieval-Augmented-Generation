

import time
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_wait_time: float = 0.0
    circuit_breaker_opens: int = 0


class CircuitBreaker:
    """Circuit breaker pattern for API failures."""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 120.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call_failed(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def call_succeeded(self):
        """Record a successful call."""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            logger.info("Circuit breaker closed")
    
    def can_attempt(self) -> bool:
        """Check if we can attempt a call."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if we should try again
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.reset_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker half-open, trying request")
                    return True
            return False
        
        # half_open state
        return True


class AdaptiveRateLimiter:
    
    def __init__(
        self,
        requests_per_minute: int = 15,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
        exponential_base: float = 2.0,
        failure_threshold: int = 3,
        reset_timeout: float = 120.0
    ):
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.exponential_base = exponential_base
        
        # Token bucket
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_update = time.time()
        self.token_refill_rate = requests_per_minute / 60.0  # tokens per second
        
        # Request timestamps (sliding window)
        self.request_times = deque(maxlen=requests_per_minute)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold, reset_timeout)
        
        # Statistics
        self.stats = RateLimiterStats()
    
    def _refill_tokens(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_elapsed = now - self.last_update
        tokens_to_add = time_elapsed * self.token_refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_update = now
    
    def _wait_for_token(self):
        """Wait until a token is available."""
        self._refill_tokens()
        
        if self.tokens >= 1:
            self.tokens -= 1
            return
        
        # Calculate wait time
        tokens_needed = 1 - self.tokens
        wait_time = tokens_needed / self.token_refill_rate
        
        logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s...")
        time.sleep(wait_time)
        self.stats.total_wait_time += wait_time
        
        self._refill_tokens()
        self.tokens -= 1
    
    def _clean_old_requests(self):
        """Remove requests older than 1 minute from sliding window."""
        cutoff_time = time.time() - 60
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time."""
        backoff = self.initial_backoff * (self.exponential_base ** attempt)
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0, 0.1 * backoff)
        return min(backoff + jitter, self.max_backoff)
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
       
        self.stats.total_requests += 1
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            # Check circuit breaker
            if not self.circuit_breaker.can_attempt():
                wait_time = self.circuit_breaker.reset_timeout
                logger.warning(f"Circuit breaker open. Waiting {wait_time}s...")
                time.sleep(wait_time)
                self.stats.total_wait_time += wait_time
                continue
            
            try:
                # Wait for rate limit
                self._wait_for_token()
                self._clean_old_requests()
                self.request_times.append(time.time())
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Success
                self.stats.successful_requests += 1
                self.circuit_breaker.call_succeeded()
                return result
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = any(
                    keyword in error_msg
                    for keyword in ["rate limit", "quota", "429", "too many requests"]
                )
                
                # Check if it's a retryable error
                is_retryable = is_rate_limit or any(
                    keyword in error_msg
                    for keyword in ["timeout", "connection", "503", "500"]
                )
                
                if not is_retryable or attempt == self.max_retries:
                    self.stats.failed_requests += 1
                    self.circuit_breaker.call_failed()
                    logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                    raise
                
                # Calculate backoff
                backoff_time = self._calculate_backoff(attempt)
                self.stats.retried_requests += 1
                
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying in {backoff_time:.2f}s... Error: {e}"
                )
                
                time.sleep(backoff_time)
                self.stats.total_wait_time += backoff_time
        
        # Should not reach here, but just in case
        self.stats.failed_requests += 1
        raise last_exception
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "retried_requests": self.stats.retried_requests,
            "success_rate": (
                self.stats.successful_requests / self.stats.total_requests * 100
                if self.stats.total_requests > 0 else 0
            ),
            "total_wait_time": self.stats.total_wait_time,
            "avg_wait_per_request": (
                self.stats.total_wait_time / self.stats.total_requests
                if self.stats.total_requests > 0 else 0
            ),
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_tokens": self.tokens
        }


def rate_limited(rate_limiter: AdaptiveRateLimiter):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return rate_limiter.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create rate limiter (15 requests per minute)
    limiter = AdaptiveRateLimiter(requests_per_minute=15)
    
    @rate_limited(limiter)
    def mock_api_call(i):
        """Simulate an API call."""
        print(f"API call {i}")
        if i % 10 == 0:  # Simulate occasional failure
            raise Exception("Simulated rate limit error")
        return f"Result {i}"
    
    # Test with multiple calls
    for i in range(20):
        try:
            result = mock_api_call(i)
            print(f"  -> {result}")
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    # Print statistics
    print("\nRate Limiter Statistics:")
    stats = limiter.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")