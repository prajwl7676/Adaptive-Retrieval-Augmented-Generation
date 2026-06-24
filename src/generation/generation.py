import time
import google.generativeai as genai
from typing import List, Dict, Tuple, Iterator
import requests
import google.api_core.exceptions
import logging
import tenacity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        
        self.api_key = api_key
        self.default_model = default_model
        genai.configure(api_key=self.api_key)
        logger.info(f"AnswerGenerator initialized with default model: {self.default_model}")

    @tenacity.retry(
        # Wait strategy: exponential backoff from 1 second up to 10 seconds.
        # multiplier=1: base delay is 1 * (2^attempt_number - 1)
        # min=1: minimum wait time is 1 second
        # max=10: maximum wait time for a single retry is 10 seconds
        wait=tenacity.wait_exponential(multiplier=60, min=1, max=300),
        
        # Stop strategy: stop after 5 attempts in total (1 original + 4 retries).
        stop=tenacity.stop_after_attempt(5),
        
        # Retry conditions: only retry for specific transient API errors.
        retry=(
            tenacity.retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) | # Rate Limit (429)
            tenacity.retry_if_exception_type(google.api_core.exceptions.InternalServerError) | # 500 errors
            tenacity.retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable) | # 503 errors
            tenacity.retry_if_exception_type(google.api_core.exceptions.DeadlineExceeded) # 504 timeouts
        ),
        
        # Log before each retry attempt.
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        
        # If all retries fail, re-raise the last exception.
        reraise=True
    )
    def _make_gemini_api_call(self, messages: List[Dict[str, str]],
                              model: str,
                              max_tokens: int,
                              temperature: float,
                              stream: bool = False) -> Tuple[str, float] | Iterator[str]:
        
        logger.info(f"--- GENERATION PHASE (Gemini SDK) ---")
        start_time = time.time()
        
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            logger.info(f"Attempting Gemini API call (model: {model}, stream: {stream})...")
            model_instance = genai.GenerativeModel(model_name=model)

            response = model_instance.generate_content(
                messages,
                generation_config=generation_config,
                stream=stream
            )

            if stream:
                logger.info(f"Started streaming response from Gemini API.")
                return (chunk.text for chunk in response) # Yield generator for streaming
            else:
                answer = response.text.strip()
                generation_time = time.time() - start_time
                logger.info(f"Generated answer in {generation_time:.4f} seconds (model: {model}).")
                return answer, generation_time

        # These exceptions will be caught by tenacity if they are in the 'retry' list.
        # If tenacity exhausts its retries, the final exception will be re-raised from here.
        except google.api_core.exceptions.ResourceExhausted as e:
            logger.warning(f"Gemini API rate limit hit ({e.code}): {e}. This error is retriable.")
            raise # Re-raise for tenacity to handle
        except google.api_core.exceptions.InternalServerError as e:
            logger.warning(f"Gemini API Internal Server Error ({e.code}): {e}. This error is retriable.")
            raise # Re-raise for tenacity to handle
        except google.api_core.exceptions.ServiceUnavailable as e:
            logger.warning(f"Gemini API Service Unavailable ({e.code}): {e}. This error is retriable.")
            raise # Re-raise for tenacity to handle
        except google.api_core.exceptions.DeadlineExceeded as e:
            logger.warning(f"Gemini API Deadline Exceeded ({e.code}): {e}. This error is retriable.")
            raise # Re-raise for tenacity to handle

        # --- Non-retriable Google API Errors (or other Google API errors) ---
        except google.api_core.exceptions.BadRequest as e:
            logger.error(f"Gemini API Bad Request (Non-retriable, {e.code}): {e}. Check input format/length.")
            raise # This error is typically due to bad input, so no retry.
        except google.api_core.exceptions.Unauthorized as e:
            logger.critical(f"Gemini API Unauthorized (Non-retriable, {e.code}): {e}. Please check your API key.")
            raise # API key issue, no retry needed.
        except google.api_core.exceptions.GoogleAPIError as e:
            # Catch any other specific Google API errors not explicitly handled above.
            logger.error(f"Unhandled Gemini API error ({e.code}): {e}. Review exception type for specific handling.")
            raise
        
        # --- Catch-all for any other unexpected Python errors ---
        except Exception as e:
            logger.critical(f"An unexpected non-GoogleAPI error occurred during API call: {type(e).__name__} - {e}")
            raise # Re-raise to ensure calling functions are aware of the failure.

    def generate_answer_gemini(self, prompt: str, model: str = None,
                               max_tokens: int = 500, temperature: float = 0.2) -> Tuple[str, float]:
        
        actual_model = model if model is not None else self.default_model
        messages = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        try:
            return self._make_gemini_api_call(messages, actual_model, max_tokens, temperature, stream=False)
        except Exception as e:
            # Catch the final exception if tenacity exhausts its retries or for non-retriable errors.
            logger.error(f"Failed to generate answer for prompt after all retries: {e}")
            return f"Error generating response: {str(e)}", time.time() - time.time() # Return 0 for duration

    def generate_answer_with_messages(self, messages: List[Dict[str, str]],
                                      model: str = None,
                                      max_tokens: int = 500,
                                      temperature: float = 0.2) -> Tuple[str, float]:
       
        actual_model = model if model is not None else self.default_model
        try:
            return self._make_gemini_api_call(messages, actual_model, max_tokens, temperature, stream=False)
        except Exception as e:
            logger.error(f"Failed to generate answer with messages after all retries: {e}")
            return f"Error generating response: {str(e)}", time.time() - time.time() # Return 0 for duration

    def generate_streaming_answer(self, messages: List[Dict[str, str]],
                                  model: str = None,
                                  max_tokens: int = 500,
                                  temperature: float = 0.2) -> Iterator[str]:
        
        actual_model = model if model is not None else self.default_model
        try:
            # yield from ensures that if _make_gemini_api_call returns an iterator,
            # its items are directly yielded by this function.
            yield from self._make_gemini_api_call(messages, actual_model, max_tokens, temperature, stream=True)
        except Exception as e:
            logger.error(f"Failed to initiate or complete streaming generation after all retries: {e}")
            yield f"Error generating response: {str(e)}" # Yield error message as a single chunk