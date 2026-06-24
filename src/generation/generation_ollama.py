import time
import requests
import logging
from typing import List, Dict, Tuple, Iterator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, api_key: str = None, default_model: str = "qwen3:8b"):
        """
        Initializes the AnswerGenerator with Ollama (local model).
        api_key parameter is kept for compatibility but not used.
        """
        self.default_model = default_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_chat_url = "http://localhost:11434/api/chat"
        logger.info(f"AnswerGenerator initialized with Ollama model: {self.default_model}")
        
        # Test connection to Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Successfully connected to Ollama server")
            else:
                logger.warning("⚠️  Ollama server not responding. Make sure 'ollama serve' is running!")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama. Make sure 'ollama serve' is running! Error: {e}")

    def generate_answer_gemini(self, prompt: str, model: str = None,
                               max_tokens: int = 3000, temperature: float = 0.7) -> Tuple[str, float]:
        
        logger.info(f"--- GENERATION PHASE (Ollama Local) ---")
        start_time = time.time()
        actual_model = model if model else self.default_model
        
        try:
            logger.info(f"Calling Ollama API (model: {actual_model})...")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": actual_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            answer = response.json()["response"].strip()
            generation_time = time.time() - start_time
            logger.info(f"Generated answer in {generation_time:.4f} seconds (model: {actual_model}).")
            return answer, generation_time
            
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama. Make sure 'ollama serve' is running!"
            logger.error(error_msg)
            return f"Error: {error_msg}", 0.0
        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timed out after 120 seconds"
            logger.error(error_msg)
            return f"Error: {error_msg}", 0.0
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}", 0.0

    def generate_answer_with_messages(self, messages: List[Dict[str, str]],
                                      model: str = None,
                                      max_tokens: int = 500,
                                      temperature: float = 0.2) -> Tuple[str, float]:
        
        logger.info(f"--- GENERATION PHASE (Ollama Local - Chat Format) ---")
        start_time = time.time()
        actual_model = model if model else self.default_model
        
        try:
            # Convert messages format for Ollama
            ollama_messages = []
            for msg in messages:
                if isinstance(msg.get('content'), str):
                    ollama_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
                elif isinstance(msg.get('parts'), list):
                    # Handle Gemini-style format
                    content = " ".join([part.get('text', '') for part in msg['parts']])
                    ollama_messages.append({
                        "role": msg.get('role', 'user'),
                        "content": content
                    })
            
            logger.info(f"Calling Ollama Chat API (model: {actual_model})...")
            response = requests.post(
                self.ollama_chat_url,
                json={
                    "model": actual_model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"].strip()
            generation_time = time.time() - start_time
            logger.info(f"Generated answer in {generation_time:.4f} seconds.")
            return answer, generation_time
            
        except Exception as e:
            logger.error(f"Error generating response with messages: {e}")
            return f"Error: {str(e)}", 0.0

    def generate_streaming_answer(self, messages: List[Dict[str, str]],
                                  model: str = None,
                                  max_tokens: int = 500,
                                  temperature: float = 0.2) -> Iterator[str]:
        
        actual_model = model if model else self.default_model
        
        try:
            # Convert messages
            ollama_messages = []
            for msg in messages:
                if isinstance(msg.get('content'), str):
                    ollama_messages.append({"role": msg['role'], "content": msg['content']})
                elif isinstance(msg.get('parts'), list):
                    content = " ".join([part.get('text', '') for part in msg['parts']])
                    ollama_messages.append({"role": msg.get('role', 'user'), "content": content})
            
            response = requests.post(
                self.ollama_chat_url,
                json={
                    "model": actual_model,
                    "messages": ollama_messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"]["content"]
                        
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error: {str(e)}"

    def evaluate_answer_match(self, question: str, expected: str, generated: str, model: str = None) -> bool:
        
        logger.info(f"--- EVALUATION PHASE (LLM Judge) ---")
        
        prompt = f"""
You are an impartial judge evaluating the correctness of an AI-generated answer.
        
Question: {question}
Ground Truth Answer: {expected}
Generated Answer: {generated}

Task: Determine if the Generated Answer captures the core meaning of the Ground Truth Answer. 
- Ignore minor differences in phrasing, punctuation, or casing.
- If the generated answer adds extra correct context, it is a MATCH.
- If the generated answer contradicts the ground truth or misses key facts, it is NO MATCH.

Respond ONLY with the word "YES" if it matches, or "NO" if it does not. Do not provide explanations.
"""
        
        try:
            response, _ = self.generate_answer_gemini(
                prompt=prompt, 
                model=model, 
                max_tokens=10,  
                temperature=0.0 
            )
            
            # Parse the result
            clean_response = response.strip().upper()
            is_match = "YES" in clean_response
            
            logger.info(f"LLM Judge result: {clean_response} ({'Match' if is_match else 'No Match'})")
            return is_match
            
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            return False