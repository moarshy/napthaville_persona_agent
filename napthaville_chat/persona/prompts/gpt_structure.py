import json
import time
from typing import Optional, Any, Dict, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import instructor

load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

def temp_sleep(seconds: float = 0.1) -> None:
    """Add a small delay between API calls."""
    time.sleep(seconds)

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def chat_completion_request(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> str:
    """
    Make a chat completion request to OpenAI API with retry logic.
    
    Args:
        prompt: The input prompt
        model: The model to use (default: gpt-4o-mini)
        temperature: Controls randomness (0-1)
    
    Returns:
        The generated response text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        raise

def get_embedding(
    text: str,
    model: str = "text-embedding-ada-002"
) -> List[float]:
    """
    Get embeddings for input text.
    
    Args:
        text: Input text to embed
        model: Embedding model to use
    
    Returns:
        List of embedding values
    """
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    
    try:
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise

# Convenience functions for different models
def gpt4_request(prompt: str) -> str:
    """Wrapper for GPT-4 requests."""
    return chat_completion_request(prompt, model="gpt-4")

def gpt35_request(prompt: str) -> str:
    """Wrapper for GPT-3.5 requests."""
    return chat_completion_request(prompt, model="gpt-3.5-turbo")

# Example usage
if __name__ == '__main__':
    def validate_response(response: str, **kwargs) -> bool:
        """Example validation function."""
        if len(response.strip()) <= 1:
            return False
        if len(response.strip().split()) > 1:
            return False
        return True

    def cleanup_response(response: str, **kwargs) -> str:
        """Example cleanup function."""
        return response.strip()