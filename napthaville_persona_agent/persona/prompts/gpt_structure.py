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

def generate_prompt(curr_input, prompt_lib_file): 
    """
    Takes in the current input (e.g. comment that you want to classify) and 
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this 
    function replaces this substr with the actual curr_input to produce the 
    final prompt that will be sent to the GPT3 server. 
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the prompt file. 
    RETURNS: 
      a str prompt that will be sent to OpenAI's GPT server.  
    """
    if type(curr_input) == type("string"): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    try:
        with open(prompt_lib_file, "r", encoding='utf-8') as f:
            prompt = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return ""

    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    
    return prompt.strip()

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def chat_completion_request(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
) -> str:
    """
    Make a chat completion request to OpenAI API with retry logic.
    
    Args:
        prompt: The input prompt
        model: The model to use (default: gpt-3.5-turbo)
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

def safe_generate_response(
    prompt: str,
    example_output: str,
    special_instruction: str,
    model: str = "gpt-3.5-turbo",
    repeat: int = 3,
    fail_safe_response: str = "error",
    func_validate: Optional[callable] = None,
    func_clean_up: Optional[callable] = None,
    verbose: bool = False
) -> Any:
    """
    Safely generate a response with validation and cleanup.
    
    Args:
        prompt: The input prompt
        example_output: Example of expected output format
        special_instruction: Additional instructions for output formatting
        model: The model to use
        repeat: Number of retry attempts
        fail_safe_response: Default response on failure
        func_validate: Validation function
        func_clean_up: Cleanup function
        verbose: Whether to print debug information
    
    Returns:
        Processed and validated response
    """
    formatted_prompt = f'"""\n{prompt}\n"""\n'
    formatted_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    formatted_prompt += "Example output json:\n"
    formatted_prompt += f'{{"output": "{str(example_output)}"}}'

    if verbose:
        print("PROMPT:", formatted_prompt)

    for attempt in range(repeat):
        try:
            response = chat_completion_request(formatted_prompt, model=model)
            # Find the last valid JSON object in the response
            end_index = response.rfind('}') + 1
            response_json = json.loads(response[:end_index])
            cleaned_response = response_json["output"]

            if func_validate and func_validate(cleaned_response, prompt=prompt):
                return func_clean_up(cleaned_response, prompt=prompt) if func_clean_up else cleaned_response

            if verbose:
                print(f"Attempt {attempt + 1} failed validation")
                print(f"Response: {cleaned_response}")

        except Exception as e:
            if verbose:
                print(f"Error on attempt {attempt + 1}: {str(e)}")

    return fail_safe_response

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

def instructor_request(prompt: str, response_model, model: str = "gpt-4o-mini") -> str:
    """Wrapper for instructor requests."""

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

    # Test the safe_generate_response function
    test_prompt = "Generate a single word response for: What color is the sky?"
    result = safe_generate_response(
        prompt=test_prompt,
        example_output="blue",
        special_instruction="Respond with a single word only.",
        func_validate=validate_response,
        func_clean_up=cleanup_response,
        verbose=True
    )
    print(f"Final result: {result}")