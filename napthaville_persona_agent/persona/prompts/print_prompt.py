from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

@dataclass
class Persona:
    """Basic persona information structure."""
    name: str
    traits: Dict[str, Any] = None
    
    def __str__(self) -> str:
        base = f"Persona: {self.name}"
        if self.traits:
            traits_str = "\n".join(f"  {k}: {v}" for k, v in self.traits.items())
            return f"{base}\nTraits:\n{traits_str}"
        return base

class PromptLogger:
    """Handles logging and printing of prompts and their context."""
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize the prompt logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (default: INFO)
            console_output: Whether to also print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("PromptLogger")
        self.logger.setLevel(log_level)
        
        # File handler
        log_file = self.log_dir / f"prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(message)s')  # Simpler format for console
            )
            self.logger.addHandler(console_handler)

    def format_section(
        self,
        title: str,
        content: Any,
        width: int = 70
    ) -> str:
        """Format a section with title and content."""
        separator = "-" * width
        if content is None:
            content = "None"
        elif isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        else:
            content = str(content)
            
        return f"\n{title}\n{separator}\n{content}\n"

    def print_prompt_run(
        self,
        prompt_template: Optional[str] = None,
        persona: Optional[Persona] = None,
        gpt_params: Optional[Dict[str, Any]] = None,
        prompt_input: Optional[Union[str, Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Print and log a complete prompt run with all context.
        
        Args:
            prompt_template: Name or identifier of the prompt template
            persona: Persona object containing agent details
            gpt_params: Parameters used for the GPT API call
            prompt_input: Raw input provided to the prompt template
            prompt: Final formatted prompt sent to GPT
            output: Response received from GPT
            metadata: Additional metadata to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build the complete log message
        sections = [
            ("=== Prompt Run Log ===", timestamp),
            ("Template", prompt_template),
            ("Persona", persona),
            ("GPT Parameters", gpt_params),
            ("Prompt Input", prompt_input),
            ("Final Prompt", prompt),
            ("Output", output),
            ("Metadata", metadata)
        ]
        
        log_message = "\n"
        for title, content in sections:
            if content is not None:  # Only include non-None sections
                log_message += self.format_section(title, content)
        
        log_message += "\n=== End of Prompt Run ===\n\n"
        
        # Log the message
        self.logger.info(log_message)

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error with context."""
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f"\nContext: {json.dumps(context, indent=2)}"
        self.logger.error(error_msg)

# Example usage
if __name__ == "__main__":
    # Initialize the logger
    logger = PromptLogger(log_dir="prompt_logs")
    
    # Create a sample persona
    test_persona = Persona(
        name="TestBot",
        traits={
            "role": "assistant",
            "personality": "helpful and direct",
            "expertise": ["python", "prompt engineering"]
        }
    )
    
    # Sample GPT parameters
    gpt_params = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Test the logger
    try:
        logger.print_prompt_run(
            prompt_template="test_template",
            persona=test_persona,
            gpt_params=gpt_params,
            prompt_input="What is the capital of France?",
            prompt="Please tell me: What is the capital of France?",
            output="The capital of France is Paris.",
            metadata={"session_id": "test-123", "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.log_error(e, {"step": "prompt_run", "template": "test_template"})