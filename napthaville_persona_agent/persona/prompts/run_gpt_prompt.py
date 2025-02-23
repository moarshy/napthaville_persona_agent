import random
import string
import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
from napthaville_persona_agent.persona.prompts.gpt_structure import safe_generate_response, generate_prompt
from napthaville_persona_agent.persona.prompts.print_prompt import PromptLogger

file_path = Path(__file__).parent

logger = PromptLogger(log_dir="prompt_logs")

# Type definitions
PromptResponse = Tuple[Any, List[Any]]  # (output, [output, prompt, params, input, fallback])

class TimeOfDay(Enum):
    """Represents different parts of the day for wake/sleep scheduling."""
    EARLY_MORNING = "early_morning"
    MORNING = "morning" 
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    LATE_NIGHT = "late_night"

@dataclass
class PromptConfig:
    """Configuration for a GPT prompt execution."""
    template_path: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop_tokens: Optional[List[str]] = None

def get_random_alphanumeric(min_length: int = 6, max_length: int = 6) -> str:
    """
    Generate a random alphanumeric string between min_length and max_length.
    
    Args:
        min_length: Minimum length of the generated string
        max_length: Maximum length of the generated string
        
    Returns:
        A random string containing letters and numbers
        
    Raises:
        ValueError: If min_length > max_length or if either is negative
    """
    if min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length")
    if min_length < 0 or max_length < 0:
        raise ValueError("Length parameters cannot be negative")
        
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def run_gpt_prompt_wake_up_hour(
    persona: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Union[PromptResponse, Dict[str, Any]]:
    """
    Determines the wake-up hour for a persona based on their traits and lifestyle.
    
    Args:
        persona: Persona object containing relevant attributes
        test_input: Optional test inputs for debugging/testing
        verbose: Whether to print detailed debug information
        
    Returns:
        For normal operation: A tuple containing (wake_up_hour, prompt_details)
        For testing: A dict with wake_up_hour and helper functions
    """
    def create_prompt_input(persona: Any, test_input: Optional[List[str]] = None) -> List[str]:
        if test_input:
            return test_input
        return [
            persona.scratch.get_str_iss(),
            persona.scratch.get_str_lifestyle(),
            persona.scratch.get_str_firstname()
        ]

    def clean_up_response(response: str, prompt: str = "") -> int:
        try:
            response = response.strip().lower()
            if ":" in response:
                hour = int(response.split(":")[0])
            else:
                hour = int(response.split("am")[0])
            
            if not (1 <= hour <= 12):
                raise ValueError(f"Invalid hour: {hour}")
            return hour
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse wake-up hour from: {response}") from e

    def validate_response(response: str, prompt: str = "") -> bool:
        try:
            clean_up_response(response, prompt)
            return True
        except ValueError:
            return False

    def get_fallback() -> int:
        return 8  # 8am is a reasonable default wake-up time

    # For testing, return the helper functions
    if test_input == "TEST_MODE":
        return {
            "clean_up_response": clean_up_response,
            "validate_response": validate_response
        }

    # Normal operation
    prompt_template = file_path / "v2/wake_up_hour_v1.txt"
    prompt_input = create_prompt_input(persona, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fallback = get_fallback()

    output = safe_generate_response(
        prompt=prompt,
        example_output="7am",
        special_instruction="Respond with only the wake up hour in AM format (e.g. '7am' or '7:00am')",
        func_validate=validate_response,
        func_clean_up=clean_up_response,
        repeat=5,
        fail_safe_response=fallback,
        verbose=verbose
    )

    if verbose:
        logger.print_prompt_run(
            prompt_template=str(prompt_template),
            persona=persona,
            prompt_input=prompt_input,
            prompt=prompt,
            output=output
        )

    return output, [output, prompt, None, prompt_input, fallback]

def run_gpt_prompt_daily_plan(
    persona: Any,
    wake_up_hour: int,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[List[str], List[Any]]:
    """
    Generate a daily plan for a persona, starting from their wake-up hour.
    
    Args:
        persona: The Persona class instance
        wake_up_hour: Hour when the persona wakes up (24-hour format)
        test_input: Optional test inputs for debugging/testing
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - List of daily activities as strings
        - List containing [output, prompt, params, prompt_input, fallback]
    """
    def create_prompt_input(
        persona: Any,
        wake_up_hour: int,
        test_input: Optional[List[str]] = None
    ) -> List[str]:
        """Create the input list for the prompt template."""
        if test_input:
            return test_input
            
        return [
            persona.scratch.get_str_iss(),
            persona.scratch.get_str_lifestyle(),
            persona.scratch.get_str_curr_date_str(),
            persona.scratch.get_str_firstname(),
            f"{str(wake_up_hour)}:00 am"
        ]

    def get_fallback() -> List[str]:
        """Return a sensible fallback daily schedule."""
        return [
            'wake up and complete the morning routine at 6:00 am',
            'eat breakfast at 7:00 am',
            'read a book from 8:00 am to 12:00 pm',
            'have lunch at 12:00 pm',
            'take a nap from 1:00 pm to 4:00 pm',
            'relax and watch TV from 7:00 pm to 8:00 pm',
            'go to bed at 11:00 pm'
        ]

    prompt_template = file_path / "v2/daily_planning_v6.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    example = [
        "exercise from 6:00 am to 7:00 am",
        "eat breakfast at 7:30 am",
        "work from 8:00 am to 12:00 pm"
    ]
    
    output = safe_generate_response(
        prompt=prompt,
        example_output=example,
        special_instruction=(
            "Return a list of daily activities with specific times. "
            "Each activity must include either 'at' or 'from/to' with times in AM/PM format. "
            "Activities should span the full day and be reasonably spaced."
        ),
        func_validate=validate_response,
        func_clean_up=clean_up_response,
        repeat=5,
        fail_safe_response=get_fallback(),
        verbose=verbose
    )

    # Prepend wake-up activity
    output = [f"wake up and complete the morning routine at {wake_up_hour}:00 am"] + output

    if verbose:
        logger.print_prompt_run(
            prompt_template=str(prompt_template),
            persona=persona,
            prompt_input=prompt_input,
            prompt=prompt,
            output=output
        )

    return output, [output, prompt, None, prompt_input, get_fallback()]

if __name__ == "__main__":
    def test_get_random_alphanumeric():
        """Test get_random_alphanumeric function with various inputs."""
        print("\nTesting get_random_alphanumeric:")
        
        # Test default parameters
        result = get_random_alphanumeric()
        print(f"Default params result: {result}")
        assert len(result) == 6, f"Expected length 6, got {len(result)}"
        assert result.isalnum(), "Result should be alphanumeric"
        
        # Test custom lengths
        result = get_random_alphanumeric(3, 3)
        print(f"Fixed length 3 result: {result}")
        assert len(result) == 3, f"Expected length 3, got {len(result)}"
        
        # Test range of lengths
        lengths = set()
        for _ in range(100):
            result = get_random_alphanumeric(4, 8)
            lengths.add(len(result))
            assert 4 <= len(result) <= 8, f"Length {len(result)} outside range 4-8"
        print(f"Length distribution for range 4-8: {sorted(lengths)}")
        
        # Test error cases
        try:
            get_random_alphanumeric(5, 3)
            assert False, "Should raise ValueError for min > max"
        except ValueError as e:
            print("Correctly caught min > max error")
            
        try:
            get_random_alphanumeric(-1, 5)
            assert False, "Should raise ValueError for negative min"
        except ValueError as e:
            print("Correctly caught negative min error")
        
        print("All get_random_alphanumeric tests passed!")

    def test_run_gpt_prompt_wake_up_hour():
        """Test run_gpt_prompt_wake_up_hour function with both unit tests and LLM integration."""
        print("\nTesting run_gpt_prompt_wake_up_hour:")
        
        class MockPersona:
            class Scratch:
                def get_str_iss(self):
                    return "Test persona traits"
                def get_str_lifestyle(self):
                    return "Early riser, health-conscious"
                def get_str_firstname(self):
                    return "TestPerson"
            def __init__(self):
                self.scratch = self.Scratch()
        
        # Part 1: Unit Tests for Helper Functions
        print("\nPart 1: Testing helper functions:")
        test_persona = MockPersona()
        helpers = run_gpt_prompt_wake_up_hour(test_persona, test_input="TEST_MODE")
        clean_up_response = helpers["clean_up_response"]
        validate_response = helpers["validate_response"]
        
        # Test response parsing
        test_cases = [
            ("7am", 7),
            ("7:00am", 7),
            ("10am", 10),
            ("6:30am", 6),
        ]
        
        print("Testing response parsing:")
        for input_str, expected in test_cases:
            cleaned = clean_up_response(input_str)
            assert cleaned == expected, f"Failed parsing {input_str}"
            print(f"Successfully parsed '{input_str}' to {cleaned}")
        
        # Test validation
        print("\nTesting validation:")
        valid_cases = ["7am", "7:00am", "11am", "6:30am"]
        invalid_cases = ["13am", "0am", "7pm", "abc", ""]
        
        for case in valid_cases:
            assert validate_response(case), f"Should accept valid input: {case}"
            print(f"Correctly validated {case}")
            
        for case in invalid_cases:
            assert not validate_response(case), f"Should reject invalid input: {case}"
            print(f"Correctly rejected {case}")
        
        # Part 2: Integration Tests with LLM
        print("\nPart 2: Testing LLM integration:")
        
        # Test case 1: Early riser
        early_riser = MockPersona()
        early_riser.scratch.get_str_lifestyle = lambda: "Early riser who loves to start the day at dawn, very health-conscious"
        print("\nTesting early riser persona...")
        wake_hour, details = run_gpt_prompt_wake_up_hour(early_riser, verbose=False)
        print(f"Early riser wake hour: {wake_hour}")
        assert isinstance(wake_hour, int), "Wake hour should be an integer"
        assert 4 <= wake_hour <= 7, f"Early riser should wake between 4-7am, got {wake_hour}"
        
        # Test case 2: Night owl
        night_owl = MockPersona()
        night_owl.scratch.get_str_lifestyle = lambda: "Night owl who stays up late working on creative projects"
        print("\nTesting night owl persona...")
        wake_hour, details = run_gpt_prompt_wake_up_hour(night_owl, verbose=False)
        print(f"Night owl wake hour: {wake_hour}")
        assert isinstance(wake_hour, int), "Wake hour should be an integer"
        assert 8 <= wake_hour <= 11, f"Night owl should wake between 8-11am, got {wake_hour}"
        
        print("All run_gpt_prompt_wake_up_hour tests passed!")

    def test_daily_plan():
        """Test the daily plan generation functionality."""
        print("\nTesting daily plan generation:")

        class MockPersona:
            class Scratch:
                def get_str_iss(self):
                    return "Test persona traits"
                def get_str_lifestyle(self):
                    return "Early riser, health-conscious"
                def get_str_firstname(self):
                    return "TestPerson"
                def get_str_curr_date_str(self):
                    return datetime.datetime.now().strftime("%A %B %d")
            def __init__(self):
                self.scratch = self.Scratch()

        # Part 1: Unit Tests
        print("\nPart 1: Testing helper functions:")
        
        # Get test helpers
        test_persona = MockPersona()
        helpers = run_gpt_prompt_daily_plan(test_persona, 7, test_input="TEST_MODE")
        clean_up_response = helpers["clean_up_response"]
        validate_response = helpers["validate_response"]

        # Test response cleaning
        test_cases = [
            (
                "1) eat breakfast at 7:00 am 2) work from 8:00 am to 12:00 pm 3",
                ["eat breakfast at 7:00 am", "work from 8:00 am to 12:00 pm"]
            ),
            (
                "1) morning routine at 6:00 am. 2) exercise at 7:00 am, 3) rest at 8:00 am",
                ["morning routine at 6:00 am", "exercise at 7:00 am", "rest at 8:00 am"]
            )
        ]

        print("Testing response parsing:")
        for input_str, expected in test_cases:
            result = clean_up_response(input_str)
            assert result == expected, f"Failed parsing: {input_str}"
            print(f"Successfully parsed activity list")

        # Test validation
        print("\nTesting validation:")
        valid_cases = [
            "1) breakfast at 7:00 am 2) work from 9:00 am to 5:00 pm",
            "1) exercise at 8:00 am, 2) rest at 9:00 am"
        ]
        invalid_cases = [
            "1) just doing stuff 2) more stuff",
            "",
            "1) breakfast 7:00 2) lunch 12:00"
        ]

        for case in valid_cases:
            assert validate_response(case), f"Should accept valid input: {case}"
            print(f"Correctly validated schedule")

        for case in invalid_cases:
            assert not validate_response(case), f"Should reject invalid input: {case}"
            print(f"Correctly rejected invalid schedule")

        # Part 2: Integration Tests
        print("\nPart 2: Testing LLM integration:")

        # Test case 1: Early riser
        early_riser = MockPersona()
        early_riser.scratch.get_str_lifestyle = lambda: "Early riser who loves exercise and productivity"
        print("\nTesting early riser persona...")
        
        schedule, details = run_gpt_prompt_daily_plan(early_riser, 5, verbose=True)
        print(f"Generated schedule with {len(schedule)} activities:")
        for activity in schedule:
            print(f"  - {activity}")
            
        assert len(schedule) >= 5, "Should have at least 5 activities"
        assert "wake up" in schedule[0].lower(), "First activity should be waking up"
        assert all(" at " in act or " from " in act for act in schedule), "All activities should have times"

        # Test case 2: Night owl
        night_owl = MockPersona()
        night_owl.scratch.get_str_lifestyle = lambda: "Night owl who works in creative industries"
        print("\nTesting night owl persona...")
        
        schedule, details = run_gpt_prompt_daily_plan(night_owl, 9, verbose=True)
        print(f"Generated schedule with {len(schedule)} activities:")
        for activity in schedule:
            print(f"  - {activity}")
            
        assert len(schedule) >= 5, "Should have at least 5 activities"
        assert "wake up" in schedule[0].lower(), "First activity should be waking up"
        assert all(" at " in act or " from " in act for act in schedule), "All activities should have times"

        print("All daily plan tests passed!")


    # Run the tests
    test_get_random_alphanumeric()
    test_run_gpt_prompt_wake_up_hour()
    test_daily_plan()