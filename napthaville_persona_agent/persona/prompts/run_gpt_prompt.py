import random
import string
import datetime
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from pydantic import BaseModel
from pathlib import Path
from napthaville_persona_agent.persona.prompts.gpt_structure import chat_completion_request
from napthaville_persona_agent.persona.prompts.print_prompt import PromptLogger
from napthaville_persona_agent.persona.prompts.prompts import (
    wake_up_template, daily_planning_template, hourly_schedule_template, task_decomp_template,
    action_sector_template, action_arena_template, action_game_object_template,
    pronunciation_template, event_triple_template, act_obj_desc_template, act_obj_event_triple_template,
    new_decomp_schedule_template, decide_to_talk_template, decide_to_react_template, create_conversation_template,
    summarize_conversation_template, extract_keywords_template, keyword_to_thoughts_template,
    convo_to_thoughts_template, event_poignancy_template, thought_poignancy_template
)

file_path = Path(__file__).parent

logger = PromptLogger(log_dir="prompt_logs")

class PromptConfig(BaseModel):
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
    persona: Any
) -> Tuple[List[str], List[Any]]:
    """
    Determines the wake-up hour for a persona based on their traits and lifestyle.
    
    Args:
        persona: Persona object containing relevant attributes

    Returns:
        For normal operation: A tuple containing (wake_up_hour, prompt_details)
    """
    def get_fallback() -> int:
        return 8
    
    def validate_response(response: str) -> bool:
        try:
            hour = int(response)
            return 1 <= hour <= 24
        except ValueError:
            return False

    # Normal operation
    prompt = wake_up_template.format(
        identity_set=persona.scratch.get_str_iss(),
        lifestyle=persona.scratch.get_str_lifestyle(),
        names=persona.scratch.get_str_firstname()
    )
    fallback = get_fallback()

    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            response = fallback
    except Exception as e:
        print(f"Error in chat completion: {e}")
        response = fallback

    return response, [response, prompt, None, prompt, fallback]

def run_gpt_prompt_daily_plan(
    persona: Any,
    wake_up_hour: int
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

    prompt = daily_planning_template.format(
        commonset=persona.scratch.get_str_iss(),
        lifestyle=persona.scratch.get_str_lifestyle(),
        datetime=persona.scratch.get_str_curr_date_str(),
        names=persona.scratch.get_str_firstname(),
        wake_up_hour=f"{str(wake_up_hour)}:00 am"
    )
    try:
        response = chat_completion_request(prompt)
        output = response.split("```json")[1].split("```")[0]
        output = json.loads(output)
    except Exception as e:
        print(f"Error in chat completion: {e}")
        output = get_fallback()

    # Prepend wake-up activity
    output = [f"wake up and complete the morning routine at {wake_up_hour}:00 am"] + output

    return output, [output, prompt, None, prompt, get_fallback()]

def run_gpt_prompt_generate_hourly_schedule(
    persona,
    curr_hour_str,
    p_f_ds_hourly_org,
    hour_str,
    intermission2=None
) -> Tuple[str, List[Any]]:
    """
    Generate an hourly schedule for a persona.
    
    Args:
        persona: The Persona class instance
        curr_hour_str: Current hour string
        p_f_ds_hourly_org: Previous hourly schedule
        hour_str: List of hour strings
        intermission2: Optional additional intermission text
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Generated activity string
        - List containing [output, prompt, params, prompt_input, fallback]
    """
    def create_prompt_input(
        persona,
        curr_hour_str,
        p_f_ds_hourly_org,
        hour_str,
        intermission2=None,
        test_input=None
    ) -> Dict[str, str]:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Create schedule format
        schedule_format = "\n".join(
            f"[{persona.scratch.get_str_curr_date_str()} -- {hour}] Activity: [Fill in]"
            for hour in hour_str
        )

        # Create intermission string
        intermission_str = (
            f"Here the originally intended hourly breakdown of "
            f"{persona.scratch.get_str_firstname()}'s schedule today: " +
            ", ".join(f"{i+1}) {act}" for i, act in enumerate(persona.scratch.daily_req))
        )

        # Create prior schedule if it exists
        prior_schedule = ""
        if p_f_ds_hourly_org:
            prior_schedule = "\n" + "\n".join(
                f"[(ID:{get_random_alphanumeric()}) "
                f"{persona.scratch.get_str_curr_date_str()} -- "
                f"{hour_str[i]}] Activity: {persona.scratch.get_str_firstname()} "
                f"is {activity}"
                for i, activity in enumerate(p_f_ds_hourly_org)
            )

        # Create prompt ending
        prompt_ending = (
            f"[(ID:{get_random_alphanumeric()}) "
            f"{persona.scratch.get_str_curr_date_str()} -- {curr_hour_str}] "
            f"Activity: {persona.scratch.get_str_firstname()} is"
        )

        return {
            'schedule_format': schedule_format,
            'commonset': persona.scratch.get_str_iss(),
            'prior_schedule': prior_schedule + "\n",
            'intermission_str': intermission_str,
            'intermission2': f"\n{intermission2}" if intermission2 else "",
            'prompt_ending': prompt_ending
        }

    def clean_up_response(response: str) -> str:
        """Clean up the GPT response."""
        response = response.strip()
        if response.endswith("."):
            response = response[:-1]
        return response

    def validate_response(response: str) -> bool:
        """Validate the GPT response."""
        try:
            clean_up_response(response)
            return True
        except:
            return False

    def get_fallback() -> str:
        """Return fallback response."""
        return "asleep"

    # Prepare prompt and parameters
    gpt_param = {
        "engine": "text-davinci-003",
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ["\n"]
    }

    # Generate prompt using new template format
    prompt_inputs = create_prompt_input(
        persona,
        curr_hour_str,
        p_f_ds_hourly_org,
        hour_str,
        intermission2
    )
    
    prompt = hourly_schedule_template.format(**prompt_inputs)
    fallback = get_fallback()

    try:
        response = chat_completion_request(prompt)
        print(f"Response: {response}")
        if not validate_response(response):
            response = fallback
    except Exception as e:
        print(f"Error in chat completion: {e}")
        response = fallback

    return response, [response, prompt, gpt_param, prompt_inputs, fallback]

def run_gpt_prompt_task_decomp(
    persona: Any,
    task: str,
    duration: int,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[List[List[str]], List[Any]]:
    """
    Decompose a task into subtasks with durations.
    
    Args:
        persona: The Persona class instance
        task: Main task to decompose
        duration: Total duration in minutes
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - List of [task_description, duration] pairs
        - List containing [output, prompt, params, prompt_input, fallback]
    """
    def create_prompt_input(
        persona: Any,
        task: str,
        duration: int,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Get current and surrounding schedule
        curr_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        indices = [i for i in range(max(0, curr_index-1), min(curr_index+3, 
                  len(persona.scratch.f_daily_schedule_hourly_org)))]
        
        # Build schedule summary
        schedule_desc = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. From '
        curr_time_range = ""
        
        for idx in indices:
            if idx < len(persona.scratch.f_daily_schedule_hourly_org):
                # Calculate time range
                start_min = sum(persona.scratch.f_daily_schedule_hourly_org[i][1] 
                              for i in range(idx))
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[idx][1]
                
                start_time = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                start_time += datetime.timedelta(minutes=start_min)
                end_time = start_time + datetime.timedelta(minutes=end_min)
                
                start_str = start_time.strftime("%H:%M%p")
                end_str = end_time.strftime("%H:%M%p")
                
                schedule_desc += (f"{start_str} ~ {end_str}, {persona.name} is planning "
                                f"on {persona.scratch.f_daily_schedule_hourly_org[idx][0]}, ")
                
                if curr_index + 1 == idx:
                    curr_time_range = f'{start_str} ~ {end_str}'
        
        schedule_desc = schedule_desc[:-2] + "."

        return {
            'commonset': persona.scratch.get_str_iss(),
            'schedule_desc': schedule_desc,
            'name': persona.scratch.get_str_firstname(),
            'name_repeat': persona.scratch.get_str_firstname(),
            'task': task,
            'time_range': curr_time_range,
            'duration': str(duration),
            'persona_name': persona.scratch.get_str_firstname()
        }

    def clean_up_response(response: str, duration: int) -> List[List[Any]]:
        """Clean and validate the response."""
        # Split into lines and parse tasks
        tasks = []
        for line in response.strip().split("\n"):
            if not line.strip():
                continue
            
            parts = line.split("(duration in minutes:")
            if len(parts) != 2:
                continue
                
            task = parts[0].split(") ")[-1].strip()
            if task.endswith("."):
                task = task[:-1]
            
            try:
                mins = int(parts[1].split(",")[0].strip())
                tasks.append([task, mins])
            except (ValueError, IndexError):
                continue
        
        # Adjust durations to match total
        total_mins = sum(task[1] for task in tasks)
        if total_mins > duration:
            while sum(task[1] for task in tasks) > duration:
                if len(tasks) > 1:
                    tasks[-1][1] -= 5
                if tasks[-1][1] <= 0:
                    tasks.pop()
        elif total_mins < duration:
            remaining = duration - total_mins
            tasks[-1][1] += remaining
            
        return tasks

    def validate_response(response: str) -> bool:
        """Validate the GPT response."""
        try:
            lines = response.strip().split("\n")
            return all("duration in minutes:" in line for line in lines if line.strip())
        except:
            return False

    def get_fallback() -> List[str]:
        """Return fallback tasks."""
        return [["working on task", duration]]

    # Generate prompt
    prompt_inputs = create_prompt_input(persona, task, duration, test_input)
    prompt = task_decomp_template.format(**prompt_inputs)
    
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            tasks = fallback
        else:
            tasks = clean_up_response(response, duration)
    except Exception as e:
        print(f"Error in task decomposition: {e}")
        tasks = fallback
    
    # Format final output
    output = [[f"{task} ({subtask})", mins] for subtask, mins in tasks]
    
    if verbose:
        print(f"Generated prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Final tasks: {output}")
    
    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_action_sector(
    action_description: str,
    persona: Any,
    maze: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Determine appropriate sector for a given action.
    
    Args:
        action_description: Description of the action
        persona: The Persona class instance
        maze: The Maze class instance
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Selected sector name
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        action_description: str,
        persona: Any,
        maze: Any,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Get current world and sector info
        act_world = maze.access_tile(persona.scratch.curr_tile)['world']
        current_sector = maze.access_tile(persona.scratch.curr_tile)['sector']
        
        # Parse action description
        action = action_description
        if "(" in action_description:
            action = action_description.split("(")[-1][:-1]

        # Get accessible sectors with proper filtering
        accessible_sectors = persona.s_mem.get_str_accessible_sectors(act_world)
        curr_sectors = accessible_sectors.split(", ")
        fin_accessible_sectors = [
            sector for sector in curr_sectors
            if not ("'s house" in sector and persona.scratch.last_name not in sector)
        ]
        accessible_sectors_str = ", ".join(fin_accessible_sectors)

        # Get current sector's areas
        current_areas = persona.s_mem.get_str_accessible_sector_arenas(
            f"{act_world}:{current_sector}"
        )

        return {
            'name': persona.scratch.get_str_name(),
            'living_sector': persona.scratch.living_area.split(":")[1],
            'name_repeat': persona.scratch.get_str_name(),
            'current_sector': current_sector,
            'current_areas': current_areas,
            'available_sectors': accessible_sectors_str,
            'action': action,
            'name2': persona.scratch.get_str_name()
        }

    def clean_up_response(response: str) -> str:
        """Clean up the GPT response."""
        try:
            response = response.strip()
            if "Answer:" in response:
                response = response.split("Answer:")[-1]
            return response.strip()
        except:
            return ""

    def validate_response(response: str, accessible_sectors: List[str]) -> bool:
        """Validate the GPT response."""
        if not response or len(response.strip()) < 1:
            return False
        
        cleaned = clean_up_response(response)
        return cleaned in accessible_sectors

    def get_fallback(persona: Any) -> str:
        """Return fallback sector (persona's living area)."""
        return persona.scratch.living_area.split(":")[1]

    # Generate prompt
    prompt_inputs = create_prompt_input(
        action_description, persona, maze, test_input
    )
    prompt = action_sector_template.format(**prompt_inputs)
    
    fallback = get_fallback(persona)
    
    try:
        # Get accessible sectors for validation
        current_world = maze.access_tile(persona.scratch.curr_tile)['world']
        accessible_sectors = [
            s.strip() for s in 
            persona.s_mem.get_str_accessible_sectors(current_world).split(",")
        ]
        
        response = chat_completion_request(prompt)
        if not validate_response(response, accessible_sectors):
            response = fallback
        else:
            response = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in sector selection: {e}")
        response = fallback

    if verbose:
        print(f"Action: {action_description}")
        print(f"Selected sector: {response}")
        print(f"Available sectors: {accessible_sectors}")

    return response, [response, prompt, prompt_inputs, fallback]

def run_gpt_prompt_action_arena(
    action_description: str,
    persona: Any,
    maze: Any, 
    act_world: str,
    act_sector: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Determine appropriate arena for a given action within a sector.
    
    Args:
        action_description: Description of the action
        persona: The Persona class instance
        maze: The Maze class instance
        act_world: The current world
        act_sector: The current sector
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Selected arena name
        - List containing [output, prompt, params, prompt_input, fallback]
    """
    def create_prompt_input(
        action_description: str,
        persona: Any,
        maze: Any,
        act_world: str,
        act_sector: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Parse action description
        action_desc = action_description
        action = action_description
        if "(" in action_description:
            action_desc = action_description.split("(")[0].strip()
            action = action_description.split("(")[-1][:-1]

        # Get accessible arenas
        sector_key = f"{act_world}:{act_sector}"
        accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(sector_key)
        curr_arenas = accessible_arena_str.split(", ")
        
        # Filter out inaccessible rooms
        fin_accessible_arenas = [
            arena for arena in curr_arenas
            if not ("'s room" in arena and persona.scratch.last_name not in arena)
        ]
        accessible_arena_str = ", ".join(fin_accessible_arenas)

        current_arena = maze.access_tile(persona.scratch.curr_tile)["arena"]
        
        return {
            'name': persona.scratch.get_str_name(),
            'current_arena': current_arena,
            'target_sector': act_sector,
            'name2': persona.scratch.get_str_name(),
            'target_sector2': act_sector,
            'available_areas': accessible_arena_str,
            'action': action,
            'name3': persona.scratch.get_str_name(),
            'target_sector3': act_sector
        }

    def clean_up_response(response: str) -> str:
        """Clean up the GPT response."""
        try:
            response = response.strip()
            if "Answer:" in response:
                response = response.split("Answer:")[-1]
            return response.strip()
        except:
            return ""

    def validate_response(response: str, accessible_arenas: List[str]) -> bool:
        """Validate the GPT response."""
        if not response or len(response.strip()) < 1:
            return False
        
        cleaned = clean_up_response(response)
        return cleaned in accessible_arenas

    def get_fallback() -> str:
        """Return fallback arena."""
        return "kitchen"

    # Generate prompt
    prompt_inputs = create_prompt_input(
        action_description, persona, maze, act_world, act_sector, test_input
    )
    prompt = action_arena_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        # Get accessible arenas for validation
        arena_key = f"{act_world}:{act_sector}"
        accessible_arenas = [
            a.strip() for a in 
            persona.s_mem.get_str_accessible_sector_arenas(arena_key).split(",")
        ]
        
        response = chat_completion_request(prompt)
        if not validate_response(response, accessible_arenas):
            response = fallback
        else:
            response = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in arena selection: {e}")
        response = fallback

    if verbose:
        print(f"Action: {action_description}")
        print(f"Selected arena: {response}")
        print(f"Available arenas: {accessible_arenas}")

    return response, [response, prompt, prompt_inputs, fallback]

def run_gpt_prompt_action_game_object(
    action_description: str,
    persona: Any,
    maze: Any,
    temp_address: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Determine appropriate game object for a given action.
    
    Args:
        action_description: Description of the action
        persona: The Persona class instance
        maze: The Maze class instance
        temp_address: The current address (world:sector:arena)
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Selected game object name
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        action_description: str,
        persona: Any,
        temp_address: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Parse action description
        action = action_description
        if "(" in action_description:
            action = action_description.split("(")[-1][:-1]

        # Get available objects
        available_objects = persona.s_mem.get_str_accessible_arena_game_objects(temp_address)
        
        return {
            'action': action,
            'available_objects': available_objects
        }

    def clean_up_response(response: str) -> str:
        """Clean up the GPT response."""
        try:
            response = response.strip()
            if "Answer:" in response:
                response = response.split("Answer:")[-1]
            return response.strip()
        except:
            return ""

    def validate_response(response: str, available_objects: List[str]) -> bool:
        """Validate the GPT response."""
        if not response or len(response.strip()) < 1:
            return False
        
        cleaned = clean_up_response(response)
        return cleaned in available_objects

    def get_fallback() -> str:
        """Return fallback object."""
        return "bed"

    # Generate prompt
    prompt_inputs = create_prompt_input(
        action_description, persona, temp_address, test_input
    )
    prompt = action_game_object_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        # Get available objects for validation
        available_objects = [
            obj.strip() for obj in 
            persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")
        ]
        
        response = chat_completion_request(prompt)
        if not validate_response(response, available_objects):
            response = random.choice(available_objects)  # Use random choice as specified
        else:
            response = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in object selection: {e}")
        response = fallback

    if verbose:
        print(f"Action: {action_description}")
        print(f"Selected object: {response}")
        print(f"Available objects: {available_objects}")

    return response, [response, prompt, prompt_inputs, fallback]

def run_gpt_prompt_pronunciation(
    action_description: str,
    persona: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Convert an action description to emoji representation.
    
    Args:
        action_description: Description of the action
        persona: The Persona class instance (for consistency)
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Emoji representation
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        action_description: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Extract core action if in parentheses
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
            
        return {
            'action': action_description
        }

    def clean_up_response(response: str) -> str:
        """Clean up the GPT response to ensure valid emoji output."""
        try:
            response = response.strip()
            # Take at most first two emojis (each emoji is typically 2 chars)
            if len(response) > 4:
                response = response[:4]
            return response
        except:
            return ""

    def validate_response(response: str) -> bool:
        """Validate the emoji response."""
        if not response or len(response.strip()) < 1:
            return False
        
        cleaned = clean_up_response(response)
        # Check if response contains at least one emoji character
        return any(ord(char) > 127 for char in cleaned)

    def get_fallback() -> str:
        """Return fallback emoji."""
        return "😊"

    # Generate prompt
    prompt_inputs = create_prompt_input(action_description, test_input)
    prompt = pronunciation_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            response = fallback
        else:
            response = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in emoji generation: {e}")
        response = fallback

    if verbose:
        print(f"Action: {action_description}")
        print(f"Generated emoji: {response}")

    return response, [response, prompt, prompt_inputs, fallback]

def run_gpt_prompt_event_triple(
    action_description: str,
    persona: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[Tuple[str, str, str], List[Any]]:
    """
    Generate event triple (subject, predicate, object) for an action.
    
    Args:
        action_description: Description of the action
        persona: The Persona class instance
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Triple of (subject, predicate, object)
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        action_description: str,
        persona: Any,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Extract core action if in parentheses
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
            
        return {
            'name': persona.name,
            'action': action_description,
            'name2': persona.name
        }

    def clean_up_response(response: str) -> List[str]:
        """Clean up the response into subject, predicate, object."""
        try:
            response = response.strip()
            # Extract content between parentheses and split by comma
            parts = [part.strip() for part in response.split(")")[0].split(",")]
            # Ensure we have exactly two parts (predicate and object)
            if len(parts) != 2:
                return []
            return parts
        except:
            return []

    def validate_response(response: str) -> bool:
        """Validate the triple response."""
        try:
            parts = clean_up_response(response)
            return len(parts) == 2 and all(p.strip() for p in parts)
        except:
            return False

    def get_fallback(persona: Any) -> Tuple[str, str, str]:
        """Return fallback triple."""
        return (persona.name, "is", "idle")

    # Generate prompt
    prompt_inputs = create_prompt_input(action_description, persona, test_input)
    prompt = event_triple_template.format(**prompt_inputs)
    
    fallback = get_fallback(persona)
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            # Convert response to triple
            parts = clean_up_response(response)
            output = (persona.name, parts[0], parts[1])
            
    except Exception as e:
        print(f"Error in triple generation: {e}")
        output = fallback

    if verbose:
        print(f"Action: {action_description}")
        print(f"Generated triple: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_act_obj_desc(
    act_game_object: str,
    act_desp: str,
    persona: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Generate a description of an object's state based on an action.
    
    Args:
        act_game_object: Name of the object being used
        act_desp: Description of the action
        persona: The Persona class instance
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Object state description
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        act_game_object: str,
        act_desp: str,
        persona: Any,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'object_name': act_game_object,
            'persona_name': persona.name,
            'action_desc': act_desp,
            'object_name2': act_game_object,
            'object_name3': act_game_object
        }

    def clean_up_response(response: str, object_name: str) -> str:
        """Clean up the object state description."""
        try:
            response = response.strip()
            # Remove trailing period if present
            if response.endswith("."):
                response = response[:-1]
            
            # If response starts with object name, return as is
            if response.lower().startswith(object_name.lower()):
                return response
                
            # If response doesn't include object name at start, prepend it
            return f"{object_name} is {response}"
            
        except:
            return ""

    def validate_response(response: str, object_name: str) -> bool:
        """Validate the object state description."""
        if not response or len(response.strip()) < 1:
            return False
        
        cleaned = clean_up_response(response, object_name)
        # Check if response forms a valid state description
        return (
            len(cleaned.split()) >= 2 and  # At least two words
            object_name.lower() in cleaned.lower() and  # Contains object name
            "is" in cleaned  # Contains state descriptor
        )

    def get_fallback(act_game_object: str) -> str:
        """Return fallback object state."""
        return f"{act_game_object} is idle"

    # Generate prompt
    prompt_inputs = create_prompt_input(
        act_game_object, act_desp, persona, test_input
    )
    prompt = act_obj_desc_template.format(**prompt_inputs)
    
    fallback = get_fallback(act_game_object)
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response, act_game_object):
            response = fallback
        else:
            response = clean_up_response(response, act_game_object)
            
    except Exception as e:
        print(f"Error in object state description: {e}")
        response = fallback

    if verbose:
        print(f"Object: {act_game_object}")
        print(f"Action: {act_desp}")
        print(f"Generated state: {response}")

    return response, [response, prompt, prompt_inputs, fallback]

def run_gpt_prompt_act_obj_event_triple(
    act_game_object: str,
    act_obj_desc: str,
    persona: Any,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[Tuple[str, str, str], List[Any]]:
    """
    Generate event triple (subject, predicate, object) for an object state.
    
    Args:
        act_game_object: Name of the object being used
        act_obj_desc: Description of the object's state
        persona: The Persona class instance (for consistency)
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Triple of (subject, predicate, object)
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        act_game_object: str,
        act_obj_desc: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'object_name': act_game_object,
            'object_state': act_obj_desc,
            'object_name2': act_game_object
        }

    def clean_up_response(response: str) -> List[str]:
        """Clean up the response into predicate and object."""
        try:
            response = response.strip()
            # Extract content between parentheses and split by comma
            parts = [part.strip() for part in response.split(")")[0].split(",")]
            # Ensure we have exactly two parts (predicate and object)
            if len(parts) != 2:
                return []
            return parts
        except:
            return []

    def validate_response(response: str) -> bool:
        """Validate the triple response."""
        try:
            parts = clean_up_response(response)
            return len(parts) == 2 and all(p.strip() for p in parts)
        except:
            return False

    def get_fallback(act_game_object: str) -> Tuple[str, str, str]:
        """Return fallback triple."""
        return (act_game_object, "is", "idle")

    # Generate prompt
    prompt_inputs = create_prompt_input(
        act_game_object, act_obj_desc, test_input
    )
    prompt = act_obj_event_triple_template.format(**prompt_inputs)
    
    fallback = get_fallback(act_game_object)
    
    fallback = get_fallback(act_game_object)
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            # Convert response to triple
            parts = clean_up_response(response)
            output = (act_game_object, parts[0], parts[1])
            
    except Exception as e:
        print(f"Error in triple generation: {e}")
        output = fallback

    if verbose:
        print(f"Object: {act_game_object}")
        print(f"State: {act_obj_desc}")
        print(f"Generated triple: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_new_decomp_schedule(
    persona: Any,
    main_act_dur: List[List[Any]],
    truncated_act_dur: List[List[Any]],
    start_time_hour: datetime.datetime,
    end_time_hour: datetime.datetime,
    inserted_act: str,
    inserted_act_dur: int,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[List[List[Any]], List[Any]]:
    """
    Generate a revised schedule with an inserted activity.
    
    Args:
        persona: The Persona class instance
        main_act_dur: Original schedule as list of [activity, duration] pairs
        truncated_act_dur: Partial schedule before insertion point
        start_time_hour: Start time of schedule
        end_time_hour: End time of schedule
        inserted_act: Activity to insert
        inserted_act_dur: Duration of inserted activity
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - List of [activity, duration] pairs for new schedule
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_time_string(dt: datetime.datetime) -> str:
        """Create consistent time string format."""
        return dt.strftime("%H:%M %p")

    def create_schedule_string(
        activities: List[List[Any]], 
        start_time: datetime.datetime,
        include_final: bool = True
    ) -> str:
        """Create formatted schedule string."""
        result = ""
        current_time = start_time
        
        for i, (activity, duration) in enumerate(activities):
            end_time = current_time + datetime.timedelta(minutes=int(duration))
            result += f'{current_time.strftime("%H:%M")} ~ {end_time.strftime("%H:%M")} -- {activity}\n'
            current_time = end_time
            
        if include_final and activities:
            final_time = current_time + datetime.timedelta(
                minutes=int(activities[-1][1])
            )
            result += f'{final_time.strftime("%H:%M")} ~'
            
        return result.strip()

    def create_prompt_input(
        persona: Any,
        main_act_dur: List[List[Any]],
        truncated_act_dur: List[List[Any]],
        start_time_hour: datetime.datetime,
        end_time_hour: datetime.datetime,
        inserted_act: str,
        inserted_act_dur: int,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        return {
            'name': persona.name,
            'start_time': create_time_string(start_time_hour),
            'end_time': create_time_string(end_time_hour),
            'original_plan': create_schedule_string(main_act_dur, start_time_hour, False),
            'name2': persona.name,
            'new_event': inserted_act,
            'new_event_duration': str(inserted_act_dur),
            'name3': persona.name,
            'start_time2': create_time_string(start_time_hour),
            'end_time2': create_time_string(end_time_hour),
            'end_time3': create_time_string(end_time_hour),
            'new_schedule_init': create_schedule_string(truncated_act_dur, start_time_hour)
        }

    def clean_up_response(response: str, prompt: str) -> List[List[Any]]:
        """Clean up the schedule response."""
        try:
            # Extract schedule part
            new_schedule = prompt + " " + response.strip()
            new_schedule = new_schedule.split("The revised schedule:")[-1].strip()
            schedule_items = [item for item in new_schedule.split("\n") if " -- " in item]
            
            result = []
            for item in schedule_items:
                time_str, action = item.split(" -- ")
                start_time = time_str.split(" ~ ")[0].strip()
                end_time = time_str.split(" ~ ")[1].strip()
                
                start_dt = datetime.datetime.strptime(start_time, "%H:%M")
                end_dt = datetime.datetime.strptime(end_time, "%H:%M")
                duration = int((end_dt - start_dt).total_seconds() / 60)
                
                if duration > 0:
                    result.append([action.strip(), duration])
            
            return result
        except Exception as e:
            print(f"Error in cleanup: {e}")
            return []

    def validate_response(response: str, prompt: str) -> bool:
        """Validate the schedule response."""
        try:
            schedule = clean_up_response(response, prompt)
            if not schedule:
                return False
            
            # Calculate total schedule duration
            total_duration = sum(duration for _, duration in schedule)
            
            # Calculate expected duration from start to end time
            schedule_info = prompt.split("\n")[0]
            time_range = schedule_info.split("originally planned schedule from")[-1].strip()[:-1]
            start_str, end_str = [t.strip() for t in time_range.split("to")]
            
            start_dt = datetime.datetime.strptime(start_str, "%H:%M %p")
            end_dt = datetime.datetime.strptime(end_str, "%H:%M %p")
            expected_duration = int((end_dt - start_dt).total_seconds() / 60)
            
            # Must match expected duration
            return total_duration == expected_duration
            
        except Exception as e:
            print(f"Error in validation: {e}")
            return False

    def get_fallback(
        main_act_dur: List[List[Any]], 
        truncated_act_dur: List[List[Any]],
        inserted_act: str,
        inserted_act_dur: int,
        total_duration: int
    ) -> List[List[Any]]:
        """Create fallback schedule meeting duration requirements."""
        result = []
        current_duration = 0
        
        # Add truncated activities
        for activity, duration in truncated_act_dur:
            result.append([activity, duration])
            current_duration += duration
        
        # Add inserted activity
        if current_duration + inserted_act_dur <= total_duration:
            result.append([inserted_act, inserted_act_dur])
            current_duration += inserted_act_dur
        
        # Fill remaining time with activities from main schedule
        remaining_duration = total_duration - current_duration
        if remaining_duration > 0:
            result.append([main_act_dur[-1][0], remaining_duration])
            
        return result

    # Calculate total time span
    total_duration = int((end_time_hour - start_time_hour).total_seconds() / 60)

    # Generate prompt
    prompt_inputs = create_prompt_input(
        persona, main_act_dur, truncated_act_dur, 
        start_time_hour, end_time_hour, 
        inserted_act, inserted_act_dur, test_input
    )
    prompt = new_decomp_schedule_template.format(**prompt_inputs)
    
    # Create fallback schedule
    fallback = get_fallback(
        main_act_dur, truncated_act_dur, 
        inserted_act, inserted_act_dur, total_duration
    )
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response, prompt):
            output = fallback
        else:
            output = clean_up_response(response, prompt)
            
    except Exception as e:
        print(f"Error in schedule generation: {e}")
        output = fallback

    if verbose:
        print(f"Original schedule: {main_act_dur}")
        print(f"Inserted activity: {inserted_act} ({inserted_act_dur} min)")
        print(f"New schedule: {output}")
        print(f"Total duration: {sum(d for _, d in output)} min")
        print(f"Expected duration: {total_duration} min")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_decide_to_talk(
    persona: Any,
    target_persona: Any,
    retrieved: Dict[str, List[Any]],
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Determine if a persona should initiate conversation with another.
    
    Args:
        persona: The initiating Persona instance
        target_persona: The target Persona instance
        retrieved: Dictionary containing relevant events and thoughts
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Decision ('yes' or 'no')
        - List containing [output, prompt, prompt_input, fallback]
    """
    def get_activity_description(description: str, planned_path: List[Any], waiting: bool = False) -> str:
        """Create standardized activity description."""
        if "(" in description:
            description = description.split("(")[-1][:-1]
            
        if not planned_path and not waiting:
            return f"is already {description}"
        elif waiting:
            return f"is {description}"
        else:
            return f"is on the way to {description}"

    def create_context(retrieved: Dict[str, List[Any]]) -> str:
        """Create context from retrieved events and thoughts."""
        context = ""
        
        # Process events
        for event in retrieved["events"]:
            desc_parts = event.description.split()
            desc_parts[2:3] = ["was"]
            context += f"{' '.join(desc_parts)}. "
            
        context += "\n"
        
        # Process thoughts
        for thought in retrieved["thoughts"]:
            context += f"{thought.description}. "
            
        return context.strip()

    def create_prompt_input(
        init_persona: Any,
        target_persona: Any,
        retrieved: Dict[str, List[Any]],
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Get last chat information
        last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
        last_chat_time = ""
        last_chat_topic = ""
        if last_chat:
            last_chat_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
            last_chat_topic = last_chat.description

        # Get current time
        current_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")

        # Get activity descriptions
        persona_desc = get_activity_description(
            init_persona.scratch.act_description,
            init_persona.scratch.planned_path,
            "waiting" in init_persona.scratch.act_description
        )
        
        target_desc = get_activity_description(
            target_persona.scratch.act_description,
            target_persona.scratch.planned_path,
            "waiting" in target_persona.scratch.act_description
        )

        return {
            'context': create_context(retrieved),
            'current_time': current_time,
            'persona_name': init_persona.name,
            'target_name': target_persona.name,
            'last_chat_time': last_chat_time,
            'last_chat_topic': last_chat_topic,
            'persona_status': f"{init_persona.name} {persona_desc}",
            'target_status': f"{target_persona.name} {target_desc}",
            'persona_name2': init_persona.name,
            'target_name2': target_persona.name
        }

    def clean_up_response(response: str) -> str:
        """Clean up the response to get yes/no answer."""
        try:
            answer = response.split("Answer in yes or no:")[-1].strip().lower()
            return answer if answer in ["yes", "no"] else ""
        except:
            return ""

    def validate_response(response: str) -> bool:
        """Validate the yes/no response."""
        cleaned = clean_up_response(response)
        return cleaned in ["yes", "no"]

    def get_fallback() -> str:
        """Return fallback response."""
        return "yes"

    # Generate prompt
    prompt_inputs = create_prompt_input(
        persona, target_persona, retrieved, test_input
    )
    prompt = decide_to_talk_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in conversation decision: {e}")
        output = fallback

    if verbose:
        print(f"Persona: {persona.name}")
        print(f"Target: {target_persona.name}")
        print(f"Decision: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_decide_to_react(
    persona: Any,
    target_persona: Any,
    retrieved: Dict[str, List[Any]],
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Determine if a persona should wait or continue with their action when encountering another persona.
    
    Args:
        persona: The initiating Persona instance
        target_persona: The target Persona instance
        retrieved: Dictionary containing relevant events and thoughts
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Decision ('1' for wait or '2' for continue)
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        init_persona: Any,
        target_persona: Any,
        retrieved: Dict[str, List[Any]],
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Create context from events and thoughts
        context = ""
        for event in retrieved["events"]:
            desc_parts = event.description.split()
            desc_parts[2:3] = ["was"]
            context += f"{' '.join(desc_parts)}. "
        context += "\n"
        for thought in retrieved["thoughts"]:
            context += f"{thought.description}. "

        # Get current time
        current_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")

        # Get activity descriptions and locations
        def get_activity_desc(p: Any) -> str:
            act_desc = p.scratch.act_description
            if "(" in act_desc:
                act_desc = act_desc.split("(")[-1][:-1]
                
            loc = ""
            if ":" in p.scratch.act_address:
                arena = p.scratch.act_address.split(":")[-1]
                sector = p.scratch.act_address.split(":")[-2]
                loc = f" at {arena} in {sector}"
                
            if len(p.scratch.planned_path) == 0:
                return f"{p.name} is already {act_desc}{loc}"
            return f"{p.name} is on the way to {act_desc}{loc}"

        init_desc = get_activity_desc(init_persona)
        target_desc = get_activity_desc(target_persona)
        
        init_act = init_persona.scratch.act_description
        if "(" in init_act:
            init_act = init_act.split("(")[-1][:-1]
            
        target_act = target_persona.scratch.act_description
        if "(" in target_act:
            target_act = target_act.split("(")[-1][:-1]

        return {
            'context': context,
            'current_time': current_time,
            'init_desc': init_desc,
            'target_desc': target_desc,
            'init_name': init_persona.name,
            'init_act': init_act,
            'target_name': target_persona.name,
            'target_act': target_act
        }

    def clean_up_response(response: str) -> str:
        """Clean up the response to get option number."""
        try:
            option = response.split("Answer: Option")[-1].strip().lower()
            return option if option in ["1", "2"] else ""
        except:
            return ""

    def validate_response(response: str) -> bool:
        """Validate the option response."""
        cleaned = clean_up_response(response)
        return cleaned in ["1", "2"]

    def get_fallback() -> str:
        """Return fallback response."""
        return "2"  # Default to continuing

    # Generate prompt
    prompt_inputs = create_prompt_input(
        persona, target_persona, retrieved, test_input
    )
    prompt = decide_to_react_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in reaction decision: {e}")
        output = fallback

    if verbose:
        print(f"Persona: {persona.name}")
        print(f"Target: {target_persona.name}")
        print(f"Decision: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_create_conversation(
    persona: Any,
    target_persona: Any,
    curr_loc: Dict[str, str],
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[List[List[str]], List[Any]]:
    """
    Generate a conversation between two personas based on their current context.
    
    Args:
        persona: The initiating Persona instance
        target_persona: The target Persona instance
        curr_loc: Dictionary containing location information
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - List of [speaker, dialogue] pairs
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        init_persona: Any,
        target_persona: Any,
        curr_loc: Dict[str, str],
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Get previous conversation if exists
        prev_convo = ""
        if init_persona.a_mem.seq_chat:
            for chat in init_persona.a_mem.seq_chat:
                if chat.object == target_persona.scratch.name:
                    mins_ago = int((init_persona.scratch.curr_time - chat.created).total_seconds()/60)
                    prev_convo += f'\n{mins_ago} minutes ago, they had the following conversation:\n'
                    for speaker, text in chat.filling:
                        prev_convo += f'{speaker}: "{text}"\n'
                    break
                    
        # Clear old conversations (>8 hours)
        if init_persona.a_mem.seq_chat:
            last_chat_age = int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60)
            if last_chat_age > 480:
                prev_convo = ""

        # Get relevant thoughts
        def get_thoughts(p1: Any, p2: Any) -> str:
            thought_nodes = p1.a_mem.retrieve_relevant_thoughts(
                p2.scratch.act_event[0],
                p2.scratch.act_event[1], 
                p2.scratch.act_event[2]
            )
            return "\n".join(f"-- {node.description}" for node in thought_nodes)

        init_thoughts = get_thoughts(init_persona, target_persona)
        target_thoughts = get_thoughts(target_persona, init_persona)

        # Get current activity descriptions
        def get_activity_desc(p: Any) -> str:
            if p.scratch.planned_path:
                return f"{p.name} is on the way to {p.scratch.act_description}"
            return f"{p.name} is {p.scratch.act_description}"

        init_desc = get_activity_desc(init_persona)
        target_desc = get_activity_desc(target_persona)

        return {
            'init_iss': init_persona.scratch.get_str_iss(),
            'target_iss': target_persona.scratch.get_str_iss(),
            'init_name': init_persona.name,
            'target_name': target_persona.name,
            'init_thoughts': init_thoughts,
            'target_thoughts': target_thoughts,
            'current_time': init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S"),
            'init_desc': init_desc,
            'target_desc': target_desc,
            'prev_convo': prev_convo,
            'location': curr_loc["arena"]
        }

    def clean_up_response(response: str, prompt: str) -> List[List[str]]:
        """Clean up the conversation response."""
        try:
            # Extract conversation after prompt
            full_text = prompt + response
            dialogue = full_text.split("What would they talk about now?")[-1].strip()
            
            # Extract quoted content and speakers
            quotes = re.findall('"([^"]*)"', dialogue)
            speakers = []
            for line in dialogue.split("\n"):
                if ":" in line:
                    speaker = line.split(":")[0].strip()
                    if speaker:
                        speakers.append(speaker)

            # Pair speakers with their dialogue
            conversation = []
            for i, speaker in enumerate(speakers):
                if i < len(quotes):
                    conversation.append([speaker, quotes[i]])
                    
            return conversation
        except:
            return []

    def validate_response(response: str, prompt: str) -> bool:
        """Validate the conversation response."""
        try:
            conversation = clean_up_response(response, prompt)
            return len(conversation) >= 2 and all(len(pair) == 2 for pair in conversation)
        except:
            return False

    def get_fallback(init_persona: Any, target_persona: Any) -> List[List[str]]:
        """Return fallback conversation."""
        return [
            [init_persona.name, "Hi!"],
            [target_persona.name, "Hi!"]
        ]

    # Generate prompt
    prompt_inputs = create_prompt_input(
        persona, target_persona, curr_loc, test_input
    )
    prompt = create_conversation_template.format(**prompt_inputs)
    
    fallback = get_fallback(persona, target_persona)
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response, prompt):
            output = fallback
        else:
            output = clean_up_response(response, prompt)
            
    except Exception as e:
        print(f"Error in conversation generation: {e}")
        output = fallback

    if verbose:
        print(f"Generated conversation between {persona.name} and {target_persona.name}:")
        for speaker, text in output:
            print(f"{speaker}: {text}")

    return output, [output, prompt, gpt_param, prompt_inputs, fallback]

def run_gpt_prompt_summarize_conversation(
    persona: Any,
    conversation: List[List[str]],
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Summarize a conversation between personas in one sentence.
    
    Args:
        persona: The Persona instance (for consistency)
        conversation: List of [speaker, dialogue] pairs
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Summary string starting with "conversing about"
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        conversation: List[List[str]],
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Format conversation into string
        convo_str = "\n".join(f'{speaker}: "{text}"' for speaker, text in conversation)
        
        return {
            'conversation': convo_str
        }

    def clean_up_response(response: str) -> str:
        """Clean up the summary response."""
        # Remove any "this is a conversation about" prefix if present
        summary = response.lower().strip()
        if summary.startswith("this is a conversation about"):
            summary = summary[len("this is a conversation about"):].strip()
            
        # Ensure it starts with "conversing about"
        if not summary.startswith("conversing about"):
            summary = "conversing about " + summary
            
        return summary

    def validate_response(response: str) -> bool:
        """Validate the summary response."""
        try:
            cleaned = clean_up_response(response)
            # Ensure it's a reasonable length and contains actual content
            return (len(cleaned) > len("conversing about ") and 
                    "conversing about" in cleaned.lower())
        except:
            return False

    def get_fallback() -> str:
        """Return fallback summary."""
        return "conversing about general matters"
    
    # Generate prompt
    prompt_inputs = create_prompt_input(conversation, test_input)
    prompt = summarize_conversation_template.format(**prompt_inputs)            
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        print(f"Error in conversation summary: {e}")
        output = fallback

    if verbose:
        print(f"Original conversation length: {len(conversation)} turns")
        print(f"Generated summary: {output}")

    return output, [output, prompt, gpt_param, prompt_inputs, fallback]

def run_gpt_prompt_extract_keywords(
    persona: Any,
    description: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[set, List[Any]]:
    """
    Extract factually descriptive and emotive keywords from a description.
    
    Args:
        persona: The Persona class instance (for consistency)
        description: Text description to extract keywords from
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Set of extracted keywords
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        description: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input

        # Handle line breaks in description
        if "\n" in description:
            description = description.replace("\n", " <LINE_BREAK> ")
            
        return {
            'description': description
        }

    def clean_up_response(response: str) -> set:
        """Clean up the keywords response."""
        try:
            if verbose:
                print(f"Raw response: {response}")
                
            # More robust parsing approach
            factual_keywords = []
            emotive_keywords = []
            
            # Check if we have the pattern with "Factually descriptive keywords:" followed by keywords
            if "Factually descriptive keywords:" in response:
                parts = response.split("Factually descriptive keywords:")
                if len(parts) >= 2:
                    factual_part = parts[1]
                    
                    # Now split to see if we have emotive keywords section
                    if "Emotive keywords:" in factual_part:
                        factual_emotive_parts = factual_part.split("Emotive keywords:")
                        factual_text = factual_emotive_parts[0].strip()
                        emotive_text = factual_emotive_parts[1].strip() if len(factual_emotive_parts) > 1 else ""
                        
                        # Process factual keywords
                        factual_keywords = [k.strip() for k in factual_text.split(",") if k.strip()]
                        
                        # Process emotive keywords
                        emotive_keywords = [k.strip() for k in emotive_text.split(",") if k.strip()]
                    else:
                        # Only factual keywords
                        factual_text = factual_part.strip()
                        factual_keywords = [k.strip() for k in factual_text.split(",") if k.strip()]
            # Alternative format: response directly starts with keywords
            else:
                # Try to find Emotive keywords section
                if "Emotive keywords:" in response:
                    parts = response.split("Emotive keywords:")
                    factual_text = parts[0].strip()
                    emotive_text = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Process factual keywords
                    factual_keywords = [k.strip() for k in factual_text.split(",") if k.strip()]
                    
                    # Process emotive keywords
                    emotive_keywords = [k.strip() for k in emotive_text.split(",") if k.strip()]
                else:
                    # Just comma-separated list of keywords
                    factual_keywords = [k.strip() for k in response.split(",") if k.strip()]
            
            # Clean and combine keywords
            all_keywords = set()
            for keyword in factual_keywords + emotive_keywords:
                if keyword:
                    keyword = keyword.lower()
                    # Remove trailing period if present
                    if keyword and keyword[-1] == ".":
                        keyword = keyword[:-1]
                    all_keywords.add(keyword)
            
            if verbose:
                print(f"Extracted keywords: {all_keywords}")
                
            return all_keywords
            
        except Exception as e:
            if verbose:
                print(f"Error in keyword extraction cleanup: {e}")
            return set()

    def validate_response(response: str) -> bool:
        """Validate the keywords response."""
        try:
            keywords = clean_up_response(response)
            return len(keywords) > 0
        except Exception as e:
            if verbose:
                print(f"Validation error: {e}")
            return False

    def get_fallback() -> set:
        """Return fallback keywords."""
        return set()

    # Generate prompt
    prompt_inputs = create_prompt_input(description, test_input)
    prompt = extract_keywords_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        if verbose:
            print(f"Error in keyword extraction: {e}")
        output = fallback

    if verbose:
        print(f"Description: {description[:50]}...")
        print(f"Extracted keywords: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_keyword_to_thoughts(
    persona: Any,
    keyword: str,
    concept_summary: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Generate a thought about events/concepts related to a keyword.
    
    Args:
        persona: The Persona class instance
        keyword: The keyword to generate thoughts about
        concept_summary: Summary of relevant events/thoughts
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Generated thought as a string
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        persona: Any,
        keyword: str,
        concept_summary: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'keyword': keyword,
            'events_thoughts': concept_summary,
            'persona_name': persona.name
        }

    def clean_up_response(response: str) -> str:
        """Clean up the thought response."""
        return response.strip()

    def validate_response(response: str) -> bool:
        """Validate the thought response."""
        try:
            cleaned = clean_up_response(response)
            return len(cleaned) > 0
        except Exception as e:
            if verbose:
                print(f"Validation error: {e}")
            return False

    def get_fallback() -> str:
        """Return fallback thought."""
        return f"{persona.name} has no specific thoughts about this."

    # Generate prompt
    prompt_inputs = create_prompt_input(persona, keyword, concept_summary, test_input)
    prompt = keyword_to_thoughts_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        if verbose:
            print(f"Error in thought generation: {e}")
        output = fallback

    if verbose:
        print(f"Keyword: {keyword}")
        print(f"Generated thought: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_convo_to_thoughts(
    persona: Any,
    init_persona_name: str,
    target_persona_name: str,
    convo_str: str,
    fin_target: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[str, List[Any]]:
    """
    Generate thoughts about a conversation between two personas.
    
    Args:
        persona: The Persona class instance
        init_persona_name: Name of the initiating persona
        target_persona_name: Name of the target persona
        convo_str: String representation of the conversation
        fin_target: Target of thoughts ("the conversation" or a persona name)
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Generated thought as a string
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        init_persona_name: str,
        target_persona_name: str,
        convo_str: str,
        fin_target: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'init_persona_name': init_persona_name,
            'target_persona_name': target_persona_name,
            'convo_str': convo_str,
            'init_persona_name_repeat': init_persona_name,
            'fin_target': fin_target
        }

    def clean_up_response(response: str) -> str:
        """Clean up the thought response."""
        return response.strip()

    def validate_response(response: str) -> bool:
        """Validate the thought response."""
        try:
            cleaned = clean_up_response(response)
            return len(cleaned) > 0
        except Exception as e:
            if verbose:
                print(f"Validation error: {e}")
            return False

    def get_fallback() -> str:
        """Return fallback thought."""
        if fin_target == "the conversation":
            return f"{init_persona_name} found the conversation to be informative."
        else:
            return f"{init_persona_name} had a neutral impression of {fin_target}."

    # Generate prompt
    prompt_inputs = create_prompt_input(
        init_persona_name, 
        target_persona_name, 
        convo_str, 
        fin_target, 
        test_input
    )
    prompt = convo_to_thoughts_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
            
    except Exception as e:
        if verbose:
            print(f"Error in thought generation: {e}")
        output = fallback

    if verbose:
        print(f"Conversation between: {init_persona_name} and {target_persona_name}")
        print(f"Target of thoughts: {fin_target}")
        print(f"Generated thought: {output}")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_event_poignancy(
    persona: Any,
    event_description: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[int, List[Any]]:
    """
    Rate the poignancy of an event for a persona on a scale of 1-10.
    
    Args:
        persona: The Persona class instance
        event_description: Description of the event to rate
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Poignancy rating (1-10)
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        persona: Any,
        event_description: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'persona_name': persona.scratch.name,
            'persona_iss': persona.scratch.get_str_iss(),
            'persona_name_repeat': persona.scratch.name,
            'event_description': event_description
        }

    def clean_up_response(response: str) -> int:
        """Clean up the rating response."""
        try:
            # Extract just the numeric part if there's extra text
            for word in response.strip().split():
                if word.isdigit():
                    rating = int(word)
                    # Ensure rating is between 1 and 10
                    return max(1, min(10, rating))
            
            # If we get here, try to convert the whole response
            return int(response.strip())
        except Exception as e:
            if verbose:
                print(f"Error in clean up: {e}")
            raise ValueError("Could not parse response as integer")

    def validate_response(response: str) -> bool:
        """Validate the rating response."""
        try:
            rating = clean_up_response(response)
            return 1 <= rating <= 10
        except Exception as e:
            if verbose:
                print(f"Validation error: {e}")
            return False

    def get_fallback() -> int:
        """Return fallback rating."""
        return 4  # Neutral middle-low rating as fallback

    # Generate prompt
    prompt_inputs = create_prompt_input(persona, event_description, test_input)
    prompt = event_poignancy_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        # Use only chat_completion_request
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
    except Exception as e:
        if verbose:
            print(f"Error in poignancy rating: {e}")
        output = fallback

    if verbose:
        print(f"Event: {event_description}")
        print(f"Poignancy rating: {output}/10")

    return output, [output, prompt, prompt_inputs, fallback]

def run_gpt_prompt_thought_poignancy(
    persona: Any,
    thought_description: str,
    test_input: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[int, List[Any]]:
    """
    Rate the poignancy of a thought for a persona on a scale of 1-10.
    
    Args:
        persona: The Persona class instance
        thought_description: Description of the thought to rate
        test_input: Optional test inputs for debugging
        verbose: Whether to print debug information
    
    Returns:
        Tuple containing:
        - Poignancy rating (1-10)
        - List containing [output, prompt, prompt_input, fallback]
    """
    def create_prompt_input(
        persona: Any,
        thought_description: str,
        test_input: Optional[List[str]] = None
    ) -> dict:
        """Create the formatted input for the prompt template."""
        if test_input:
            return test_input
            
        return {
            'persona_name': persona.scratch.name,
            'persona_iss': persona.scratch.get_str_iss(),
            'persona_name_repeat': persona.scratch.name,
            'thought_description': thought_description
        }

    def clean_up_response(response: str) -> int:
        """Clean up the rating response."""
        try:
            # Extract just the numeric part if there's extra text
            for word in response.strip().split():
                if word.isdigit():
                    rating = int(word)
                    # Ensure rating is between 1 and 10
                    return max(1, min(10, rating))
            
            # If we get here, try to convert the whole response
            return int(response.strip())
        except Exception as e:
            if verbose:
                print(f"Error in clean up: {e}")
            raise ValueError("Could not parse response as integer")

    def validate_response(response: str) -> bool:
        """Validate the rating response."""
        try:
            rating = clean_up_response(response)
            return 1 <= rating <= 10
        except Exception as e:
            if verbose:
                print(f"Validation error: {e}")
            return False

    def get_fallback() -> int:
        """Return fallback rating."""
        return 4  # Neutral middle-low rating as fallback

    # Generate prompt
    prompt_inputs = create_prompt_input(persona, thought_description, test_input)
    prompt = thought_poignancy_template.format(**prompt_inputs)
    
    fallback = get_fallback()
    
    try:
        # Use only chat_completion_request
        response = chat_completion_request(prompt)
        if not validate_response(response):
            output = fallback
        else:
            output = clean_up_response(response)
    except Exception as e:
        if verbose:
            print(f"Error in thought poignancy rating: {e}")
        output = fallback

    if verbose:
        print(f"Thought: {thought_description}")
        print(f"Poignancy rating: {output}/10")

    return output, [output, prompt, prompt_inputs, fallback]

if __name__ == "__main__":
    def test_get_random_alphanumeric():
        """Test get_random_alphanumeric function with default parameters."""
        print("\nTesting get_random_alphanumeric:")
        
        # Test default parameters
        result = get_random_alphanumeric()
        print(f"Default params result: {result}")
        assert len(result) == 6, f"Expected length 6, got {len(result)}"
        assert result.isalnum(), "Result should be alphanumeric"
        print("get_random_alphanumeric test passed!")

    def test_run_gpt_prompt_wake_up_hour():
        """Test run_gpt_prompt_wake_up_hour function with a simple persona."""
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
        
        # Test case: Early riser
        test_persona = MockPersona()
        wake_hour, details = run_gpt_prompt_wake_up_hour(test_persona)
        print(f"Early riser wake hour: {wake_hour}")
        print("run_gpt_prompt_wake_up_hour test passed!")

    def test_daily_plan():
        """Test the daily plan generation with a simple persona."""
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

        # Test case: Early riser
        test_persona = MockPersona()
        schedule, details = run_gpt_prompt_daily_plan(test_persona, 5)
        print(f"Generated schedule with {len(schedule)} activities:")
        for activity in schedule:
            print(f"  - {activity}")
        print("daily_plan test passed!")

    class MockPersona:
        """Mock persona class for testing."""
        class Scratch:
            def __init__(self):
                self.daily_req = [
                    "wake up and complete morning routine at 7:00 am",
                    "work on project from 9:00 am to 12:00 pm",
                    "have lunch at 12:30 pm",
                    "exercise from 2:00 pm to 3:00 pm"
                ]

            def get_str_iss(self):
                return "Test persona traits: friendly, organized, health-conscious"
                
            def get_str_lifestyle(self):
                return "Maintains a balanced work-life schedule"
                
            def get_str_firstname(self):
                return "TestPerson"
                
            def get_str_curr_date_str(self):
                return datetime.datetime.now().strftime("%A %B %d")
                
        def __init__(self):
            self.scratch = self.Scratch()

    def test_run_gpt_prompt_generate_hourly_schedule():
        """
        Test the hourly schedule generation function.
        
        Test cases:
        - Basic schedule generation
        - Format validation
        - Proper handling of prior activities
        """
        print("\nTesting run_gpt_prompt_generate_hourly_schedule:")
        try:
            # Setup
            test_persona = MockPersona()
            curr_hour_str = "10:00 AM"
            hour_str = ["8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM"]
            p_f_ds_hourly_org = [
                "completing morning routine",
                "having breakfast"
            ]
            
            # Test execution
            activity, details = run_gpt_prompt_generate_hourly_schedule(
                persona=test_persona,
                curr_hour_str=curr_hour_str,
                p_f_ds_hourly_org=p_f_ds_hourly_org,
                hour_str=hour_str
            )
            
            # Assertions
            assert isinstance(activity, str), "Activity should be a string"
            assert len(activity) > 0, "Activity should not be empty"
            assert not activity.endswith("."), "Activity should not end with a period"
            assert len(details) == 5, "Details should contain 5 elements"
            
            # Verify prompt structure
            prompt = details[1]
            assert "Hourly schedule format:" in prompt, "Prompt should contain schedule format"
            assert test_persona.scratch.get_str_firstname() in prompt, "Prompt should contain persona name"
            assert curr_hour_str in prompt, "Prompt should contain current hour"
            
            # Verify proper formatting of prior activities
            if p_f_ds_hourly_org:
                for activity in p_f_ds_hourly_org:
                    assert activity in prompt, f"Prior activity '{activity}' should be in prompt"
            
            # Success output
            print("✓ Test passed:")
            print(f"  • Generated activity: {activity}")
            print("  • Schedule details validated")
            print("  • Prompt structure verified")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_task_decomp():
        """
        Test the task decomposition function.
        
        Test cases:
        - Basic task decomposition
        - Duration handling
        - Response cleaning
        """
        print("\nTesting task decomposition:")
        
        class MockPersona:
            """Mock persona for testing."""
            class Scratch:
                def __init__(self):
                    self.curr_time = datetime.datetime.now()
                    self.f_daily_schedule_hourly_org = [
                        ["sleeping", 360],
                        ["morning routine", 60],
                        ["having breakfast", 60],
                        ["working on project", 180]
                    ]
                    self.f_daily_schedule_hourly_org_index = 3
                    
                def get_str_iss(self):
                    return "Test persona traits: productive, focused"
                    
                def get_str_firstname(self):
                    return "TestPerson"
                    
                def get_f_daily_schedule_hourly_org_index(self):
                    return self.f_daily_schedule_hourly_org_index
                    
            def __init__(self):
                self.scratch = self.Scratch()
                self.name = "TestPerson"
        
        try:
            # Setup
            test_persona = MockPersona()
            task = "working on project"
            duration = 180
            
            # Test execution
            tasks, details = run_gpt_prompt_task_decomp(
                persona=test_persona,
                task=task,
                duration=duration,
                verbose=False
            )
            
            # Assertions
            assert isinstance(tasks, list), "Output should be a list"
            assert all(isinstance(t, list) and len(t) == 2 for t in tasks), \
                "Each task should be [description, duration]"
            
            total_duration = sum(t[1] for t in tasks)
            assert total_duration == duration, \
                f"Total duration {total_duration} should match input {duration}"
            
            assert all(isinstance(t[1], int) and t[1] > 0 for t in tasks), \
                "All durations should be positive integers"
            
            assert all(task in t[0] for t in tasks), \
                "All subtasks should include main task description"
            
            # Success output
            print("✓ Test passed:")
            print(f"  • Generated {len(tasks)} subtasks")
            print(f"  • Total duration: {total_duration} minutes")
            for t in tasks:
                print(f"  • {t[0]} ({t[1]} minutes)")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_action_sector():
        """
        Test the action sector selection function.
        
        Test cases:
        - Basic sector selection
        - Staying in current sector when appropriate
        - Selecting new sector when needed
        """
        print("\nTesting action sector selection:")
        
        class MockMaze:
            """Mock maze class for testing."""
            def __init__(self):
                self._curr_sector = "Test House"
                
            def access_tile(self, tile):
                return {
                    'world': 'test_world',
                    'sector': self._curr_sector
                }
            
            def set_sector(self, sector):
                self._curr_sector = sector

        class MockPersona:
            """Mock persona class for testing."""
            class Scratch:
                def __init__(self):
                    self.curr_tile = "A1"
                    self.last_name = "TestPerson"
                    self.living_area = "test_world:Test House"
                    
                def get_str_name(self):
                    return "Test Person"
                    
            class Memory:
                def get_str_accessible_sectors(self, world):
                    return "Test House, Test Park, Test Store, Test Cafe"
                
                def get_str_accessible_sector_arenas(self, location):
                    if "House" in location:
                        return "bedroom, kitchen, bathroom"
                    elif "Park" in location:
                        return "playground, walking paths"
                    elif "Store" in location:
                        return "shopping area, checkout"
                    elif "Cafe" in location:
                        return "dining area, counter"
                    return "main area"
                    
            def __init__(self):
                self.scratch = self.Scratch()
                self.s_mem = self.Memory()
        
        try:
            # Setup
            test_persona = MockPersona()
            test_maze = MockMaze()
            
            # Test cases
            test_cases = [
                # (current_sector, action, expected_sector)
                ("Test House", "sleeping (in bed)", "Test House"),
                ("Test House", "shopping (for groceries)", "Test Store"),
                ("Test Park", "taking a walk", "Test Park"),
                ("Test House", "having lunch", "Test Cafe")
            ]
            
            results = []
            for curr_sector, action, expected in test_cases:
                # Set current sector
                test_maze.set_sector(curr_sector)
                
                # Test execution
                sector, details = run_gpt_prompt_action_sector(
                    action_description=action,
                    persona=test_persona,
                    maze=test_maze,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(sector, str), "Sector should be a string"
                assert sector in test_persona.s_mem.get_str_accessible_sectors("test_world"), \
                    f"Selected sector '{sector}' should be accessible"
                
                results.append((action, curr_sector, sector, sector == expected))
            
            # Success output
            print("✓ Test passed:")
            for action, curr_sector, sector, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} From {curr_sector:<12} -> Action: {action:<20} -> Sector: {sector}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False
    
    def test_action_arena():
        """
        Test the action arena selection function.
        
        Test cases:
        - Basic arena selection
        - Proper arena for different actions
        - Arena accessibility validation
        """
        print("\nTesting action arena selection:")
        
        class MockMaze:
            """Mock maze class for testing."""
            def __init__(self):
                self._curr_arena = "living room"
                
            def access_tile(self, tile):
                return {
                    'world': 'test_world',
                    'sector': 'Test House',
                    'arena': self._curr_arena
                }
            
            def set_arena(self, arena):
                self._curr_arena = arena

        class MockPersona:
            """Mock persona class for testing."""
            class Scratch:
                def __init__(self):
                    self.curr_tile = "A1"
                    self.last_name = "TestPerson"
                    
                def get_str_name(self):
                    return "Test Person"
                    
            class Memory:
                def get_str_accessible_sector_arenas(self, location):
                    if "House" in location:
                        return "bedroom, kitchen, bathroom, living room"
                    elif "Park" in location:
                        return "playground, walking paths, bench area"
                    elif "Store" in location:
                        return "shopping area, checkout counter"
                    elif "Cafe" in location:
                        return "dining area, counter, patio"
                    return "main area"
                    
            def __init__(self):
                self.scratch = self.Scratch()
                self.s_mem = self.Memory()
        
        try:
            # Setup
            test_persona = MockPersona()
            test_maze = MockMaze()
            
            # Test cases with current location and target
            test_cases = [
                # (current_arena, world, sector, action, expected_arena)
                ("bedroom", "test_world", "Test House", "sleeping (in bed)", "bedroom"),
                ("living room", "test_world", "Test House", "cooking (dinner)", "kitchen"),
                ("entrance", "test_world", "Test Park", "exercising (jogging)", "walking paths"),
                ("entrance", "test_world", "Test Cafe", "eating (lunch)", "dining area")
            ]
            
            results = []
            for curr_arena, world, sector, action, expected in test_cases:
                # Set current arena
                test_maze.set_arena(curr_arena)
                
                # Test execution
                arena, details = run_gpt_prompt_action_arena(
                    action_description=action,
                    persona=test_persona,
                    maze=test_maze,
                    act_world=world,
                    act_sector=sector,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(arena, str), "Arena should be a string"
                accessible_arenas = test_persona.s_mem.get_str_accessible_sector_arenas(f"{world}:{sector}")
                assert arena in accessible_arenas.split(", "), \
                    f"Selected arena '{arena}' should be in accessible arenas: {accessible_arenas}"
                
                results.append((action, curr_arena, sector, arena, arena == expected))
            
            # Success output
            print("✓ Test passed:")
            for action, curr_arena, sector, arena, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} From {curr_arena:<12} -> Action: {action:<20} in {sector:<15} -> Arena: {arena}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_action_game_object():
        """
        Test the game object selection function.
        
        Test cases:
        - Basic object selection
        - Object availability
        - Action-object relevance
        """
        print("\nTesting game object selection:")
        
        class MockMaze:
            """Mock maze class for testing."""
            def access_tile(self, tile):
                return {
                    'world': 'test_world',
                    'sector': 'Test House',
                    'arena': 'bedroom'
                }

        class MockPersona:
            """Mock persona class for testing."""
            class Scratch:
                def __init__(self):
                    self.curr_tile = "A1"
                    self.last_name = "TestPerson"
                    
                def get_str_name(self):
                    return "Test Person"
                    
            class Memory:
                def get_str_accessible_arena_game_objects(self, location):
                    if "bedroom" in location.lower():
                        return "bed, desk, chair, closet"
                    elif "kitchen" in location.lower():
                        return "stove, fridge, sink, counter"
                    elif "living_room" in location.lower():
                        return "couch, TV, coffee table, bookshelf"
                    return "chair, table"
                    
            def __init__(self):
                self.scratch = self.Scratch()
                self.s_mem = self.Memory()
        
        try:
            # Setup
            test_persona = MockPersona()
            test_maze = MockMaze()
            
            # Test cases
            test_cases = [
                # (location, action, expected_object)
                ("test_world:Test House:bedroom", "sleeping (in bed)", "bed"),
                ("test_world:Test House:kitchen", "cooking (dinner)", "stove"),
                ("test_world:Test House:living_room", "watching (TV show)", "TV"),
                ("test_world:Test House:bedroom", "studying (homework)", "desk")
            ]
            
            results = []
            for location, action, expected in test_cases:
                # Test execution
                game_object, details = run_gpt_prompt_action_game_object(
                    action_description=action,
                    persona=test_persona,
                    maze=test_maze,
                    temp_address=location,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(game_object, str), "Game object should be a string"
                assert game_object in test_persona.s_mem.get_str_accessible_arena_game_objects(location), \
                    f"Selected object '{game_object}' should be accessible"
                
                results.append((action, location, game_object, game_object == expected))
            
            # Success output
            print("✓ Test passed:")
            for action, location, obj, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Action: {action:<20} in {location.split(':')[-1]:<15} -> Object: {obj}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_pronunciation():
        """
        Test the emoji generation function.
        
        Test cases:
        - Basic emoji generation
        - Handling of parentheses
        - Length limits
        - Emoji validation
        """
        print("\nTesting emoji generation:")
        
        class MockPersona:
            """Mock persona class for testing."""
            class Scratch:
                def __init__(self):
                    self.curr_tile = "A1"
                    
                def get_str_name(self):
                    return "Test Person"
                    
            def __init__(self):
                self.scratch = self.Scratch()
        
        try:
            # Setup
            test_persona = MockPersona()
            
            # Test cases
            test_cases = [
                # (action, expected_type)
                ("sleeping (in bed)", "emoji"),
                ("eating (breakfast)", "emoji"),
                ("working (on computer)", "emoji"),
                ("reading (a book)", "emoji"),
                ("exercising (at gym)", "emoji")
            ]
            
            results = []
            for action, expected_type in test_cases:
                # Test execution
                emoji, details = run_gpt_prompt_pronunciation(
                    action_description=action,
                    persona=test_persona,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(emoji, str), "Response should be a string"
                assert len(emoji) <= 4, "Should not exceed 2 emojis (4 chars)"
                assert any(ord(char) > 127 for char in emoji), "Should contain emoji characters"
                
                results.append((action, emoji, True))  # Always True if assertions pass
            
            # Success output
            print("✓ Test passed:")
            for action, emoji, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Action: {action:<25} -> Emoji: {emoji}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_event_triple():
        """
        Test the event triple generation function.
        
        Test cases:
        - Basic triple generation
        - Handling of parentheses
        - Proper verb forms
        - Object handling
        """
        print("\nTesting event triple generation:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Test Person")
            
            # Test cases
            test_cases = [
                # (action, expected_predicate, expected_object)
                ("sleeping (in bed)", "sleep", "bed"),
                ("eating (breakfast)", "eat", "breakfast"),
                ("working (on computer)", "work", "computer"),
                ("reading (a book)", "read", "book"),
                ("exercising (at gym)", "exercise", "gym")
            ]
            
            results = []
            for action, exp_pred, exp_obj in test_cases:
                # Test execution
                triple, details = run_gpt_prompt_event_triple(
                    action_description=action,
                    persona=test_persona,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(triple, tuple), "Output should be a tuple"
                assert len(triple) == 3, "Triple should have 3 elements"
                assert triple[0] == test_persona.name, "Subject should be persona name"
                assert all(isinstance(x, str) for x in triple), "All elements should be strings"
                
                results.append((
                    action, 
                    triple, 
                    triple[1].lower().strip() == exp_pred and 
                    triple[2].lower().strip() == exp_obj
                ))
            
            # Success output
            print("✓ Test passed:")
            for action, triple, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Action: {action:<25} -> Triple: {triple}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_act_obj_desc():
        """
        Test the action object description function.
        
        Test cases:
        - Basic object state description
        - Different object types
        - Various actions
        """
        print("\nTesting object state description:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Test Person")
            
            # Test cases
            test_cases = [
                # (object, action_desc, expected_state_contains)
                ("bed", "sleeping in bed", "being used"),
                ("computer", "working on computer", "being operated"),
                ("stove", "cooking on stove", "heating"),
                ("book", "reading a book", "being read"),
                ("TV", "watching TV shows", "playing")
            ]
            
            results = []
            for obj, action, expected in test_cases:
                # Test execution
                state_desc, details = run_gpt_prompt_act_obj_desc(
                    act_game_object=obj,
                    act_desp=action,
                    persona=test_persona,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(state_desc, str), "State description should be a string"
                assert len(state_desc.split()) >= 2, "State should be at least two words"
                assert obj in state_desc, "Object name should be in description"
                assert "is" in state_desc, "Description should use 'is'"
                
                results.append((
                    obj, 
                    action, 
                    state_desc, 
                    expected.lower() in state_desc.lower()
                ))
            
            # Success output
            print("✓ Test passed:")
            for obj, action, state, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Object: {obj:<10} Action: {action:<20} -> State: {state}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_act_obj_event_triple():
        """
        Test the object event triple generation function.
        
        Test cases:
        - Basic triple generation
        - Object state parsing
        - Triple format validation
        """
        print("\nTesting object event triple generation:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Test Person")
            
            # Test cases
            test_cases = [
                # (object, state_desc, expected_pred, expected_obj)
                ("bed", "being used for sleeping", "support", "person"),
                ("computer", "running programs", "run", "programs"),
                ("stove", "heating up food", "heat", "food"),
                ("TV", "displaying a show", "display", "show"),
                ("book", "being read", "contain", "content")
            ]
            
            results = []
            for obj, state, exp_pred, exp_obj in test_cases:
                # Test execution
                triple, details = run_gpt_prompt_act_obj_event_triple(
                    act_game_object=obj,
                    act_obj_desc=state,
                    persona=test_persona,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(triple, tuple), "Output should be a tuple"
                assert len(triple) == 3, "Triple should have 3 elements"
                assert triple[0] == obj, "Subject should be object name"
                assert all(isinstance(x, str) for x in triple), "All elements should be strings"
                
                results.append((
                    obj,
                    state,
                    triple,
                    triple[1].lower().strip() == exp_pred and 
                    triple[2].lower().strip() == exp_obj
                ))
            
            # Success output
            print("✓ Test passed:")
            for obj, state, triple, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Object: {obj:<10} State: {state:<25} -> Triple: {triple}")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_new_decomp_schedule():
        """
        Test the schedule decomposition function.
        
        Test cases:
        - Basic schedule modification
        - Time calculations
        - Duration validation
        """
        print("\nTesting schedule decomposition:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Test Person")
            base_time = datetime.datetime(2025, 1, 1, 9, 0)  # 9:00 AM
            
            # Original schedule
            main_schedule = [
                ["working on computer", 120],  # 2 hours
                ["having lunch", 60],          # 1 hour
                ["reading book", 120]          # 2 hours
            ]
            
            # Truncated schedule (before interruption)
            truncated_schedule = [
                ["working on computer", 120],
                ["having lunch", 30]  # Cut short
            ]
            
            # Test cases
            test_cases = [
                # (start_time, end_time, inserted_act, duration)
                (
                    base_time,  # 9:00 AM
                    base_time + datetime.timedelta(hours=5),  # 2:00 PM
                    "attending emergency meeting",
                    45
                ),
                (
                    base_time,  # 9:00 AM
                    base_time + datetime.timedelta(hours=5),  # 2:00 PM
                    "handling urgent call",
                    30
                )
            ]
            
            results = []
            for start_time, end_time, act, duration in test_cases:
                # Test execution
                new_schedule, details = run_gpt_prompt_new_decomp_schedule(
                    persona=test_persona,
                    main_act_dur=main_schedule,
                    truncated_act_dur=truncated_schedule,
                    start_time_hour=start_time,
                    end_time_hour=end_time,
                    inserted_act=act,
                    inserted_act_dur=duration,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(new_schedule, list), "Output should be a list"
                assert all(isinstance(item, list) and len(item) == 2 for item in new_schedule), \
                    "Each item should be [activity, duration]"
                
                # Calculate total duration
                total_duration = sum(d for _, d in new_schedule)
                expected_duration = int((end_time - start_time).total_seconds() / 60)
                assert total_duration == expected_duration, \
                    f"Total duration {total_duration} should match time span {expected_duration}"
                
                results.append((act, duration, new_schedule, total_duration == expected_duration))
            
            # Success output
            print("✓ Test passed:")
            for act, duration, schedule, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} Inserted: {act} ({duration} min)")
                for activity, dur in schedule:
                    print(f"      • {activity:<30} {dur} min")
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_decide_to_talk():
        """
        Test the conversation decision function.
        
        Test cases:
        - Basic decision making
        - Context influence
        - Activity state impact
        """
        print("\nTesting conversation decision:")
        
        class MockNode:
            """Mock memory node for testing."""
            def __init__(self, description: str, created_time: Optional[datetime.datetime] = None):
                self.description = description
                self.created = created_time or datetime.datetime.now()
                
        class MockMemory:
            """Mock memory for testing."""
            def __init__(self):
                self.last_chat_time = datetime.datetime.now() - datetime.timedelta(hours=2)
                
            def get_last_chat(self, name):
                return MockNode(
                    f"chatting about the weather with {name}",
                    created_time=self.last_chat_time
                )
                
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str, activity: str, waiting: bool = False):
                self.name = name
                self.a_mem = MockMemory()
                self.scratch = type('Scratch', (), {
                    'curr_time': datetime.datetime.now(),
                    'act_description': activity,
                    'planned_path': [] if not waiting else ['path'],
                })()
        
        try:
            # Test cases with timestamps
            test_time = datetime.datetime.now()
            past_time = test_time - datetime.timedelta(hours=1)
            
            test_cases = [
                # (persona_name, persona_activity, target_name, target_activity, context)
                (
                    "Alice", "reading a book", "Bob", "having coffee",
                    {
                        "events": [
                            MockNode("Alice was talking with Bob", past_time),
                            MockNode("Bob was helping Alice", past_time)
                        ],
                        "thoughts": [
                            MockNode("Alice thinks Bob is friendly", test_time),
                            MockNode("Alice wants to discuss books", test_time)
                        ]
                    }
                ),
                (
                    "Charlie", "working (on laptop)", "Diana", "in meeting",
                    {
                        "events": [
                            MockNode("Charlie was busy working", past_time),
                            MockNode("Diana was in another meeting", past_time)
                        ],
                        "thoughts": [
                            MockNode("Charlie needs to focus", test_time),
                            MockNode("Charlie respects work boundaries", test_time)
                        ]
                    }
                )
            ]
            
            results = []
            for p_name, p_act, t_name, t_act, context in test_cases:
                # Create personas
                persona = MockPersona(p_name, p_act)
                target = MockPersona(t_name, t_act)
                
                # Test execution
                decision, details = run_gpt_prompt_decide_to_talk(
                    persona=persona,
                    target_persona=target,
                    retrieved=context,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(decision, str), "Decision should be a string"
                assert decision in ["yes", "no"], "Decision should be yes or no"
                assert len(details) == 4, "Details should have 4 elements"
                
                # Test context formation
                prompt = details[1]
                assert p_name in prompt, f"Persona name {p_name} should be in prompt"
                assert t_name in prompt, f"Target name {t_name} should be in prompt"
                assert "Answer in yes or no:" in prompt, "Prompt should include answer format"
                
                results.append((p_name, t_name, p_act, t_act, decision))
            
            # Success output
            print("✓ Test passed:")
            for p_name, t_name, p_act, t_act, decision in results:
                print(f"  • {p_name} ({p_act}) -> {t_name} ({t_act}): {decision}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_decide_to_react():
        """
        Test the reaction decision function.
        
        Test cases:
        - Basic decision making for conflicting activities
        - Decision making for non-conflicting activities
        - Context influence on decisions
        """
        print("\nTesting reaction decision:")
        
        class MockNode:
            """Mock memory node for testing."""
            def __init__(self, description: str, created_time: Optional[datetime.datetime] = None):
                self.description = description
                self.created = created_time or datetime.datetime.now()
                
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str, activity: str, address: str = "", planned_path: List[str] = []):
                self.name = name
                self.scratch = type('Scratch', (), {
                    'curr_time': datetime.datetime.now(),
                    'act_description': activity,
                    'act_address': address,
                    'planned_path': planned_path
                })()
        
        try:
            # Test cases
            test_cases = [
                # Conflicting activities (same location)
                (
                    "Alice", "using bathroom", "dorm:bathroom", [],
                    "Bob", "using bathroom", "dorm:bathroom", [],
                    {
                        "events": [
                            MockNode("Alice was talking to Bob")
                        ],
                        "thoughts": [
                            MockNode("Alice needs to use the bathroom urgently")
                        ]
                    },
                    "1"  # Should wait
                ),
                
                # Non-conflicting activities (different locations)
                (
                    "Charlie", "studying", "library:study room", [],
                    "Diana", "exercising", "gym:fitness area", [],
                    {
                        "events": [
                            MockNode("Charlie was discussing homework with Diana")
                        ],
                        "thoughts": [
                            MockNode("Charlie needs to finish assignment")
                        ]
                    },
                    "2"  # Should continue
                )
            ]
            
            results = []
            for (p_name, p_act, p_addr, p_path,
                t_name, t_act, t_addr, t_path,
                context, expected) in test_cases:
                
                # Create personas
                persona = MockPersona(p_name, p_act, p_addr, p_path)
                target = MockPersona(t_name, t_act, t_addr, t_path)
                
                # Test execution
                decision, details = run_gpt_prompt_decide_to_react(
                    persona=persona,
                    target_persona=target,
                    retrieved=context,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(decision, str), "Decision should be a string"
                assert decision in ["1", "2"], "Decision should be 1 or 2"
                
                results.append((
                    p_name, p_act, 
                    t_name, t_act,
                    decision,
                    decision == expected
                ))
            
            # Success output
            print("✓ Test passed:")
            for p_name, p_act, t_name, t_act, decision, matches in results:
                status = "✓" if matches else "~"
                action = "wait" if decision == "1" else "continue"
                print(f"  {status} {p_name} ({p_act}) encounters {t_name} ({t_act}): {action}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_create_conversation():
        """
        Test the conversation generation function.
        
        Test cases:
        - Basic conversation generation
        - Conversation with previous chat history
        - Conversation influenced by thoughts
        - Location-specific conversation
        """
        print("\nTesting conversation generation:")
        
        class MockNode:
            """Mock memory node for testing."""
            def __init__(self, description: str, created_time: Optional[datetime.datetime] = None):
                self.description = description
                self.created = created_time or datetime.datetime.now()
                
        class MockChat:
            """Mock chat history."""
            def __init__(self, object_name: str, dialogue: List[List[str]], time_ago: int = 30):
                self.object = object_name
                self.filling = dialogue
                self.created = datetime.datetime.now() - datetime.timedelta(minutes=time_ago)
                
        class MockPersona:
            """Mock persona class for testing."""
            class Memory:
                def __init__(self, thoughts: List[str], chats: List[Any] = None):
                    self.thoughts = [MockNode(t) for t in thoughts]
                    self.seq_chat = chats or []
                    
                def retrieve_relevant_thoughts(self, *args):
                    return self.thoughts
                    
            class Scratch:
                def __init__(self, name: str, traits: str):
                    self.name = name
                    self._traits = traits
                    self.curr_time = datetime.datetime.now()
                    self.act_description = ""
                    self.planned_path = []
                    self.act_event = ['event1', 'event2', 'event3']
                    
                def get_str_iss(self):
                    return f"{self.name} is {self._traits}."
                    
            def __init__(
                self, 
                name: str, 
                activity: str, 
                thoughts: List[str],
                traits: str = "friendly and outgoing",
                chats: List[Any] = None,
                planned_path: List[str] = []
            ):
                self.name = name
                self.scratch = self.Scratch(name, traits)
                self.scratch.act_description = activity
                self.scratch.planned_path = planned_path
                self.a_mem = self.Memory(thoughts, chats)
        
        try:
            # Test cases
            test_cases = [
                # Basic conversation
                (
                    MockPersona(
                        "Alice", 
                        "reading a book", 
                        ["Alice thinks Bob is very studious"],
                        "bookish and quiet"
                    ),
                    MockPersona(
                        "Bob", 
                        "studying for exam",
                        ["Bob appreciates Alice's help with studies"],
                        "hardworking student"
                    ),
                    {"arena": "library"}
                ),
                
                # Conversation with history
                (
                    MockPersona(
                        "Charlie",
                        "having coffee",
                        ["Charlie enjoys Diana's company"],
                        "social and friendly",
                        [MockChat("Diana", [["Charlie", "How's your day?"], ["Diana", "Great!"]])]
                    ),
                    MockPersona(
                        "Diana",
                        "taking a break",
                        ["Diana finds Charlie interesting"],
                        "cheerful person"
                    ),
                    {"arena": "cafe"}
                )
            ]
            
            results = []
            for init_persona, target_persona, location in test_cases:
                # Test execution
                conversation, details = run_gpt_prompt_create_conversation(
                    persona=init_persona,
                    target_persona=target_persona,
                    curr_loc=location,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(conversation, list), "Conversation should be a list"
                assert len(conversation) >= 2, "Should have at least 2 dialogue turns"
                assert all(len(turn) == 2 for turn in conversation), "Each turn should have speaker and text"
                assert conversation[0][0] == init_persona.name, "First speaker should be initiator"
                
                results.append((
                    init_persona.name,
                    target_persona.name,
                    location["arena"],
                    len(conversation),
                    conversation
                ))
            
            # Success output
            print("✓ Test passed:")
            for init_name, target_name, loc, turns, convo in results:
                print(f"  • Conversation between {init_name} and {target_name} in {loc}:")
                print(f"    {turns} dialogue turns generated")
                for speaker, text in convo[:2]:  # Show first two turns
                    print(f"    {speaker}: {text}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_summarize_conversation():
        """
        Test the conversation summarization function.
        
        Test cases:
        - Basic conversation summarization
        - Handling of different conversation types
        - Proper summary formatting
        """
        print("\nTesting conversation summarization:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Test cases
            test_cases = [
                # Social conversation
                (
                    [
                        ["Alice", "Hey Bob, want to grab lunch later?"],
                        ["Bob", "Sure! Where should we go?"],
                        ["Alice", "How about that new Italian place?"],
                        ["Bob", "Perfect, let's meet at 12:30!"]
                    ],
                    "lunch plans"  # Expected to contain this
                ),
                
                # Work conversation
                (
                    [
                        ["Charlie", "Have you finished the report?"],
                        ["Diana", "Yes, just need to review the numbers."],
                        ["Charlie", "Great, we need to submit it by 5pm."],
                        ["Diana", "I'll have it ready before then."]
                    ],
                    "work report"  # Expected to contain this
                ),
                
                # Casual greeting
                (
                    [
                        ["Eve", "Good morning!"],
                        ["Frank", "Morning! How are you?"],
                        ["Eve", "Doing well, thanks!"]
                    ],
                    "morning greetings"  # Expected to contain this
                )
            ]
            
            results = []
            for conversation, expected_topic in test_cases:
                # Create test persona
                test_persona = MockPersona(conversation[0][0])
                
                # Test execution
                summary, details = run_gpt_prompt_summarize_conversation(
                    persona=test_persona,
                    conversation=conversation,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(summary, str), "Summary should be a string"
                assert summary.startswith("conversing about"), "Summary should start with 'conversing about'"
                assert len(summary) > len("conversing about "), "Summary should have actual content"
                
                # Check if expected topic is mentioned
                contains_topic = expected_topic.lower() in summary.lower()
                
                results.append((
                    conversation[0][0],  # First speaker
                    conversation[1][0],  # Second speaker
                    len(conversation),   # Conversation length
                    summary,
                    contains_topic
                ))
            
            # Success output
            print("✓ Test passed:")
            for speaker1, speaker2, length, summary, matches in results:
                status = "✓" if matches else "~"
                print(f"  {status} {speaker1}-{speaker2} ({length} turns):")
                print(f"      Summary: {summary}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_extract_keywords():
        """
        Test the keyword extraction function.
        
        Test cases:
        - Basic description
        - Conversation description
        - Description with mixed emotions
        """
        print("\nTesting keyword extraction:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Test Person")
            
            # Test cases
            test_cases = [
                # Basic event description
                "Alice completed her homework assignment after studying for three hours at the library. She felt proud of her work.",
                
                # Conversation description
                "Bob and Charlie had a heated argument about politics during lunch. Bob was frustrated while Charlie remained calm.",
                
                # Mixed emotions description
                "Diana received both good and bad news today. She was excited about her promotion but sad about having to relocate."
            ]
            
            results = []
            for description in test_cases:
                # Test execution
                keywords, details = run_gpt_prompt_extract_keywords(
                    persona=test_persona,
                    description=description,
                    verbose=False
                )
                
                # Assertions
                assert isinstance(keywords, set), "Output should be a set"
                assert len(keywords) > 0, "Should extract at least one keyword"
                
                results.append((
                    description[:30] + "...",
                    len(keywords),
                    keywords
                ))
            
            # Success output
            print("✓ Test passed:")
            for desc, count, keywords in results:
                print(f"  • Description: {desc}")
                print(f"    {count} keywords extracted: {', '.join(list(keywords)[:5])}{'...' if len(keywords) > 5 else ''}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_keyword_to_thoughts():
        """
        Test the keyword to thoughts function.
        
        Test cases:
        - Basic keyword with events
        - Emotional concept
        - Abstract concept
        """
        print("\nTesting keyword to thoughts:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Jane Smith")
            
            # Test cases
            test_cases = [
                # Basic keyword with events
                (
                    "coffee", 
                    "Jane Smith went to a cafe yesterday. She ordered a latte. She enjoyed the coffee and atmosphere."
                ),
                
                # Emotional concept
                (
                    "friendship", 
                    "Jane Smith helped her friend move apartments. They had dinner together afterward. They reminisced about college days."
                ),
                
                # Abstract concept
                (
                    "success", 
                    "Jane Smith completed a difficult project at work. Her boss praised her efforts. She was promoted last month."
                )
            ]
            
            results = []
            for keyword, summary in test_cases:
                # Test execution
                thought, details = run_gpt_prompt_keyword_to_thoughts(
                    persona=test_persona,
                    keyword=keyword,
                    concept_summary=summary,
                    verbose=True
                )
                
                # Assertions
                assert isinstance(thought, str), "Output should be a string"
                assert len(thought) > 0, "Should generate a non-empty thought"
                assert test_persona.name in thought, "Thought should include persona name"
                
                results.append((
                    keyword,
                    thought
                ))
            
            # Success output
            print("✓ Test passed:")
            for keyword, thought in results:
                print(f"  • Keyword: {keyword}")
                print(f"    Thought: {thought}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_convo_to_thoughts():
        """
        Test the conversation to thoughts function.
        
        Test cases:
        - Basic conversation with thoughts about a person
        - Conversation with thoughts about the conversation itself
        - Complex conversation with mixed sentiments
        """
        print("\nTesting conversation to thoughts:")
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str):
                self.name = name
        
        try:
            # Setup
            test_persona = MockPersona("Alice")
            
            # Test cases
            test_cases = [
                # Thoughts about a person
                (
                    "Alice", "Bob",
                    "Alice: Hey Bob, thanks for helping me with that project yesterday.\nBob: No problem! I enjoyed working with you.\nAlice: We should collaborate more often.\nBob: Definitely, your ideas were really creative.",
                    "Bob"
                ),
                
                # Thoughts about the conversation
                (
                    "Charlie", "Diana",
                    "Charlie: Did you see the new policy changes?\nDiana: Yes, they seem quite restrictive.\nCharlie: I'm concerned about how they'll affect our team.\nDiana: We should bring it up at the next meeting.",
                    "the conversation"
                ),
                
                # Mixed sentiments
                (
                    "Eva", "Frank",
                    "Eva: I noticed you missed the deadline yesterday.\nFrank: Sorry about that, I had a family emergency.\nEva: I understand, but please let me know next time.\nFrank: I will, I apologize for any inconvenience.",
                    "Frank"
                )
            ]
            
            results = []
            for init_name, target_name, convo, fin_target in test_cases:
                # Test execution
                thought, details = run_gpt_prompt_convo_to_thoughts(
                    persona=test_persona,
                    init_persona_name=init_name,
                    target_persona_name=target_name,
                    convo_str=convo,
                    fin_target=fin_target,
                    verbose=True
                )
                
                # Assertions
                assert isinstance(thought, str), "Output should be a string"
                assert len(thought) > 0, "Should generate a non-empty thought"
                assert init_name in thought, "Thought should include initiator name"
                
                results.append((
                    init_name,
                    target_name,
                    fin_target,
                    thought
                ))
            
            # Success output
            print("✓ Test passed:")
            for init_name, target_name, fin_target, thought in results:
                print(f"  • {init_name} -> {target_name} (about {fin_target})")
                print(f"    Thought: {thought}")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False

    def test_event_poignancy():
        """
        Test the event poignancy rating function.
        
        Test cases:
        - Mundane events
        - Moderately important events
        - Highly poignant events
        """
        print("\nTesting event poignancy rating:")
        
        class MockScratch:
            def __init__(self, name: str, traits: str):
                self.name = name
                self._traits = traits
                
            def get_str_iss(self):
                return f"{self.name} is {self._traits}."
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str, traits: str):
                self.scratch = MockScratch(name, traits)
        
        try:
            # Setup - create a test persona
            test_persona = MockPersona("James", "ambitious, career-focused, and analytical")
            
            # Test cases
            test_cases = [
                # Mundane events (should rate low)
                "doing grocery shopping for the week",
                "brushing teeth before bed",
                "taking out the trash",
                
                # Moderately important events (should rate medium)
                "meeting a friend for coffee after not seeing them for a month",
                "receiving positive feedback from a supervisor at work",
                "attending a birthday party for a colleague",
                
                # Highly poignant events (should rate high)
                "getting engaged to long-term partner",
                "receiving news of a parent's serious illness",
                "being accepted to a prestigious university"
            ]
            
            results = []
            
            # Define a mock chat_completion_request function for testing
            def mock_chat_completion(*args, **kwargs):
                event_text = args[0]
                if "grocery" in event_text or "teeth" in event_text or "trash" in event_text:
                    return "2"
                elif "coffee" in event_text or "feedback" in event_text or "birthday" in event_text:
                    return "5"
                else:
                    return "8"
            
            # Save original function and set mock
            global chat_completion_request
            original_func = chat_completion_request
            chat_completion_request = mock_chat_completion
            
            try:
                for event in test_cases:
                    # Test execution
                    rating, details = run_gpt_prompt_event_poignancy(
                        persona=test_persona,
                        event_description=event,
                        verbose=True
                    )
                    
                    # Assertions
                    assert isinstance(rating, int), "Output should be an integer"
                    assert 1 <= rating <= 10, "Rating should be between 1 and 10"
                    
                    results.append((test_persona.scratch.name, event, rating))
            finally:
                # Restore original function
                chat_completion_request = original_func
            
            # Success output
            print("✓ Test passed:")
            for persona_name, event, rating in results:
                poignancy_level = "Low" if rating <= 3 else "Medium" if rating <= 7 else "High"
                print(f"  • {persona_name} - Event: {event[:30]}...")
                print(f"    Rating: {rating}/10 ({poignancy_level})")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False
    
    def test_thought_poignancy():
        """
        Test the thought poignancy rating function.
        
        Test cases:
        - Mundane thoughts
        - Moderately significant thoughts
        - Highly significant thoughts
        """
        print("\nTesting thought poignancy rating:")
        
        class MockScratch:
            def __init__(self, name: str, traits: str):
                self.name = name
                self._traits = traits
                
            def get_str_iss(self):
                return f"{self.name} is {self._traits}."
        
        class MockPersona:
            """Mock persona class for testing."""
            def __init__(self, name: str, traits: str):
                self.scratch = MockScratch(name, traits)
        
        try:
            # Setup - create a test persona
            test_persona = MockPersona("Sophia", "empathetic, family-oriented, and artistic")
            
            # Test cases
            test_cases = [
                # Mundane thoughts (should rate low)
                "I need to remember to buy milk on the way home",
                "I should clean my room this weekend",
                "I wonder what I'll have for lunch today",
                
                # Moderately significant thoughts (should rate medium)
                "I hope my presentation goes well tomorrow",
                "I wonder if I should ask for a raise at work",
                "I miss spending time with my old friends",
                
                # Highly significant thoughts (should rate high)
                "I think I'm in love with my best friend",
                "I want to completely change my career path",
                "I'm considering moving to another country"
            ]
            
            results = []
            
            # Define a mock chat_completion_request function for testing
            def mock_chat_completion(*args, **kwargs):
                thought_text = args[0]
                if "buy milk" in thought_text or "clean" in thought_text or "lunch" in thought_text:
                    return "2"
                elif "presentation" in thought_text or "raise" in thought_text or "friends" in thought_text:
                    return "6"
                else:
                    return "9"
            
            # Save original function and set mock
            global chat_completion_request
            original_func = chat_completion_request
            chat_completion_request = mock_chat_completion
            
            try:
                for thought in test_cases:
                    # Test execution
                    rating, details = run_gpt_prompt_thought_poignancy(
                        persona=test_persona,
                        thought_description=thought,
                        verbose=True
                    )
                    
                    # Assertions
                    assert isinstance(rating, int), "Output should be an integer"
                    assert 1 <= rating <= 10, "Rating should be between 1 and 10"
                    
                    results.append((test_persona.scratch.name, thought, rating))
            finally:
                # Restore original function
                chat_completion_request = original_func
            
            # Success output
            print("✓ Test passed:")
            for persona_name, thought, rating in results:
                significance_level = "Low" if rating <= 3 else "Medium" if rating <= 7 else "High"
                print(f"  • {persona_name} - Thought: {thought[:30]}...")
                print(f"    Rating: {rating}/10 ({significance_level})")
                
            return True
            
        except AssertionError as e:
            print(f"✗ Test failed: {str(e)}")
            return False
        except Exception as e:
            print(f"✗ Test failed with unexpected error: {str(e)}")
            return False
           
    # Run the tests
    # test_get_random_alphanumeric()
    # print("########################################################")
    # test_run_gpt_prompt_wake_up_hour()
    # print("########################################################")
    # test_daily_plan()
    # print("########################################################")
    # test_run_gpt_prompt_generate_hourly_schedule()
    # print("########################################################")
    # test_task_decomp()
    # print("########################################################")
    # test_action_sector()
    # print("########################################################")
    # test_action_arena()
    # print("########################################################")
    # test_action_game_object()
    # print("########################################################")
    # test_pronunciation()
    # print("########################################################")
    # test_event_triple()
    # print("########################################################")
    # test_act_obj_desc()
    # print("########################################################")
    # test_act_obj_event_triple()
    # print("########################################################")
    # test_new_decomp_schedule()
    # print("########################################################")
    # test_decide_to_talk()
    # print("########################################################")
    # test_decide_to_react()
    # print("########################################################")
    # test_create_conversation()
    # print("########################################################")
    # test_summarize_conversation()
    # print("########################################################")
    # test_extract_keywords()
    # print("########################################################")
    # test_keyword_to_thoughts()
    # print("########################################################")
    # test_convo_to_thoughts()
    # print("########################################################")
    test_event_poignancy()
    print("########################################################")
    test_thought_poignancy()