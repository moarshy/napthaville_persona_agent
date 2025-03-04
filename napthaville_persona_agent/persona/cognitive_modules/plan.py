import datetime
import math
import random
import logging
from typing import Dict, List, Tuple, Optional, Union
from napthaville_persona_agent.persona.prompts.gpt_structure import get_embedding, chat_completion_request
from napthaville_persona_agent.persona.prompts.run_gpt_prompt import (
    run_gpt_prompt_wake_up_hour,
    run_gpt_prompt_daily_plan,
    run_gpt_prompt_generate_hourly_schedule,
    run_gpt_prompt_task_decomp,
    run_gpt_prompt_action_sector,
    run_gpt_prompt_action_arena,
    run_gpt_prompt_action_game_object,
    run_gpt_prompt_pronunciation,
    run_gpt_prompt_event_triple,
    run_gpt_prompt_act_obj_desc,
    run_gpt_prompt_act_obj_event_triple,
    run_gpt_prompt_summarize_conversation,
    run_gpt_prompt_decide_to_talk,
    run_gpt_prompt_decide_to_react,
    run_gpt_prompt_new_decomp_schedule
)
from napthaville_persona_agent.persona.cognitive_modules.retrieve import new_retrieve
from napthaville_persona_agent.persona.cognitive_modules.converse import agent_chat_v2

from napthaville_chat.run import run as chat_run

# Configure logging
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG = False

##############################################################################
# CHAPTER 1: Schedule Generation Functions
##############################################################################

def generate_wake_up_hour(persona) -> int:
    """
    Generates the time when the persona wakes up.
    
    This becomes an integral part of the process for generating the
    persona's daily plan.
    
    Args:
        persona: The Persona class instance
        
    Returns:
        int: Hour when the persona wakes up (0-23)
    
    Example:
        8
    """
    return int(run_gpt_prompt_wake_up_hour(persona)[0])


def generate_first_daily_plan(persona, wake_up_hour: int) -> List[str]:
    """
    Generates the daily plan for the persona.
    
    Creates a long-term plan spanning a day. Returns a list of actions
    that the persona will take today.
    
    Args:
        persona: The Persona class instance
        wake_up_hour: Hour the persona wakes up (0-23)
        
    Returns:
        List[str]: Daily actions in broad strokes
    
    Example:
        ['wake up and complete the morning routine at 6:00 am', 
         'have breakfast and brush teeth at 6:30 am',
         'work on painting project from 8:00 am to 12:00 pm', 
         'have lunch at 12:00 pm', ...]
    """
    return run_gpt_prompt_daily_plan(persona, wake_up_hour)[0]


def generate_hourly_schedule(persona, wake_up_hour: int) -> List[List[Union[str, int]]]:
    """
    Creates an hourly schedule list based on the persona's daily plan.
    Each element is [activity, duration_in_minutes].
    """
    hour_str = [
        "00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
        "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
        "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
        "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
        "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"
    ]

    n_m1_activity = []
    diversity_repeat_count = 3

    for _ in range(diversity_repeat_count):
        if len(set(n_m1_activity)) >= 5:
            break

        n_m1_activity = []
        remaining_wake_up_hour = wake_up_hour

        for curr_hour_str in hour_str:
            if remaining_wake_up_hour > 0:
                n_m1_activity.append("sleeping")
                remaining_wake_up_hour -= 1
            else:
                activity, _ = run_gpt_prompt_generate_hourly_schedule(
                    persona, curr_hour_str, n_m1_activity, hour_str
                )
                n_m1_activity.append(activity)

    # Compress consecutive identical activities and convert hours to minutes
    compressed = []
    prev = None
    count = 0

    for act in n_m1_activity:
        if act != prev:
            if prev is not None:
                compressed.append([prev, count * 60])
            prev = act
            count = 1
        else:
            count += 1

    if prev is not None:
        compressed.append([prev, count * 60])

    return compressed


def generate_task_decomp(persona, task: str, duration: int) -> List[List[Union[str, int]]]:
    """
    Decomposes a task into smaller subtasks.
    
    Args:
        persona: The Persona class instance
        task: Description of the task (e.g., "waking up and starting her morning routine")
        duration: Number of minutes the task is meant to last
        
    Returns:
        List[List[Union[str, int]]]: List of subtasks and their durations
    
    Example:
        [['going to the bathroom', 5], ['getting dressed', 5], 
         ['eating breakfast', 15], ...]
    """
    return run_gpt_prompt_task_decomp(persona, task, duration)[0]


##############################################################################
# CHAPTER 2: Action Generation Functions
##############################################################################

def generate_action_sector(act_desp: str, persona, maze) -> str:
    """
    Determines the sector where an action will take place.
    
    Args:
        act_desp: Description of the action (e.g., "sleeping")
        persona: The Persona class instance
        maze: The current Maze instance
        
    Returns:
        str: Sector identifier (e.g., "residential")
    """
    return run_gpt_prompt_action_sector(act_desp, persona, maze)[0]


def generate_action_arena(act_desp: str, persona, maze, act_world: str, act_sector: str) -> str:
    """
    Determines the arena (specific location) where an action will take place.
    
    Args:
        act_desp: Description of the action (e.g., "sleeping")
        persona: The Persona class instance
        maze: The current Maze instance
        act_world: World identifier
        act_sector: Sector identifier
        
    Returns:
        str: Arena identifier (e.g., "bedroom 2")
    """
    return run_gpt_prompt_action_arena(act_desp, persona, maze, act_world, act_sector)[0]


def generate_action_game_object(act_desp: str, act_address: str, persona, maze_data) -> str:
    """
    Determines the game object to interact with during an action.
    
    Args:
        act_desp: Description of the action (e.g., "sleeping")
        act_address: Address where the action will take place
        persona: The Persona class instance
        maze_data: The maze data
        
    Returns:
        str: Game object identifier (e.g., "bed")
    """
    # If no objects are accessible in this arena, return random
    if not persona.s_mem.get_str_accessible_arena_game_objects(act_address):
        return "<random>"
    
    return run_gpt_prompt_action_game_object(act_desp, persona, maze_data, act_address)[0]


def generate_action_pronunciatio(act_desp: str, persona) -> str:
    """
    Creates an emoji representation of an action.
    
    Args:
        act_desp: Description of the action (e.g., "sleeping")
        persona: The Persona class instance
        
    Returns:
        str: Emoji string representing the action (e.g., "💤")
    """
    try:
        emoji = run_gpt_prompt_pronunciation(act_desp, persona)[0]
        # Return default emoji if result is empty
        return emoji if emoji else "🙂"
    except Exception:
        # Return default emoji on error
        return "🙂"


def generate_action_event_triple(act_desp: str, persona) -> Tuple[str, str, str]:
    """
    Generates a subject-predicate-object triple representing an action.
    
    Args:
        act_desp: Description of the action (e.g., "sleeping")
        persona: The Persona class instance
        
    Returns:
        Tuple[str, str, str]: Subject, predicate, object triple
    """
    return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_act_obj_desc(act_game_object: str, act_desp: str, persona) -> str:
    """
    Generates a description for how an object is being used.
    
    Args:
        act_game_object: The game object being used
        act_desp: Description of the action
        persona: The Persona class instance
        
    Returns:
        str: Description of object usage
    """
    return run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona)[0]


def generate_act_obj_event_triple(act_game_object: str, act_obj_desc: str, persona) -> Tuple[str, str, str]:
    """
    Generates a subject-predicate-object triple for an object being used.
    
    Args:
        act_game_object: The game object being used
        act_obj_desc: Description of how the object is used
        persona: The Persona class instance
        
    Returns:
        Tuple[str, str, str]: Subject, predicate, object triple
    """
    return run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona)[0]


##############################################################################
# CHAPTER 3: Conversation Functions
##############################################################################

def generate_convo(maze, init_persona, target_persona) -> Tuple[List, int]:
    """
    Generates a conversation between two personas.
    
    Args:
        maze: The current Maze instance
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        
    Returns:
        Tuple[List, int]: Conversation utterances and duration in minutes
    """ 
    # Generate conversation using the most recent version
    convo = agent_chat_v2(maze, init_persona, target_persona)
    
    # Format conversation for processing
    all_utt = "\n".join(f"{speaker}: {utt}" for speaker, utt in convo)
    
    # Calculate conversation length in minutes based on text length
    convo_length = math.ceil(int(len(all_utt) / 8) / 30)
    
    return convo, convo_length


def generate_convo_summary(persona, convo: List) -> str:
    """
    Generates a summary of a conversation.
    
    Args:
        persona: The Persona class instance
        convo: List of conversation utterances
        
    Returns:
        str: Summary of the conversation
    """
    return run_gpt_prompt_summarize_conversation(persona, convo)[0]


def generate_decide_to_talk(init_persona, target_persona, retrieved) -> bool:
    """
    Determines if a persona decides to talk to another persona.
    
    Args:
        init_persona: The persona potentially initiating conversation
        target_persona: The potential conversation target
        retrieved: Retrieved memory context
        
    Returns:
        bool: True if persona decides to talk, False otherwise
    """
    decision = run_gpt_prompt_decide_to_talk(init_persona, target_persona, retrieved)[0]
    return decision.lower() == "yes"


def generate_decide_to_react(init_persona, target_persona, retrieved) -> str:
    """
    Determines how a persona reacts to another persona.
    
    Args:
        init_persona: The persona potentially reacting
        target_persona: The persona being reacted to
        retrieved: Retrieved memory context
        
    Returns:
        str: Reaction mode (e.g., "1" for wait, "2" for do other things)
    """
    return run_gpt_prompt_decide_to_react(init_persona, target_persona, retrieved)[0]


def generate_new_decomp_schedule(
    persona, 
    inserted_act, 
    inserted_act_dur,  
    start_hour, 
    end_hour
): 
    # Step 1: Setting up the core variables for the function. 
    # <p> is the persona whose schedule we are editing right now. 
    p = persona
    # <today_min_pass> indicates the number of minutes that have passed today. 
    today_min_pass = (int(p.scratch.curr_time.hour) * 60 
                        + int(p.scratch.curr_time.minute) + 1)
  
    # Step 2: We need to create <main_act_dur> and <truncated_act_dur>. 
    # These are basically a sub-component of <f_daily_schedule> of the persona,
    # but focusing on the current decomposition. 
    # Here is an example for <main_act_dur>: 
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (uses the restroom)', 5]
    # ['wakes up and completes her morning routine (washes her ...)', 10]
    # ['wakes up and completes her morning routine (makes her bed)', 5]
    # ['wakes up and completes her morning routine (eats breakfast)', 15]
    # ['wakes up and completes her morning routine (gets dressed)', 10]
    # ['wakes up and completes her morning routine (leaves her ...)', 5]
    # ['wakes up and completes her morning routine (starts her ...)', 5]
    # ['preparing for her day (waking up at 6am)', 5]
    # ['preparing for her day (making her bed)', 5]
    # ['preparing for her day (taking a shower)', 15]
    # ['preparing for her day (getting dressed)', 5]
    # ['preparing for her day (eating breakfast)', 10]
    # ['preparing for her day (brushing her teeth)', 5]
    # ['preparing for her day (making coffee)', 5]
    # ['preparing for her day (checking her email)', 5]
    # ['preparing for her day (starting to work on her painting)', 5]
    # 
    # And <truncated_act_dur> concerns only until where an event happens. 
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (wakes up at 6am)', 2]
    main_act_dur = []
    truncated_act_dur = []
    dur_sum = 0 # duration sum
    count = 0 # enumerate count
    truncated_fin = False 

    print ("DEBUG::: ", persona.scratch.name)
    for act, dur in p.scratch.f_daily_schedule: 
        if (dur_sum >= start_hour * 60) and (dur_sum < end_hour * 60): 
            main_act_dur += [[act, dur]]
        if dur_sum <= today_min_pass:
            truncated_act_dur += [[act, dur]]
        elif dur_sum > today_min_pass and not truncated_fin: 
            # We need to insert that last act, duration list like this one: 
            # e.g., ['wakes up and completes her morning routine (wakes up...)', 2]
            truncated_act_dur += [[p.scratch.f_daily_schedule[count][0], 
                                dur_sum - today_min_pass]] 
            truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
            # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass + 1) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
            print ("DEBUG::: ", truncated_act_dur)

            # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
            truncated_fin = True
        dur_sum += dur
        count += 1

    persona_name = persona.name 
    main_act_dur = main_act_dur

    x = truncated_act_dur[-1][0].split("(")[0].strip() + " (on the way to " + truncated_act_dur[-1][0].split("(")[-1][:-1] + ")"
    truncated_act_dur[-1][0] = x 

    if "(" in truncated_act_dur[-1][0]: 
        inserted_act = truncated_act_dur[-1][0].split("(")[0].strip() + " (" + inserted_act + ")"

    # To do inserted_act_dur+1 below is an important decision but I'm not sure
    # if I understand the full extent of its implications. Might want to 
    # revisit. 
    truncated_act_dur += [[inserted_act, inserted_act_dur]]
    start_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                    + datetime.timedelta(hours=start_hour))
    end_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                    + datetime.timedelta(hours=end_hour))

    return run_gpt_prompt_new_decomp_schedule(persona, 
                                                main_act_dur, 
                                                truncated_act_dur, 
                                                start_time_hour,
                                                end_time_hour,
                                                inserted_act,
                                                inserted_act_dur)[0]


##############################################################################
# CHAPTER 4: Identity and Planning Functions
##############################################################################

def revise_identity(persona):
    """
    Revises the persona's identity based on recent events and memories.
    
    Updates the persona's current status and daily plan based on
    retrieved memories and reflections.
    
    Args:
        persona: The Persona class instance
    """
    p_name = persona.scratch.name
    
    # Retrieve relevant memories
    focal_points = [
        f"{p_name}'s plan for {persona.scratch.get_str_curr_date_str()}.",
        f"Important recent events for {p_name}'s life."
    ]
    retrieved = new_retrieve(persona, focal_points)
    
    # Format retrieved statements
    statements = "[Statements]\n"
    for key, val in retrieved.items():
        for memory in val:
            statements += f"{memory.created.strftime('%A %B %d -- %H:%M %p')}: {memory.embedding_key}\n"
    
    # Generate planning note
    plan_prompt = (
        f"{statements}\n"
        f"Given the statements above, is there anything that {p_name} should remember as they plan for"
        f" *{persona.scratch.curr_time.strftime('%A %B %d')}*? "
        f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
        f"Write the response from {p_name}'s perspective."
    )
    plan_note = chat_completion_request(plan_prompt)
    
    # Generate thought note about feelings
    thought_prompt = (
        f"{statements}\n"
        f"Given the statements above, how might we summarize {p_name}'s feelings about their days up to now?\n\n"
        f"Write the response from {p_name}'s perspective."
    )
    thought_note = chat_completion_request(thought_prompt)
    
    # Update persona's current status
    currently_prompt = (
        f"{p_name}'s status from {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
        f"{persona.scratch.currently}\n\n"
        f"{p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
        f"{(plan_note + thought_note).replace(chr(10), '')}\n\n"
        f"It is now {persona.scratch.curr_time.strftime('%A %B %d')}. Given the above, write {p_name}'s status for {persona.scratch.curr_time.strftime('%A %B %d')} "
        f"that reflects {p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {p_name}."
        f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
        f"Follow this format below:\nStatus: <new status>"
    )
    new_currently = chat_completion_request(currently_prompt)
    persona.scratch.currently = new_currently
    
    # Update daily plan requirements
    daily_req_prompt = (
        f"{persona.scratch.get_str_iss()}\n"
        f"Today is {persona.scratch.curr_time.strftime('%A %B %d')}. Here is {persona.scratch.name}'s plan today "
        f"in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
        f"Follow this format (the list should have 4~6 items but no more):\n"
        f"1. wake up and complete the morning routine at <time>, 2. ..."
    )
    new_daily_req = chat_completion_request(daily_req_prompt).replace(chr(10), ' ')
    persona.scratch.daily_plan_req = new_daily_req
    logger.info(f"Updated daily plan: {new_daily_req}")
    return persona


##############################################################################
# CHAPTER 5: Core Planning Functions
##############################################################################

def _long_term_planning(persona, new_day: str):
    """
    Formulates the persona's daily long-term plan.
    
    Creates wake-up hour and hourly schedule based on the persona's routine.
    Adds the plan to memory for future reference.
    
    Args:
        persona: The Persona class instance
        new_day: "First day" or "New day" indicator
    """
    # Generate wake up hour
    wake_up_hour = generate_wake_up_hour(persona)
    print(f"Wake up hour: {wake_up_hour}")
    
    # Create daily plan based on day type
    if new_day == "First day":
        # Initial plan for first day of simulation
        persona.scratch.daily_req = generate_first_daily_plan(persona, wake_up_hour)
    elif new_day == "New day":
        # Update identity and plan for subsequent days
        revise_identity(persona)
        # Preserve existing daily requirements
        persona.scratch.daily_req = persona.scratch.daily_req
    
    # Generate hourly schedule based on daily requirements
    persona.scratch.f_daily_schedule = generate_hourly_schedule(persona, wake_up_hour)
    persona.scratch.f_daily_schedule_hourly_org = persona.scratch.f_daily_schedule[:]
    
    # Add plan to memory
    thought = f"This is {persona.scratch.name}'s plan for {persona.scratch.curr_time.strftime('%A %B %d')}:"
    thought += ", ".join(f" {item}" for item in persona.scratch.daily_req) + "."
    
    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = (persona.scratch.name, "plan", persona.scratch.curr_time.strftime('%A %B %d'))
    keywords = set(["plan"])
    thought_poignancy = 5
    thought_embedding_pair = (thought, get_embedding(thought))
    
    persona.a_mem.add_thought(
        created, expiration, s, p, o, 
        thought, keywords, thought_poignancy, 
        thought_embedding_pair, None
    )
    return persona

def _determine_action(persona, maze_data):
    """
    Creates the next action sequence for the persona.
    
    Decomposes tasks as needed and sets up action-related variables.
    
    Args:
        persona: The Persona class instance
        maze_data: The current maze data
    """
    def determine_decomp(act_desp: str, act_dura: int) -> bool:
        """
        Determines whether an action needs to be decomposed.
        
        Args:
            act_desp: Description of the action
            act_dura: Duration of the action in minutes
            
        Returns:
            bool: True if action should be decomposed, False otherwise
        """
        # Don't decompose sleeping activities that are longer than an hour
        if "sleep" not in act_desp and "bed" not in act_desp:
            return True
        elif "sleeping" in act_desp or "asleep" in act_desp or "in bed" in act_desp:
            return False
        elif "sleep" in act_desp or "bed" in act_desp:
            if act_dura > 60:
                return False
        return True
    
    # Get current schedule index
    curr_index = persona.scratch.get_f_daily_schedule_index()
    curr_index_60 = persona.scratch.get_f_daily_schedule_index(advance=60)
    
    # Decompose activities for first hour of the day
    if curr_index == 0:
        act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
        if act_dura >= 60 and determine_decomp(act_desp, act_dura):
            persona.scratch.f_daily_schedule[curr_index:curr_index+1] = (
                generate_task_decomp(persona, act_desp, act_dura)
            )
            
        if curr_index_60 + 1 < len(persona.scratch.f_daily_schedule):
            act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60+1]
            if act_dura >= 60 and determine_decomp(act_desp, act_dura):
                persona.scratch.f_daily_schedule[curr_index_60+1:curr_index_60+2] = (
                    generate_task_decomp(persona, act_desp, act_dura)
                )
    
    # Decompose activities for current hour (if not first hour)
    if curr_index_60 < len(persona.scratch.f_daily_schedule) and persona.scratch.curr_time.hour < 23:
        act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60]
        if act_dura >= 60 and determine_decomp(act_desp, act_dura):
            persona.scratch.f_daily_schedule[curr_index_60:curr_index_60+1] = (
                generate_task_decomp(persona, act_desp, act_dura)
            )
    
    # Fill in any schedule gaps (ensure total is 1440 minutes)
    total_minutes = sum(duration for _, duration in persona.scratch.f_daily_schedule)
    if total_minutes < 1440:
        logger.info(f"Adding {1440 - total_minutes} minutes of sleep to complete schedule")
        persona.scratch.f_daily_schedule += [["sleeping", 1440 - total_minutes]]
    
    # Get current action
    act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
    
    # Generate location and object details
    act_world = maze_data["curr_tile_data"]["world"]
    act_sector = generate_action_sector(act_desp, persona, maze_data)
    act_arena = generate_action_arena(act_desp, persona, maze_data, act_world, act_sector)
    act_address = f"{act_world}:{act_sector}:{act_arena}"
    act_game_object = generate_action_game_object(act_desp, act_address, persona, maze_data)
    new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
    
    # Generate action details
    act_pron = generate_action_pronunciatio(act_desp, persona)
    act_event = generate_action_event_triple(act_desp, persona)
    
    # Generate object interaction details
    act_obj_desp = generate_act_obj_desc(act_game_object, act_desp, persona)
    act_obj_pron = generate_action_pronunciatio(act_obj_desp, persona)
    act_obj_event = generate_act_obj_event_triple(act_game_object, act_obj_desp, persona)
    
    # Add action to persona's queue
    persona.scratch.add_new_action(
        new_address, int(act_dura), act_desp, act_pron, act_event,
        None, None, None, None,
        act_obj_desp, act_obj_pron, act_obj_event
    )
    return persona

def _choose_retrieved(persona, retrieved: Dict) -> Optional[Dict]:
    """
    Selects an event to focus on from retrieved memories.
    
    Args:
        persona: The Persona class instance
        retrieved: Dictionary of retrieved memories
        
    Returns:
        Optional[Dict]: Selected memory context or None if nothing relevant
    """
    # Remove self events
    copy_retrieved = retrieved.copy()
    for event_desc, rel_ctx in copy_retrieved.items():
        curr_event = rel_ctx["curr_event"]
        if curr_event.subject == persona.name:
            del retrieved[event_desc]
    
    if not retrieved:
        return None
    
    # Prioritize persona events
    priority = []
    for event_desc, rel_ctx in retrieved.items():
        curr_event = rel_ctx["curr_event"]
        if ":" not in curr_event.subject and curr_event.subject != persona.name:
            priority.append(rel_ctx)
            
    if priority:
        return random.choice(priority)
    
    # Skip idle events
    priority = []
    for event_desc, rel_ctx in retrieved.items():
        if "is idle" not in event_desc:
            priority.append(rel_ctx)
            
    if priority:
        return random.choice(priority)
        
    return None


def _should_react(persona, retrieved: Dict, personas: Dict) -> Union[str, bool]:
    """
    Determines how a persona should react to a retrieved event.
    
    Args:
        persona: The Persona class instance
        retrieved: Dictionary of retrieved context
        personas: Dictionary of all personas' scratch
        
    Returns:
        Union[str, bool]: Reaction mode or False if no reaction
    """
    def lets_talk(init_persona, target_persona_name, target_persona, retrieved) -> bool:
        """
        Determines if two personas should engage in conversation.
        
        Args:
            init_persona: Initiating persona
            target_persona: Target persona
            retrieved: Retrieved memory context
            
        Returns:
            bool: True if personas should talk, False otherwise
        """
        # Check if personas are available for conversation
        if (not target_persona["act_address"] or
            not target_persona["act_description"] or
            not init_persona.scratch.act_address or
            not init_persona.scratch.act_description):
            return False
        
        # Check if either persona is sleeping
        if ("sleeping" in target_persona["act_description"] or
            "sleeping" in init_persona.scratch.act_description):
            return False
        
        # Don't start conversations late at night
        if init_persona.scratch.curr_time.hour == 23:
            return False
        
        # Don't talk to waiting personas
        if "<waiting>" in target_persona["act_address"]:
            return False
        
        # Don't talk if either persona is already in conversation
        if (target_persona["chatting_with"] or
            init_persona.scratch.chatting_with):
            return False
        
        # Check conversation cooldown period
        if target_persona_name in init_persona.scratch.chatting_with_buffer:
            if init_persona.scratch.chatting_with_buffer[target_persona_name] > 0:
                return False
        
        # Use LLM to decide whether to talk
        return generate_decide_to_talk(init_persona, target_persona, retrieved)

    def lets_react(init_persona, target_persona_name, target_persona, retrieved) -> Union[str, bool]:
        """
        Determines if and how a persona should react to another persona.
        
        Args:
            init_persona: Reacting persona
            target_persona: Persona being reacted to
            retrieved: Retrieved memory context
            
        Returns:
            Union[str, bool]: Reaction mode or False if no reaction
        """
        # Check if personas are available for interaction
        if (not target_persona["act_address"] or
            not target_persona["act_description"] or
            not init_persona.scratch.act_address or
            not init_persona.scratch.act_description):
            return False
        
        # Check if either persona is sleeping
        if ("sleeping" in target_persona["act_description"] or
            "sleeping" in init_persona.scratch.act_description):
            return False
        
        # Don't react late at night
        if init_persona.scratch.curr_time.hour == 23:
            return False
        
        # Don't react to waiting personas
        if "waiting" in target_persona["act_description"]:
            return False
            
        # Don't react if no planned path
        if init_persona.scratch.planned_path == []:
            return False
        
        # Only react if in same location
        if init_persona.scratch.act_address != target_persona["act_address"]:
            return False
        
        # Get reaction mode from LLM
        react_mode = generate_decide_to_react(init_persona, target_persona, retrieved)
        
        if react_mode == "1":
            # Wait until target persona finishes their activity
            wait_until = (
                (target_persona["act_start_time"] + 
                datetime.timedelta(minutes=target_persona["scratch"]["act_duration"] - 1))
                .strftime("%B %d, %Y, %H:%M:%S")
            )
            return f"wait: {wait_until}"
        elif react_mode == "2":
            # Do other things
            return False
        else:
            # Keep current plan
            return False
    
    # If the persona is already chatting, don't react
    if persona.scratch.chatting_with:
        return False
        
    # If persona is in waiting state, don't react
    if "<waiting>" in persona.scratch.act_address:
        return False
    
    # Get the current event from retrieved context
    curr_event = retrieved["curr_event"]
    
    # Handle persona events (events where subject is another persona)
    if ":" not in curr_event.subject:
        # Check if personas should talk
        if lets_talk(
            persona, 
            curr_event.subject, 
            personas[curr_event.subject], 
            retrieved
        ):
            return f"chat with {curr_event.subject}"
            
        # Check if persona should otherwise react
        react_mode = lets_react(
            persona, 
            curr_event.subject, 
            personas[curr_event.subject], 
            retrieved
        )
        return react_mode
        
    # No reaction for non-persona events
    return False


async def plan(persona, maze_data, personas: Dict, new_day: Union[str, bool], retrieved: Dict) -> str:
    """
    Main cognitive function for planning persona actions.
    
    Takes retrieved memory and perception context to conduct both
    long-term and short-term planning for the persona.
    
    Args:
        persona: The Persona class instance
        maze_data: Current Maze instance of the world
        personas: Dictionary of persona names to Persona instances
        new_day: Indicates if this is a new day cycle
            False: Not a new day
            "First day": Start of simulation
            "New day": Beginning of a new day
        retrieved: Dictionary of retrieved memories and relevant context
        
    Returns:
        str: Target action address of the persona
    """
    # PART 1: Generate the hourly schedule if it's a new day
    if new_day:
        _long_term_planning(persona, new_day)

    # PART 2: Create a new action if current one has finished
    if persona.scratch.act_check_finished():
        _determine_action(persona, maze_data)

    # PART 3: Process perceived events that may require response
    focused_event = False
    if retrieved:
        # Choose which event to focus on from retrieved memories
        focused_event = _choose_retrieved(persona, retrieved)
    
    # Determine how to react to the focused event
    if focused_event:
        reaction_mode = _should_react(persona, focused_event, personas)
        if reaction_mode:
            # Handle different reaction types
            # if reaction_mode.startswith("chat with"):
            #     _chat_react(maze_data, persona, focused_event, reaction_mode, personas)
            # elif reaction_mode.startswith("wait"):
            #     _wait_react(persona, reaction_mode)
            chat_input_data = {
                "init_persona_name": persona.name,
                "target_persona_name": focused_event["curr_event"].subject,
                "reaction_mode": reaction_mode,
                "maze_data": maze_data
            }
            await chat_run(chat_input_data)

            # Update the persona's memory
            await persona.load_memory()

    # Chat-related state cleanup
    if persona.scratch.act_event[1] != "chat with":
        persona.scratch.chatting_with = None
        persona.scratch.chat = None
        persona.scratch.chatting_end_time = None
    
    # Update conversation cooldown timers
    curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
    for persona_name, buffer_count in list(curr_persona_chat_buffer.items()):
        if persona_name != persona.scratch.chatting_with:
            persona.scratch.chatting_with_buffer[persona_name] = max(0, buffer_count - 1)
    
    return persona.scratch.act_address


def _chat_react(maze, persona, focused_event: Dict, reaction_mode: str, personas: Dict):
    """
    Creates a conversation reaction between two personas.
    
    Generates a conversation, summarizes it, and updates both personas'
    schedules to reflect the conversation.
    
    Args:
        maze: The current Maze instance
        persona: The initiating persona
        focused_event: Retrieved context driving the reaction
        reaction_mode: The reaction mode (must be "chat with [name]")
        personas: Dictionary of all personas
    """
    # Get the personas involved in conversation
    init_persona = persona
    target_persona = personas[reaction_mode[9:].strip()]
    
    # Generate the conversation and its duration
    convo, duration_min = generate_convo(maze, init_persona, target_persona)
    convo_summary = generate_convo_summary(init_persona, convo)
    
    # Set up conversation parameters
    inserted_act = convo_summary
    inserted_act_dur = duration_min
    act_start_time = target_persona.scratch.act_start_time
    
    # Calculate when the conversation will end
    curr_time = target_persona.scratch.curr_time
    if curr_time.second != 0:
        temp_curr_time = curr_time + datetime.timedelta(seconds=60 - curr_time.second)
        chatting_end_time = temp_curr_time + datetime.timedelta(minutes=inserted_act_dur)
    else:
        chatting_end_time = curr_time + datetime.timedelta(minutes=inserted_act_dur)
    
    # Update both personas' schedules
    for role, p in [("init", init_persona), ("target", target_persona)]:
        if role == "init":
            # Set up initiating persona's conversation
            act_address = f"<persona> {target_persona.name}"
            act_event = (p.name, "chat with", target_persona.name)
            chatting_with = target_persona.name
            chatting_with_buffer = {}
            chatting_with_buffer[target_persona.name] = 800  # Cooldown for next conversation
        else:
            # Set up target persona's conversation
            act_address = f"<persona> {init_persona.name}"
            act_event = (p.name, "chat with", init_persona.name)
            chatting_with = init_persona.name
            chatting_with_buffer = {}
            chatting_with_buffer[init_persona.name] = 800  # Cooldown for next conversation
        
        # Common parameters for both personas
        act_pronunciatio = "💬"
        act_obj_description = None
        act_obj_pronunciatio = None
        act_obj_event = (None, None, None)
        
        # Create reaction action and update schedule
        _create_react(
            p, inserted_act, inserted_act_dur,
            act_address, act_event, chatting_with, convo, chatting_with_buffer, 
            chatting_end_time, act_pronunciatio, act_obj_description, 
            act_obj_pronunciatio, act_obj_event, act_start_time
        )


def _wait_react(persona, reaction_mode: str):
    """
    Creates a waiting reaction for a persona.
    
    Updates the persona's schedule to wait until a specific time
    before continuing with their original activity.
    
    Args:
        persona: The persona that will wait
        reaction_mode: The waiting instruction (format: "wait: [datetime]")
    """
    p = persona
    
    # Parse the original activity being waited for
    inserted_act = f'waiting to start {p.scratch.act_description.split("(")[-1][:-1]}'
    
    # Parse end time from reaction mode
    end_time = datetime.datetime.strptime(reaction_mode[6:].strip(), "%B %d, %Y, %H:%M:%S")
    
    # Calculate duration in minutes
    current_time_mins = p.scratch.curr_time.minute + p.scratch.curr_time.hour * 60
    end_time_mins = end_time.minute + end_time.hour * 60
    inserted_act_dur = end_time_mins - current_time_mins + 1
    
    # Ensure positive duration (handle day boundary cases)
    if inserted_act_dur <= 0:
        inserted_act_dur = 1  # Minimum wait of 1 minute
    
    # Set up waiting parameters
    act_address = f"<waiting> {p.scratch.curr_tile[0]} {p.scratch.curr_tile[1]}"
    act_event = (p.name, "waiting to start", p.scratch.act_description.split("(")[-1][:-1])
    act_pronunciatio = "⌛"
    
    # Create reaction action and update schedule
    _create_react(
        p, inserted_act, inserted_act_dur,
        act_address, act_event, None, None, None, None,
        act_pronunciatio, None, None, (None, None, None)
    )


def _create_react(persona, inserted_act: str, inserted_act_dur: int,
                 act_address: str, act_event: Tuple[str, str, str],
                 chatting_with: Optional[str], chat: Optional[List],
                 chatting_with_buffer: Optional[Dict[str, int]],
                 chatting_end_time: Optional[datetime.datetime], 
                 act_pronunciatio: str, act_obj_description: Optional[str],
                 act_obj_pronunciatio: Optional[str], 
                 act_obj_event: Tuple[Optional[str], Optional[str], Optional[str]],
                 act_start_time: Optional[datetime.datetime] = None):
    """
    Creates a reaction action for a persona and updates their schedule.
    
    Args:
        persona: The persona creating the reaction
        inserted_act: Description of the activity to insert
        inserted_act_dur: Duration of activity in minutes
        act_address: Address where activity takes place
        act_event: Subject-predicate-object triple for the activity
        chatting_with: Name of persona being chatted with (if any)
        chat: List of chat utterances (if any)
        chatting_with_buffer: Cooldown timer for future chats
        chatting_end_time: When the conversation will end
        act_pronunciatio: Emoji representation of activity
        act_obj_description: Description of object interaction
        act_obj_pronunciatio: Emoji for object interaction
        act_obj_event: Triple for object interaction
        act_start_time: When the activity starts
    """
    p = persona 
    
    # Calculate schedule position for insertion
    min_sum = 0
    for i in range(p.scratch.get_f_daily_schedule_hourly_org_index()): 
        min_sum += p.scratch.f_daily_schedule_hourly_org[i][1]
    start_hour = int(min_sum / 60)
    
    # Determine end hour for scheduling block
    current_activity_duration = p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1]
    
    if current_activity_duration >= 120:
        # If current activity is long, use just that activity
        end_hour = start_hour + current_activity_duration / 60
    elif (p.scratch.get_f_daily_schedule_hourly_org_index() + 1 < len(p.scratch.f_daily_schedule_hourly_org)):
        # Use current and next activity if current is short
        next_activity_duration = p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index() + 1][1]
        end_hour = start_hour + ((current_activity_duration + next_activity_duration) / 60)
    else:
        # Fallback if at end of schedule
        end_hour = start_hour + 2
        
    end_hour = int(end_hour)
    
    # Find indices in schedule for replacement
    dur_sum = 0
    count = 0 
    start_index = None
    end_index = None
    
    for act, dur in p.scratch.f_daily_schedule:
        if dur_sum >= start_hour * 60 and start_index is None:
            start_index = count
        if dur_sum >= end_hour * 60 and end_index is None: 
            end_index = count
        dur_sum += dur
        count += 1
    
    # If we couldn't find valid indices, use safe defaults
    if start_index is None:
        start_index = 0
    if end_index is None or end_index <= start_index:
        end_index = start_index + 1
    
    # Generate new schedule with inserted activity
    ret = generate_new_decomp_schedule(p, inserted_act, inserted_act_dur, 
                                      start_hour, end_hour)
    
    # Update schedule and add new action
    p.scratch.f_daily_schedule[start_index:end_index] = ret
    p.scratch.add_new_action(
        act_address,
        inserted_act_dur,
        inserted_act,
        act_pronunciatio,
        act_event,
        chatting_with,
        chat,
        chatting_with_buffer,
        chatting_end_time,
        act_obj_description,
        act_obj_pronunciatio,
        act_obj_event,
        act_start_time
    )