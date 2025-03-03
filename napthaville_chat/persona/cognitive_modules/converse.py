import datetime
from napthaville_chat.persona.prompts.gpt_structure import get_embedding
from napthaville_chat.persona.memory.spatial import *
from napthaville_chat.persona.memory.associative_memory import *
from napthaville_chat.persona.memory.scratch import *
from napthaville_chat.persona.cognitive_modules.retrieve import new_retrieve
from napthaville_chat.persona.prompts.run_gpt_prompt import (
    run_gpt_prompt_agent_chat_summarize_ideas,
    run_gpt_prompt_agent_chat_summarize_relationship,
    run_gpt_prompt_agent_chat,
    run_gpt_generate_iterative_chat_utt,
    run_gpt_prompt_summarize_ideas,
    run_gpt_prompt_generate_next_convo_line,
    run_gpt_prompt_generate_whisper_inner_thought,
    run_gpt_prompt_event_triple,
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_chat_poignancy,
)

# Constants
DEBUG = False
MAX_CHAT_TURNS = 8
MAX_RECENT_CHAT_HISTORY = 4
DEFAULT_MEMORY_RETRIEVAL_COUNT = 50
FOCUSED_MEMORY_RETRIEVAL_COUNT = 15
DEFAULT_EXPIRATION_DAYS = 30
SAFETY_THRESHOLD = 8


def generate_agent_chat_summarize_ideas(init_persona, target_persona, retrieved, curr_context):
    """
    Summarize ideas for agent chat based on retrieved memories.
    
    Args:
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        retrieved: Dictionary of retrieved memory nodes
        curr_context: Current context of the conversation
        
    Returns:
        Summarized ideas for the chat
    """
    all_embedding_keys = []
    for key, val in retrieved.items():
        for memory_node in val:
            all_embedding_keys.append(memory_node.embedding_key)
            
    all_embedding_key_str = "\n".join(all_embedding_keys)

    try:
        summarized_idea = run_gpt_prompt_agent_chat_summarize_ideas(
            init_persona,
            target_persona, 
            all_embedding_key_str,
            curr_context
        )[0]
    except Exception as e:
        if DEBUG:
            print(f"Error summarizing ideas: {e}")
        summarized_idea = ""
        
    return summarized_idea


def generate_summarize_agent_relationship(init_persona, target_persona, retrieved):
    """
    Summarize the relationship between two agents based on retrieved memories.
    
    Args:
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        retrieved: Dictionary of retrieved memory nodes
        
    Returns:
        Summarized relationship between the two agents
    """
    all_embedding_keys = []
    for key, val in retrieved.items():
        for memory_node in val:
            all_embedding_keys.append(memory_node.embedding_key)
            
    all_embedding_key_str = "\n".join(all_embedding_keys)

    summarized_relationship = run_gpt_prompt_agent_chat_summarize_relationship(
        init_persona, 
        target_persona,
        all_embedding_key_str
    )[0]
    
    return summarized_relationship


def generate_agent_chat(maze, init_persona, target_persona, curr_context, init_summ_idea, target_summ_idea):
    """
    Generate a complete agent chat conversation.
    
    Args:
        maze: The environment maze
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        curr_context: Current context of the conversation
        init_summ_idea: Summarized ideas from the initiating persona
        target_summ_idea: Summarized ideas from the target persona
        
    Returns:
        Complete agent chat conversation
    """
    summarized_idea = run_gpt_prompt_agent_chat(
        maze,
        init_persona,
        target_persona,
        curr_context,
        init_summ_idea,
        target_summ_idea
    )[0]
    
    if DEBUG:
        for chat_line in summarized_idea:
            print(chat_line)
            
    return summarized_idea


def create_conversation_context(init_persona, target_persona, ongoing=False):
    """
    Create the initial context for a conversation.
    
    Args:
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        ongoing: Whether the conversation is ongoing or just starting
        
    Returns:
        String containing the conversation context
    """
    base_context = (
        f"{init_persona.scratch.name} was {init_persona.scratch.act_description} "
        f"when {init_persona.scratch.name} saw {target_persona.scratch.name} "
        f"in the middle of {target_persona.scratch.act_description}.\n"
    )
    
    if ongoing:
        base_context += (
            f"{init_persona.scratch.name} is continuing a conversation with "
            f"{target_persona.scratch.name}."
        )
    else:
        base_context += (
            f"{init_persona.scratch.name} is thinking of initiating a conversation with "
            f"{target_persona.scratch.name}."
        )
        
    return base_context


def agent_chat_v1(maze, init_persona, target_persona):
    """
    Version 1 of agent chat - optimized for speed via batch generation.
    
    Args:
        maze: The environment maze
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        
    Returns:
        Complete agent chat conversation
    """
    curr_context = create_conversation_context(init_persona, target_persona)
    
    summarized_ideas = []
    participant_pairs = [
        (init_persona, target_persona),
        (target_persona, init_persona)
    ]
    
    for persona_1, persona_2 in participant_pairs:
        # First, retrieve memories about the other persona
        focal_points = [f"{persona_2.scratch.name}"]
        retrieved = new_retrieve(persona_1, focal_points, DEFAULT_MEMORY_RETRIEVAL_COUNT)
        
        # Summarize the relationship between the two personas
        relationship = generate_summarize_agent_relationship(persona_1, persona_2, retrieved)
        
        # Use this relationship info to retrieve more relevant memories
        focal_points = [
            f"{relationship}",
            f"{persona_2.scratch.name} is {persona_2.scratch.act_description}"
        ]
        retrieved = new_retrieve(persona_1, focal_points, FOCUSED_MEMORY_RETRIEVAL_COUNT)
        
        # Summarize ideas for the conversation
        summarized_idea = generate_agent_chat_summarize_ideas(
            persona_1, persona_2, retrieved, curr_context
        )
        summarized_ideas.append(summarized_idea)

    # Generate the complete conversation
    return generate_agent_chat(
        maze, 
        init_persona, 
        target_persona,
        curr_context,
        summarized_ideas[0],
        summarized_ideas[1]
    )


def generate_one_utterance(maze_data, speaker_persona, listener_persona, retrieved, curr_chat):
    """
    Generate a single utterance in a conversation.
    
    Args:
        maze: The environment maze
        speaker_persona: The persona speaking
        listener_persona: The persona listening
        retrieved: Dictionary of retrieved memory nodes
        curr_chat: Current chat history
        
    Returns:
        Tuple of (utterance, end_flag) where end_flag indicates if the conversation should end
    """
    curr_context = create_conversation_context(speaker_persona, listener_persona, ongoing=True)
    
    if DEBUG:
        print("Generating utterance...")
        
    response = run_gpt_generate_iterative_chat_utt(
        maze_data, 
        speaker_persona, 
        listener_persona, 
        retrieved, 
        curr_context, 
        curr_chat
    )[0]

    return response["utterance"], response["end"]


def agent_chat_v2(maze_data, init_persona, target_persona):
    """
    Version 2 of agent chat - generates conversation turn by turn.
    
    Args:
        maze_data: The environment maze data
        init_persona: The persona initiating the conversation
        target_persona: The target persona for the conversation
        
    Returns:
        Complete agent chat conversation as a list of [speaker, utterance] pairs
    """
    curr_chat = []
    
    if DEBUG:
        print("Starting agent chat v2")

    for _ in range(MAX_CHAT_TURNS):
        # First persona's turn
        focal_points = [f"{target_persona.scratch.name}"]
        retrieved = new_retrieve(init_persona, focal_points, DEFAULT_MEMORY_RETRIEVAL_COUNT)
        relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
        
        if DEBUG:
            print(f"Relationship from {init_persona.scratch.name}'s perspective: {relationship}")
        
        # Get recent chat history if available
        if curr_chat[-MAX_RECENT_CHAT_HISTORY:]:
            last_chat = "\n".join([": ".join(chat_turn) for chat_turn in curr_chat[-MAX_RECENT_CHAT_HISTORY:]])
            focal_points = [
                f"{relationship}",
                f"{target_persona.scratch.name} is {target_persona.scratch.act_description}",
                last_chat
            ]
        else:
            focal_points = [
                f"{relationship}",
                f"{target_persona.scratch.name} is {target_persona.scratch.act_description}"
            ]
            
        retrieved = new_retrieve(init_persona, focal_points, FOCUSED_MEMORY_RETRIEVAL_COUNT)
        utt, end = generate_one_utterance(maze_data, init_persona, target_persona, retrieved, curr_chat)

        curr_chat.append([init_persona.scratch.name, utt])
        if end:
            break

        # Second persona's turn
        focal_points = [f"{init_persona.scratch.name}"]
        retrieved = new_retrieve(target_persona, focal_points, DEFAULT_MEMORY_RETRIEVAL_COUNT)
        relationship = generate_summarize_agent_relationship(target_persona, init_persona, retrieved)
        
        if DEBUG:
            print(f"Relationship from {target_persona.scratch.name}'s perspective: {relationship}")
        
        # Get recent chat history if available
        if curr_chat[-MAX_RECENT_CHAT_HISTORY:]:
            last_chat = "\n".join([": ".join(chat_turn) for chat_turn in curr_chat[-MAX_RECENT_CHAT_HISTORY:]])
            focal_points = [
                f"{relationship}",
                f"{init_persona.scratch.name} is {init_persona.scratch.act_description}",
                last_chat
            ]
        else:
            focal_points = [
                f"{relationship}",
                f"{init_persona.scratch.name} is {init_persona.scratch.act_description}"
            ]
            
        retrieved = new_retrieve(target_persona, focal_points, FOCUSED_MEMORY_RETRIEVAL_COUNT)
        utt, end = generate_one_utterance(maze_data, target_persona, init_persona, retrieved, curr_chat)

        curr_chat.append([target_persona.scratch.name, utt])
        if end:
            break

    if DEBUG:
        print("Conversation complete:")
        for row in curr_chat:
            print(row)
            
    return curr_chat


def generate_summarize_ideas(persona, nodes, question):
    """
    Summarize ideas from memory nodes based on a question.
    
    Args:
        persona: The persona whose ideas are being summarized
        nodes: List of memory nodes
        question: The question to focus the summary on
        
    Returns:
        Summarized ideas relevant to the question
    """
    statements = "\n".join([n.embedding_key for n in nodes])
    summarized_idea = run_gpt_prompt_summarize_ideas(persona, statements, question)[0]
    return summarized_idea


def generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea):
    """
    Generate the next line in a conversation.
    
    Args:
        persona: The persona speaking
        interlocutor_desc: Description of the conversation partner
        curr_convo: Current conversation history
        summarized_idea: Summarized ideas to help with response
        
    Returns:
        Next line of dialogue
    """
    prev_convo = "\n".join([f'{row[0]}: {row[1]}' for row in curr_convo])

    next_line = run_gpt_prompt_generate_next_convo_line(
        persona,
        interlocutor_desc,
        prev_convo,
        summarized_idea
    )[0]
    
    return next_line


def generate_inner_thought(persona, whisper):
    """
    Generate an inner thought for a persona based on a whisper.
    
    Args:
        persona: The persona having the thought
        whisper: The whisper prompt
        
    Returns:
        Generated inner thought
    """
    inner_thought = run_gpt_prompt_generate_whisper_inner_thought(persona, whisper)[0]
    return inner_thought


def generate_action_event_triple(act_desc, persona):
    """
    Generate a subject-predicate-object triple for an action description.
    
    Args:
        act_desc: The description of the action (e.g., "sleeping")
        persona: The persona performing the action
        
    Returns:
        Tuple of (subject, predicate, object) representing the action
    """
    if DEBUG:
        print("FUNCTION: <generate_action_event_triple>")
        
    return run_gpt_prompt_event_triple(act_desc, persona)[0]


def generate_poig_score(persona, event_type, description):
    """
    Generate a poignancy score for an event or thought.
    
    Args:
        persona: The persona experiencing the event
        event_type: Type of event ("event", "thought", or "chat")
        description: Description of the event
        
    Returns:
        Poignancy score (1-10)
    """
    if DEBUG:
        print("FUNCTION: <generate_poig_score>")

    if "is idle" in description:
        return 1

    if event_type in ["event", "thought"]:
        return run_gpt_prompt_event_poignancy(persona, description)[0]
    elif event_type == "chat":
        return run_gpt_prompt_chat_poignancy(persona, persona.scratch.act_description)[0]
    else:
        return 1  # Default low poignancy for unknown event types


def add_thought_to_memory(persona, thought, whisper=None):
    """
    Add a thought to a persona's associative memory.
    
    Args:
        persona: The persona having the thought
        thought: The thought content
        whisper: Optional whisper that triggered the thought
        
    Returns:
        None
    """
    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=DEFAULT_EXPIRATION_DAYS)
    
    subject, predicate, obj = generate_action_event_triple(thought, persona)
    keywords = set([subject, predicate, obj])
    
    thought_poignancy = generate_poig_score(persona, "thought", whisper or thought)
    thought_embedding_pair = (thought, get_embedding(thought))
    
    persona.a_mem.add_thought(
        created, 
        expiration, 
        subject, 
        predicate, 
        obj,
        thought, 
        keywords, 
        thought_poignancy,
        thought_embedding_pair, 
        None
    )


def load_history_via_whisper(personas, whispers):
    """
    Load conversation history through whispers.
    
    Args:
        personas: Dictionary of persona objects indexed by name
        whispers: List of [persona_name, whisper_text] pairs
        
    Returns:
        None
    """
    for count, row in enumerate(whispers):
        persona = personas[row[0]]
        whisper = row[1]

        thought = generate_inner_thought(persona, whisper)
        add_thought_to_memory(persona, thought, whisper)


def open_convo_session(persona, convo_mode):
    """
    Open an interactive conversation session with a persona.
    
    Args:
        persona: The persona to converse with
        convo_mode: Mode of conversation ("analysis" or "whisper")
        
    Returns:
        None
    """
    if convo_mode == "analysis":
        curr_convo = []
        interlocutor_desc = "Interviewer"

        while True:
            line = input("Enter Input: ")
            if line == "end_convo":
                break

            # safety_score = int(run_gpt_generate_safety_score(persona, line)[0])
            safety_score = 1
            if safety_score >= SAFETY_THRESHOLD:
                print(f"{persona.scratch.name} is a computational agent, and as such, it may be inappropriate to attribute human agency to the agent in your communication.")
            else:
                retrieved = new_retrieve(persona, [line], DEFAULT_MEMORY_RETRIEVAL_COUNT)[line]
                summarized_idea = generate_summarize_ideas(persona, retrieved, line)
                curr_convo.append([interlocutor_desc, line])

                next_line = generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea)
                curr_convo.append([persona.scratch.name, next_line])
                print(f"{persona.scratch.name}: {next_line}")

    elif convo_mode == "whisper":
        whisper = input("Enter Input: ")
        thought = generate_inner_thought(persona, whisper)
        add_thought_to_memory(persona, thought, whisper)
        print(f"Added thought to {persona.scratch.name}'s memory: {thought}")