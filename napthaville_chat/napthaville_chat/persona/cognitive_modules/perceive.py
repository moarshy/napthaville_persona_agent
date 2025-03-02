import math
from operator import itemgetter
from napthaville_persona_agent.persona.prompts.gpt_structure import get_embedding
from napthaville_persona_agent.persona.prompts.run_gpt_prompt import (
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_chat_poignancy
)


def generate_poig_score(persona, event_type, description):
    """
    Generate a poignancy score for an event or chat.
    
    Args:
        persona: The Persona instance
        event_type: Type of the event ("event" or "chat")
        description: Description of the event/chat
        
    Returns:
        Poignancy score (1-10)
    """
    if "is idle" in description:
        return 1

    if event_type == "event":
        return run_gpt_prompt_event_poignancy(persona, description)[0]
    elif event_type == "chat":
        return run_gpt_prompt_chat_poignancy(persona, persona.scratch.act_description)[0]

# from pydantic import BaseModel
# from typing import Tuple, List, Dict, Any

# class PerceiveMaze(BaseModel):
#     curr_tile: Tuple[int, int]
#     vision_r: int
#     nearby_tiles: List[Tuple[int, int]]
#     access_tiles: Dict[Tuple[int, int], Dict[str, Any]]
#     curr_arena_path: List[str]

def perceive(persona, maze_data):
    nearby_tiles = maze_data["nearby_tiles"]
    curr_arena_path = maze_data["curr_arena_path"]['path']
    nearby_tiles_arena_path = [i['path'] for i in maze_data["nearby_tiles_arena_path"]]
    nearby_tiles_data = maze_data["nearby_tiles_data"]

    for j, tile in enumerate(nearby_tiles):
        i = nearby_tiles_data[j]
        if i["world"]:
            if i["world"] not in persona.s_mem.tree:
                persona.s_mem.tree[i["world"]] = {}
        if i["sector"]:
            if i["sector"] not in persona.s_mem.tree[i["world"]]:
                persona.s_mem.tree[i["world"]][i["sector"]] = {}
        if i["arena"]:
            if i["arena"] not in persona.s_mem.tree[i["world"]][i["sector"]]:
                persona.s_mem.tree[i["world"]][i["sector"]][i["arena"]] = []
        if i["game_object"]:
            if i["game_object"] not in persona.s_mem.tree[i["world"]][i["sector"]][i["arena"]]:
                persona.s_mem.tree[i["world"]][i["sector"]][i["arena"]].append(i["game_object"])

    curr_arena_path = maze_data["curr_arena_path"]['path']
    
    percept_events_set = set()
    
    # We will order our percept based on the distance, with the closest ones
    # getting priorities.
    percept_events_list = []
    
    # First, put all events occurring in nearby tiles into percept_events_list
    for j, tile in enumerate(nearby_tiles):
        tile_details = nearby_tiles_data[j]
        if tile_details["events"] and nearby_tiles_arena_path[j] == curr_arena_path:
            # Calculate the distance between the persona's current tile and the target tile
            dist = math.dist(
                [tile[0], tile[1]],
                [persona.scratch.curr_tile[0], persona.scratch.curr_tile[1]]
            )
            # Add relevant events to our temp set/list with distance info
            for event in tile_details["events"]:
                if event not in percept_events_set:
                    percept_events_list.append([dist, event])
                    percept_events_set.add(event)

    percept_events_list = sorted(percept_events_list, key=itemgetter(0))
    perceived_events = [event for _, event in percept_events_list[:persona.scratch.att_bandwidth]]
    ret_events = []
    for p_event in perceived_events:
        s, p, o, desc = p_event
        if not p:
            # If the predicate is not present, default the event to "idle"
            p = "is"
            o = "idle"
            desc = "idle"
        
        desc = f"{s.split(':')[-1]} is {desc}"
        p_event = (s, p, o)

        latest_events = persona.a_mem.get_summarized_latest_events(persona.scratch.retention)
        if p_event not in latest_events:
            # Manage keywords
            keywords = set()
            sub = p_event[0]
            obj = p_event[2]
            
            if ":" in p_event[0]:
                sub = p_event[0].split(":")[-1]
            if ":" in p_event[2]:
                obj = p_event[2].split(":")[-1]
                
            keywords.update([sub, obj])

            # Get event embedding
            desc_embedding_in = desc
            if "(" in desc:
                desc_embedding_in = desc_embedding_in.split("(")[1].split(")")[0].strip()
                
            if desc_embedding_in in persona.a_mem.embeddings:
                event_embedding = persona.a_mem.embeddings[desc_embedding_in]
            else:
                event_embedding = get_embedding(desc_embedding_in)
                
            event_embedding_pair = (desc_embedding_in, event_embedding)
            
            # Get event poignancy
            event_poignancy = generate_poig_score(persona, "event", desc_embedding_in)

            # If we observe the persona's self chat, include that in memory
            chat_node_ids = []
            if p_event[0] == f"{persona.name}" and p_event[1] == "chat with":
                curr_event = persona.scratch.act_event
                
                if persona.scratch.act_description in persona.a_mem.embeddings:
                    chat_embedding = persona.a_mem.embeddings[persona.scratch.act_description]
                else:
                    chat_embedding = get_embedding(persona.scratch.act_description)
                    
                chat_embedding_pair = (persona.scratch.act_description, chat_embedding)
                chat_poignancy = generate_poig_score(persona, "chat", persona.scratch.act_description)
                
                chat_node = persona.a_mem.add_chat(
                    persona.scratch.curr_time,
                    None,
                    curr_event[0],
                    curr_event[1],
                    curr_event[2],
                    persona.scratch.act_description,
                    keywords,
                    chat_poignancy,
                    chat_embedding_pair,
                    persona.scratch.chat
                )
                chat_node_ids = [chat_node.node_id]

            # Add the current event to the agent's memory
            new_event_node = persona.a_mem.add_event(
                persona.scratch.curr_time,
                None,
                s, p, o,
                desc,
                keywords,
                event_poignancy,
                event_embedding_pair,
                chat_node_ids
            )
            
            ret_events.append(new_event_node)
            persona.scratch.importance_trigger_curr -= event_poignancy
            persona.scratch.importance_ele_n += 1

    return ret_events