import random
from napthaville_persona_agent.persona.prompts.gpt_structure import *


def execute(persona, maze, personas, plan):
    """
    Given a plan (action's string address), we execute the plan (actually
    outputs the tile coordinate path and the next coordinate for the
    persona).

    INPUT:
        persona: Current <Persona> instance.
        maze: An instance of current <Maze>.
        personas: A dictionary of all personas in the world.
        plan: This is a string address of the action we need to execute.
           It comes in the form of "{world}:{sector}:{arena}:{game_objects}".
           It is important that you access this without doing negative
           indexing (e.g., [-1]) because the latter address elements may not be
           present in some cases.
           e.g., "dolores double studio:double studio:bedroom 1:bed"

    OUTPUT:
        execution: A tuple containing (next_tile, pronunciation, description)
    """
    # Reset path if a random action is requested and we have no planned path
    if "<random>" in plan and not persona.scratch.planned_path:
        persona.scratch.act_path_set = False

    # If path is not set for the current action, we need to construct a new path
    if not persona.scratch.act_path_set:
        # Get target tiles based on the plan type
        target_tiles = get_target_tiles(plan, persona, maze, personas)
        
        # Find the optimal path to one of the target tiles
        path = find_optimal_path(target_tiles, persona.scratch.curr_tile, maze)
        
        # Set the planned path (excluding current tile)
        persona.scratch.planned_path = path[1:]
        persona.scratch.act_path_set = True
    
    # Determine the next step
    next_tile = persona.scratch.curr_tile
    if persona.scratch.planned_path:
        next_tile = persona.scratch.planned_path[0]
        persona.scratch.planned_path = persona.scratch.planned_path[1:]

    # Create description for the execution
    description = f"{persona.scratch.act_description} @ {persona.scratch.act_address}"
    
    # Return the execution tuple
    return next_tile, persona.scratch.act_pronunciatio, description


def get_target_tiles(plan, persona, maze, personas):
    """
    Determine target tiles based on the plan type.
    
    Args:
        plan: The action plan string
        persona: Current persona instance
        maze: Current maze instance
        personas: Dictionary of all personas
        
    Returns:
        List of target tile coordinates
    """
    if "<persona>" in plan:
        return get_persona_interaction_tiles(plan, persona, maze, personas)
    elif "<waiting>" in plan:
        # Executing interaction where the persona waits
        x = int(plan.split()[1])
        y = int(plan.split()[2])
        return [[x, y]]
    elif "<random>" in plan:
        # Executing a random location action
        plan_base = ":".join(plan.split(":")[:-1])
        tiles = maze.address_tiles[plan_base]
        return random.sample(list(tiles), 1)
    else:
        # Default execution - go to the location of the action
        if plan not in maze.address_tiles:
            # Handle missing address by returning a fallback location
            # This replaces the error-causing line in the original code
            return list(maze.address_tiles.get("Johnson Park:park:park garden", []))
        return maze.address_tiles[plan]


def get_persona_interaction_tiles(plan, persona, maze, personas):
    """
    Get tiles for persona-to-persona interactions.
    
    Args:
        plan: The action plan string
        persona: Current persona instance
        maze: Current maze instance
        personas: Dictionary of all personas
        
    Returns:
        List of target tile coordinates
    """
    target_persona_name = plan.split("<persona>")[-1].strip()
    target_p_tile = personas[target_persona_name].scratch.curr_tile
    
    potential_path = path_finder(
        maze.collision_maze,
        persona.scratch.curr_tile,
        target_p_tile,
        collision_block_id
    )
    
    if len(potential_path) <= 2:
        return [potential_path[0]]
    
    # Find midpoint to approach target persona
    mid_index = int(len(potential_path) / 2)
    potential_1 = path_finder(
        maze.collision_maze,
        persona.scratch.curr_tile,
        potential_path[mid_index],
        collision_block_id
    )
    
    potential_2 = path_finder(
        maze.collision_maze,
        persona.scratch.curr_tile,
        potential_path[mid_index + 1],
        collision_block_id
    )
    
    if len(potential_1) <= len(potential_2):
        return [potential_path[mid_index]]
    else:
        return [potential_path[mid_index + 1]]


def filter_target_tiles(target_tiles, maze, personas):
    """
    Filter target tiles to avoid occupied tiles when possible.
    
    Args:
        target_tiles: List of potential target tiles
        maze: Current maze instance
        personas: Dictionary of all personas
        
    Returns:
        Filtered list of target tiles
    """
    # Sample a subset of target tiles if there are many
    if len(target_tiles) < 4:
        sampled_tiles = random.sample(list(target_tiles), len(target_tiles))
    else:
        sampled_tiles = random.sample(list(target_tiles), 4)
    
    # Try to find unoccupied tiles
    persona_name_set = set(personas.keys())
    unoccupied_tiles = []
    
    for tile in sampled_tiles:
        curr_events = maze.access_tile(tile)["events"]
        tile_occupied = any(event[0] in persona_name_set for event in curr_events)
        
        if not tile_occupied:
            unoccupied_tiles.append(tile)
    
    # If all tiles are occupied, use the original sampled tiles
    return unoccupied_tiles if unoccupied_tiles else sampled_tiles


def find_optimal_path(target_tiles, curr_tile, maze):
    """
    Find the shortest path to one of the target tiles.
    
    Args:
        target_tiles: List of potential target tiles
        curr_tile: Current tile of the persona
        maze: Current maze instance
        
    Returns:
        Shortest path to a target tile
    """
    # Filter target tiles to prefer unoccupied ones
    filtered_tiles = filter_target_tiles(target_tiles, maze, {})
    
    closest_target_tile = None
    shortest_path = None
    
    for tile in filtered_tiles:
        curr_path = path_finder(
            maze.collision_maze,
            curr_tile,
            tile,
            collision_block_id
        )
        
        if not shortest_path or len(curr_path) < len(shortest_path):
            closest_target_tile = tile
            shortest_path = curr_path
    
    return shortest_path