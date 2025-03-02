from typing import Dict, List, Any
import numpy as np
from napthaville_persona_agent.persona.memory.associative_memory import ConceptNode
from napthaville_persona_agent.persona.prompts.gpt_structure import get_embedding

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two non-zero vectors
    of an inner product space, representing similarity between them.
    
    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)
        
    Returns:
        float: Cosine similarity value between -1 and 1
        
    Example:
        >>> cos_sim(np.array([0.3, 0.2, 0.5]), np.array([0.2, 0.2, 0.5]))
        0.9797958971132712
    """
    # Avoid division by zero
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize_dict_floats(d: Dict[Any, float], target_min: float, target_max: float) -> Dict[Any, float]:
    """
    Normalize the float values in a dictionary to a target range.
    
    Args:
        d: Dictionary with float values to normalize
        target_min: Minimum value in the normalized range
        target_max: Maximum value in the normalized range
        
    Returns:
        Dict: New dictionary with normalized values
        
    Example:
        >>> normalize_dict_floats({'a': 1.2, 'b': 3.4, 'c': 5.6, 'd': 7.8}, -5, 5)
        {'a': -5.0, 'b': -1.666..., 'c': 1.666..., 'd': 5.0}
    """
    if not d:
        return {}
        
    min_val = min(d.values())
    max_val = max(d.values())
    range_val = max_val - min_val
    
    normalized_dict = {}
    
    if range_val == 0:
        # All values are the same, set to middle of target range
        mid_value = (target_max + target_min) / 2
        for key in d:
            normalized_dict[key] = mid_value
    else:
        # Apply min-max normalization
        for key, val in d.items():
            normalized_dict[key] = (
                (val - min_val) * (target_max - target_min) / range_val + target_min
            )
            
    return normalized_dict


def top_highest_x_values(d: Dict[Any, float], x: int) -> Dict[Any, float]:
    """
    Extract the top x key-value pairs with the highest values from a dictionary.
    
    Args:
        d: Input dictionary
        x: Number of top values to extract
        
    Returns:
        Dict: Dictionary containing the top x key-value pairs
        
    Example:
        >>> top_highest_x_values({'a': 1.2, 'b': 3.4, 'c': 5.6, 'd': 7.8}, 3)
        {'d': 7.8, 'c': 5.6, 'b': 3.4}
    """
    # Limit x to the dictionary size
    x = min(x, len(d))
    
    # Sort and get top x items
    top_items = sorted(d.items(), key=lambda item: item[1], reverse=True)[:x]
    
    # Convert back to dictionary
    return dict(top_items)


def extract_recency(persona, nodes: List) -> Dict[str, float]:
    """
    Calculate recency scores for a list of memory nodes.
    
    Recency score decays exponentially based on the position in the list,
    with more recent nodes (later in the list) having higher scores.
    
    Args:
        persona: Persona object containing recency parameters
        nodes: List of memory nodes in chronological order
        
    Returns:
        Dict: Dictionary mapping node IDs to recency scores
    """
    if not nodes:
        return {}
        
    # Calculate recency values with exponential decay
    recency_vals = [
        persona.scratch.recency_decay ** i for i in range(1, len(nodes) + 1)
    ]
    
    # Map node IDs to their recency values
    recency_out = {node.node_id: recency_vals[count] for count, node in enumerate(nodes)}
    
    return recency_out


def extract_importance(persona, nodes: List) -> Dict[str, float]:
    """
    Extract importance scores for a list of memory nodes.
    
    The importance score is based on each node's poignancy value.
    
    Args:
        persona: Persona object
        nodes: List of memory nodes
        
    Returns:
        Dict: Dictionary mapping node IDs to importance scores
    """
    if not nodes:
        return {}
        
    # Map node IDs to their poignancy values
    return {node.node_id: node.poignancy for node in nodes}


def extract_relevance(persona, nodes: List, focal_pt: str) -> Dict[str, float]:
    """
    Calculate relevance scores between memory nodes and a focal point.
    
    Relevance is measured using cosine similarity between embeddings.
    
    Args:
        persona: Persona object containing embeddings
        nodes: List of memory nodes
        focal_pt: String describing the current focus point
        
    Returns:
        Dict: Dictionary mapping node IDs to relevance scores
    """
    if not nodes:
        return {}
        
    # Get embedding for the focal point
    focal_embedding = get_embedding(focal_pt)
    
    # Calculate cosine similarity for each node
    relevance_out = {}
    for node in nodes:
        node_embedding = persona.a_mem.embeddings[node.embedding_key]
        relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)
    
    return relevance_out


def retrieve(persona, perceived: List[ConceptNode]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve relevant memories based on perceived events.
    
    For each perceived event, retrieve relevant events and thoughts from memory.
    
    Args:
        persona: Persona object containing associative memory
        perceived: List of ConceptNode events perceived by the persona
        
    Returns:
        Dict: Dictionary mapping event descriptions to relevant memories
    """
    retrieved = {}
    
    for event in perceived:
        # For each perceived event, create a container for relevant memories
        retrieved[event.description] = {
            "curr_event": event,
            "events": list(persona.a_mem.retrieve_relevant_events(
                event.subject, event.predicate, event.object
            )),
            "thoughts": list(persona.a_mem.retrieve_relevant_thoughts(
                event.subject, event.predicate, event.object
            ))
        }
    
    return retrieved


def new_retrieve(persona, focal_points: List[str], n_count: int = 30) -> Dict[str, List]:
    """
    Enhanced memory retrieval based on multiple factors.
    
    Retrieves memories related to focal points using a weighted combination of
    recency, relevance, and importance scores.
    
    Args:
        persona: Persona object containing memory
        focal_points: List of focal points (strings) to retrieve memories for
        n_count: Maximum number of memories to retrieve per focal point
        
    Returns:
        Dict: Dictionary mapping focal points to lists of relevant memory nodes
    """
    # Initialize output dictionary
    retrieved = {}
    
    # Global weights for combining scores
    global_weights = [0.5, 3, 2]  # For recency, relevance, importance
    
    for focal_pt in focal_points:
        # Get all memory nodes (excluding idle nodes)
        nodes = [
            [i.last_accessed, i]
            for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
            if "idle" not in i.embedding_key
        ]
        
        # Sort nodes chronologically
        nodes = sorted(nodes, key=lambda x: x[0])
        nodes = [i for _, i in nodes]
        
        if not nodes:
            retrieved[focal_pt] = []
            continue
            
        # Calculate component scores
        recency_scores = extract_recency(persona, nodes)
        recency_scores = normalize_dict_floats(recency_scores, 0, 1)
        
        importance_scores = extract_importance(persona, nodes)
        importance_scores = normalize_dict_floats(importance_scores, 0, 1)
        
        relevance_scores = extract_relevance(persona, nodes, focal_pt)
        relevance_scores = normalize_dict_floats(relevance_scores, 0, 1)
        
        # Combine scores with weights
        master_scores = {}
        for node_id in recency_scores:
            master_scores[node_id] = (
                persona.scratch.recency_w * recency_scores[node_id] * global_weights[0] +
                persona.scratch.relevance_w * relevance_scores[node_id] * global_weights[1] +
                persona.scratch.importance_w * importance_scores[node_id] * global_weights[2]
            )
        
        # Debug output
        if hasattr(persona, 'debug') and persona.debug:
            for node_id, score in master_scores.items():
                node = persona.a_mem.id_to_node[node_id]
                print(f"Node: {node.embedding_key}, Total score: {score}")
                print(f"  Recency: {persona.scratch.recency_w * recency_scores[node_id] * global_weights[0]}")
                print(f"  Relevance: {persona.scratch.relevance_w * relevance_scores[node_id] * global_weights[1]}")
                print(f"  Importance: {persona.scratch.importance_w * importance_scores[node_id] * global_weights[2]}")
        
        # Get top n_count nodes
        top_scores = top_highest_x_values(master_scores, n_count)
        top_nodes = [persona.a_mem.id_to_node[node_id] for node_id in top_scores]
        
        # Update last_accessed time for retrieved nodes
        for node in top_nodes:
            node.last_accessed = persona.scratch.curr_time
        
        # Store results
        retrieved[focal_pt] = top_nodes
    
    return retrieved
