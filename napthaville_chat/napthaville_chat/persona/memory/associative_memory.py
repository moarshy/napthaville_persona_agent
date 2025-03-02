import json
import datetime
from napthaville_persona_agent.persona.prompts.gpt_structure import get_embedding


class ConceptNode:
    """
    Represents a single memory node in the associative memory.
    
    Each node can be an event, thought, or chat, and contains metadata
    about when it was created, its importance (poignancy), and semantic content.
    """
    def __init__(self,
                 node_id, node_count, type_count, node_type, depth,
                 created, expiration, 
                 s, p, o, 
                 description, embedding_key, poignancy, keywords, filling):
        """
        Initialize a new ConceptNode.
        
        Args:
            node_id: Unique identifier for this node
            node_count: Position in the overall sequence of nodes
            type_count: Position in type-specific sequence
            node_type: Type of node ("thought", "event", or "chat")
            depth: Recursion depth for thought nodes
            created: Datetime when the node was created
            expiration: Datetime when the node expires (can be None)
            s: Subject of the node (typically agent name)
            p: Predicate/relation
            o: Object
            description: Text description of the memory
            embedding_key: Key to retrieve the vector embedding
            poignancy: Importance score (1-10)
            keywords: Set of relevant keywords
            filling: Additional content (e.g., for chat nodes)
        """
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.type = node_type  # thought / event / chat
        self.depth = depth

        self.created = created
        self.expiration = expiration
        self.last_accessed = self.created

        self.subject = s
        self.predicate = p
        self.object = o

        self.description = description
        self.embedding_key = embedding_key
        self.poignancy = poignancy
        self.keywords = keywords
        self.filling = filling

    def spo_summary(self):
        """
        Get a tuple of (subject, predicate, object) for this node.
        
        Returns:
            Tuple of (subject, predicate, object)
        """
        return (self.subject, self.predicate, self.object)


class AssociativeMemory:
    """
    Long-term memory system for generative agents.
    
    This implements the Memory Stream module from the generative agents paper,
    storing and retrieving memories (events, thoughts, and conversations).
    """
    def __init__(self, associative_memory):
        """
        Initialize the associative memory.
        
        Args:
            f_saved: Path to saved memory data, or False if starting fresh
        """
        # Core data structures
        self.id_to_node = dict()  # Maps node IDs to ConceptNode objects

        # Chronological sequences for each node type
        self.seq_event = []       # Events in reverse chronological order
        self.seq_thought = []     # Thoughts in reverse chronological order
        self.seq_chat = []        # Chats in reverse chronological order

        # Keyword-to-node mappings for fast retrieval
        self.kw_to_event = dict()
        self.kw_to_thought = dict()
        self.kw_to_chat = dict()

        # Keyword strength tracking (frequency)
        self.kw_strength_event = dict()
        self.kw_strength_thought = dict()

        # Vector embeddings for semantic retrieval
        self.embeddings = {}
        
        # Load existing memory if provided
        if associative_memory:
            try:
                self.embeddings = associative_memory["embeddings"]
                
                # Load all nodes
                nodes_load = associative_memory["nodes"]
                for count in range(len(nodes_load.keys())):
                    node_id = f"node_{str(count+1)}"
                    node_details = nodes_load[node_id]

                    node_count = node_details["node_count"]
                    type_count = node_details["type_count"]
                    node_type = node_details["type"]
                    depth = node_details["depth"]

                    created = datetime.datetime.strptime(node_details["created"], 
                                                        '%Y-%m-%d %H:%M:%S')
                    expiration = None
                    if node_details["expiration"]:
                        expiration = datetime.datetime.strptime(node_details["expiration"],
                                                                '%Y-%m-%d %H:%M:%S')

                    s = node_details["subject"]
                    p = node_details["predicate"]
                    o = node_details["object"]

                    description = node_details["description"]
                    embedding_pair = (node_details["embedding_key"], 
                                    self.embeddings[node_details["embedding_key"]])
                    poignancy = node_details["poignancy"]
                    keywords = set(node_details["keywords"])
                    filling = node_details["filling"]
                    
                    # Add the node to the appropriate collection
                    if node_type == "event":
                        self.add_event(created, expiration, s, p, o, 
                                description, keywords, poignancy, embedding_pair, filling)
                    elif node_type == "chat":
                        self.add_chat(created, expiration, s, p, o, 
                                description, keywords, poignancy, embedding_pair, filling)
                    elif node_type == "thought":
                        self.add_thought(created, expiration, s, p, o, 
                                description, keywords, poignancy, embedding_pair, filling)

                # Load keyword strength data
                kw_strength_load = associative_memory["kw_strength"]
                if "kw_strength_event" in kw_strength_load and kw_strength_load["kw_strength_event"]:
                    self.kw_strength_event = kw_strength_load["kw_strength_event"]
                if "kw_strength_thought" in kw_strength_load and kw_strength_load["kw_strength_thought"]:
                    self.kw_strength_thought = kw_strength_load["kw_strength_thought"]
            except Exception as e:
                print(f"Error loading associative memory: {e}")
                # Initialize empty memory if loading fails
                pass

    def to_dict(self):
        """
        Save the associative memory to disk.
        
        Args:
            out_json: Directory path where memory files will be saved
        """
        # Save nodes
        r = dict()
        for count in range(len(self.id_to_node.keys()), 0, -1):
            node_id = f"node_{str(count)}"
            node = self.id_to_node[node_id]

            r[node_id] = dict()
            r[node_id]["node_count"] = node.node_count
            r[node_id]["type_count"] = node.type_count
            r[node_id]["type"] = node.type
            r[node_id]["depth"] = node.depth

            r[node_id]["created"] = node.created.strftime('%Y-%m-%d %H:%M:%S')
            r[node_id]["expiration"] = None
            if node.expiration:
                r[node_id]["expiration"] = (node.expiration
                                            .strftime('%Y-%m-%d %H:%M:%S'))

            r[node_id]["subject"] = node.subject
            r[node_id]["predicate"] = node.predicate
            r[node_id]["object"] = node.object

            r[node_id]["description"] = node.description
            r[node_id]["embedding_key"] = node.embedding_key
            r[node_id]["poignancy"] = node.poignancy
            r[node_id]["keywords"] = list(node.keywords)
            r[node_id]["filling"] = node.filling


        # Save keyword strength data
        kw_strength = dict()
        kw_strength["kw_strength_event"] = self.kw_strength_event
        kw_strength["kw_strength_thought"] = self.kw_strength_thought

        return {"nodes": r, "kw_strength": kw_strength, "embeddings": self.embeddings}

    def add_event(self, created, expiration, s, p, o, 
                  description, keywords, poignancy, 
                  embedding_pair, filling):
        """
        Add an event node to the associative memory.
        
        Args:
            created: Datetime when the event occurred
            expiration: Datetime when the event expires (can be None)
            s: Subject of the event (typically agent name)
            p: Predicate/relation
            o: Object
            description: Text description of the event
            keywords: Set of relevant keywords
            poignancy: Importance score (1-10)
            embedding_pair: (key, vector) pair for semantic embedding
            filling: Additional content (typically empty for events)
            
        Returns:
            The created ConceptNode
        """
        # Setting up the node ID and counts
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_event) + 1
        node_type = "event"
        node_id = f"node_{str(node_count)}"
        depth = 0

        # Node type specific clean up
        if "(" in description:
            description = (" ".join(description.split()[:3]) 
                          + " " 
                          + description.split("(")[-1][:-1])

        # Creating the ConceptNode object
        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                          created, expiration, 
                          s, p, o, 
                          description, embedding_pair[0], 
                          poignancy, keywords, filling)

        # Creating various dictionary cache for fast access
        self.seq_event.insert(0, node)  # Add to front of list
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_event:
                self.kw_to_event[kw].insert(0, node)  # Add to front of list
            else:
                self.kw_to_event[kw] = [node]
        self.id_to_node[node_id] = node

        # Adding in the keyword strength (skip for idle events)
        if f"{p} {o}" != "is idle":
            for kw in keywords:
                if kw in self.kw_strength_event:
                    self.kw_strength_event[kw] += 1
                else:
                    self.kw_strength_event[kw] = 1

        # Store the embedding
        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node

    def add_thought(self, created, expiration, s, p, o, 
                   description, keywords, poignancy, 
                   embedding_pair, filling):
        """
        Add a thought node to the associative memory.
        
        Args:
            created: Datetime when the thought occurred
            expiration: Datetime when the thought expires (can be None)
            s: Subject of the thought (typically agent name)
            p: Predicate/relation
            o: Object
            description: Text description of the thought
            keywords: Set of relevant keywords
            poignancy: Importance score (1-10)
            embedding_pair: (key, vector) pair for semantic embedding
            filling: Related node IDs this thought is based on
            
        Returns:
            The created ConceptNode
        """
        # Setting up the node ID and counts
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_thought) + 1
        node_type = "thought"
        node_id = f"node_{str(node_count)}"
        
        # Calculate depth for thought nodes (1 + max depth of referenced nodes)
        depth = 1
        try:
            if filling:
                depth += max([self.id_to_node[i].depth for i in filling])
        except:
            pass

        # Creating the ConceptNode object
        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                          created, expiration, 
                          s, p, o, 
                          description, embedding_pair[0], poignancy, keywords, filling)

        # Creating various dictionary cache for fast access
        self.seq_thought.insert(0, node)  # Add to front of list
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_thought:
                self.kw_to_thought[kw].insert(0, node)  # Add to front of list
            else:
                self.kw_to_thought[kw] = [node]
        self.id_to_node[node_id] = node

        # Adding in the keyword strength (skip for idle thoughts)
        if f"{p} {o}" != "is idle":
            for kw in keywords:
                if kw in self.kw_strength_thought:
                    self.kw_strength_thought[kw] += 1
                else:
                    self.kw_strength_thought[kw] = 1

        # Store the embedding
        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        return node

    def add_chat(self, created, expiration, s, p, o, 
                description, keywords, poignancy, 
                embedding_pair, filling):
        """
        Add a chat node to the associative memory.
        
        Args:
            created: Datetime when the chat occurred
            expiration: Datetime when the chat expires (can be None)
            s: Subject of the chat (typically agent name)
            p: Predicate (usually "chat with")
            o: Object (other agent name)
            description: Text description/summary of the chat
            keywords: Set of relevant keywords
            poignancy: Importance score (1-10)
            embedding_pair: (key, vector) pair for semantic embedding
            filling: List of [speaker, text] pairs containing the conversation
            
        Returns:
            The created ConceptNode
        """
        # Setting up the node ID and counts
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_chat) + 1
        node_type = "chat"
        node_id = f"node_{str(node_count)}"
        depth = 0

        # Creating the ConceptNode object
        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                          created, expiration, 
                          s, p, o, 
                          description, embedding_pair[0], poignancy, keywords, filling)

        # Creating various dictionary cache for fast access
        self.seq_chat.insert(0, node)  # Add to front of list
        keywords = [i.lower() for i in keywords]
        for kw in keywords:
            if kw in self.kw_to_chat:
                self.kw_to_chat[kw].insert(0, node)  # Add to front of list
            else:
                self.kw_to_chat[kw] = [node]
        self.id_to_node[node_id] = node

        # Store the embedding
        self.embeddings[embedding_pair[0]] = embedding_pair[1]
            
        return node

    def get_summarized_latest_events(self, retention):
        """
        Get a set of (subject, predicate, object) tuples from recent events.
        
        Args:
            retention: Number of most recent events to include
            
        Returns:
            Set of (subject, predicate, object) tuples
        """
        ret_set = set()
        for e_node in self.seq_event[:retention]:
            ret_set.add(e_node.spo_summary())
        return ret_set

    def get_str_seq_events(self):
        """
        Get a string representation of all events in the memory.
        
        Returns:
            String containing all events
        """
        ret_str = ""
        for count, event in enumerate(self.seq_event):
            ret_str += f'{"Event", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n'
        return ret_str

    def get_str_seq_thoughts(self):
        """
        Get a string representation of all thoughts in the memory.
        
        Returns:
            String containing all thoughts
        """
        ret_str = ""
        for count, event in enumerate(self.seq_thought):
            ret_str += f'{"Thought", len(self.seq_thought) - count, ": ", event.spo_summary(), " -- ", event.description}'
        return ret_str

    def get_str_seq_chats(self):
        """
        Get a string representation of all chats in the memory.
        
        Returns:
            String containing all chats
        """
        ret_str = ""
        for count, event in enumerate(self.seq_chat):
            # Fixed: Use event.object directly instead of event.object.content
            ret_str += f"with {event.object} ({event.description})\n"
            ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
            for row in event.filling:
                ret_str += f"{row[0]}: {row[1]}\n"
        return ret_str

    def retrieve_relevant_thoughts(self, s_content, p_content, o_content):
        """
        Retrieve thoughts relevant to the given subject, predicate, and object.
        
        Args:
            s_content: Subject content to match
            p_content: Predicate content to match
            o_content: Object content to match
            
        Returns:
            Set of relevant ConceptNode objects
        """
        contents = [s_content, p_content, o_content]

        ret = []
        for i in contents:
            if isinstance(i, str) and i.lower() in self.kw_to_thought:
                ret += self.kw_to_thought[i.lower()]

        return set(ret)

    def retrieve_relevant_events(self, s_content, p_content, o_content):
        """
        Retrieve events relevant to the given subject, predicate, and object.
        
        Args:
            s_content: Subject content to match
            p_content: Predicate content to match
            o_content: Object content to match
            
        Returns:
            Set of relevant ConceptNode objects
        """
        contents = [s_content, p_content, o_content]

        ret = []
        for i in contents:
            if isinstance(i, str) and i in self.kw_to_event:
                ret += self.kw_to_event[i]

        return set(ret)

    def get_last_chat(self, target_persona_name):
        """
        Get the most recent chat with a specific persona.
        
        Args:
            target_persona_name: Name of the other persona
            
        Returns:
            ConceptNode of the most recent chat, or False if none found
        """
        if target_persona_name.lower() in self.kw_to_chat:
            return self.kw_to_chat[target_persona_name.lower()][0]
        else:
            return False
        
if __name__ == "__main__":
    from pathlib import Path
    import os
    
    # Get the absolute path of the repository
    repo_path = Path(__file__).parent.parent.parent.parent
    
    # Construct the path to the test data
    test_data_path = repo_path / "test_data" / "July1_the_ville_isabella_maria_klaus-step-3-19" / "personas" / "Isabella Rodriguez" / "bootstrap_memory" / "associative_memory"
    
    print(f"Looking for associative memory at: {test_data_path}")
    
    associative_memory = AssociativeMemory(test_data_path)
    print("get_str_seq_chats")
    print(associative_memory.get_str_seq_chats())

    # get_summarized_latest_events
    print("get_summarized_latest_events")
    print(associative_memory.get_summarized_latest_events(10))

    # get_str_seq_events
    print("get_str_seq_events")
    print(associative_memory.get_str_seq_events())

    # get_str_seq_thoughts
    print("get_str_seq_thoughts")
    print(associative_memory.get_str_seq_thoughts())

    # get_str_seq_chats
    print("get_str_seq_chats")
    print(associative_memory.get_str_seq_chats())

    # retrieve_relevant_thoughts
    print("retrieve_relevant_thoughts")
    print(associative_memory.retrieve_relevant_thoughts("Isabella Rodriguez", "chat with", "Maria Lopez"))

    # retrieve_relevant_events
    print("retrieve_relevant_events")
    print(associative_memory.retrieve_relevant_events("Isabella Rodriguez", "chat with", "Maria Lopez"))

    # get_last_chat
    print("get_last_chat")
    print(associative_memory.get_last_chat("Maria Lopez"))

    def run_add_function_tests():
        """
        Test the add_event, add_thought, and add_chat functions of AssociativeMemory.
        """
        print("\n=== RUNNING ADD FUNCTION TESTS ===")
        
        # Create a new empty memory
        memory = AssociativeMemory(False)
        
        # Current time for all test entries
        test_time = datetime.datetime.now()
        
        # Test 1: Adding an event
        print("\nTest 1: Adding an event")
        test_event = memory.add_event(
            created=test_time,
            expiration=None,
            s="Isabella Rodriguez",
            p="is",
            o="painting",
            description="painting a landscape on canvas",
            keywords={"painting", "art", "canvas"},
            poignancy=6,
            embedding_pair=("test_event", [0.1, 0.2, 0.3]),
            filling=[]
        )
        
        # Assert event was added correctly
        assert len(memory.seq_event) == 1, f"Expected 1 event, got {len(memory.seq_event)}"
        assert memory.seq_event[0].subject == "Isabella Rodriguez", f"Expected subject 'Isabella Rodriguez', got '{memory.seq_event[0].subject}'"
        assert memory.seq_event[0].predicate == "is", f"Expected predicate 'is', got '{memory.seq_event[0].predicate}'"
        assert memory.seq_event[0].object == "painting", f"Expected object 'painting', got '{memory.seq_event[0].object}'"
        assert memory.seq_event[0].description == "painting a landscape on canvas", f"Description mismatch"
        assert memory.seq_event[0].poignancy == 6, f"Expected poignancy 6, got {memory.seq_event[0].poignancy}"
        
        print("✓ Event was added successfully")
        print(f"  Event ID: {test_event.node_id}")
        print(f"  Event summary: {test_event.subject} {test_event.predicate} {test_event.object}")
        print(f"  Description: {test_event.description}")
        
        # Test 2: Adding a thought
        print("\nTest 2: Adding a thought")
        test_thought = memory.add_thought(
            created=test_time,
            expiration=None,
            s="Isabella Rodriguez",
            p="thinks about",
            o="her art exhibition",
            description="wondering if her artwork will be well-received at the upcoming exhibition",
            keywords={"exhibition", "art", "anxiety"},
            poignancy=8,
            embedding_pair=("test_thought", [0.3, 0.4, 0.5]),
            filling=[]
        )
        
        # Assert thought was added correctly
        assert len(memory.seq_thought) == 1, f"Expected 1 thought, got {len(memory.seq_thought)}"
        assert memory.seq_thought[0].subject == "Isabella Rodriguez", f"Expected subject 'Isabella Rodriguez', got '{memory.seq_thought[0].subject}'"
        assert memory.seq_thought[0].predicate == "thinks about", f"Expected predicate 'thinks about', got '{memory.seq_thought[0].predicate}'"
        assert memory.seq_thought[0].object == "her art exhibition", f"Expected object 'her art exhibition', got '{memory.seq_thought[0].object}'"
        assert "exhibition" in memory.seq_thought[0].description.lower(), f"Description should mention exhibition"
        assert memory.seq_thought[0].poignancy == 8, f"Expected poignancy 8, got {memory.seq_thought[0].poignancy}"
        
        print("✓ Thought was added successfully")
        print(f"  Thought ID: {test_thought.node_id}")
        print(f"  Thought summary: {test_thought.subject} {test_thought.predicate} {test_thought.object}")
        print(f"  Description: {test_thought.description}")
        
        # Test 3: Adding a chat
        print("\nTest 3: Adding a chat")
        test_chat_content = [
            ["Isabella Rodriguez", "Hi Maria, how are you today?"],
            ["Maria Lopez", "I'm good, Isabella! How's your painting coming along?"],
            ["Isabella Rodriguez", "It's going well, I'm making progress on the landscape piece."]
        ]
        
        test_chat = memory.add_chat(
            created=test_time,
            expiration=None,
            s="Isabella Rodriguez",
            p="chat with",
            o="Maria Lopez",
            description="discussing Isabella's painting progress",
            keywords={"Maria", "painting", "landscape", "progress"},
            poignancy=5,
            embedding_pair=("test_chat", [0.5, 0.6, 0.7]),
            filling=test_chat_content
        )
        
        # Assert chat was added correctly
        assert len(memory.seq_chat) == 1, f"Expected 1 chat, got {len(memory.seq_chat)}"
        assert memory.seq_chat[0].subject == "Isabella Rodriguez", f"Expected subject 'Isabella Rodriguez', got '{memory.seq_chat[0].subject}'"
        assert memory.seq_chat[0].predicate == "chat with", f"Expected predicate 'chat with', got '{memory.seq_chat[0].predicate}'"
        assert memory.seq_chat[0].object == "Maria Lopez", f"Expected object 'Maria Lopez', got '{memory.seq_chat[0].object}'"
        assert "painting" in memory.seq_chat[0].description.lower(), f"Description should mention painting"
        assert memory.seq_chat[0].poignancy == 5, f"Expected poignancy 5, got {memory.seq_chat[0].poignancy}"
        assert len(memory.seq_chat[0].filling) == 3, f"Expected 3 chat turns, got {len(memory.seq_chat[0].filling)}"
        
        print("✓ Chat was added successfully")
        print(f"  Chat ID: {test_chat.node_id}")
        print(f"  Chat summary: {test_chat.subject} {test_chat.predicate} {test_chat.object}")
        print(f"  Description: {test_chat.description}")
        print(f"  Chat turns: {len(test_chat.filling)}")
        
        # Test 4: Keyword indexing
        print("\nTest 4: Testing keyword indexing")
        # Check if keywords are properly indexed
        assert "painting" in memory.kw_to_event, "Keyword 'painting' should be in event keywords"
        assert "exhibition" in memory.kw_to_thought, "Keyword 'exhibition' should be in thought keywords"
        assert "maria" in memory.kw_to_chat, "Keyword 'maria' should be in chat keywords"
        
        # Check that memory retrieval works correctly
        painting_events = memory.retrieve_relevant_events("Isabella Rodriguez", "is", "painting")
        assert len(painting_events) > 0, "Should retrieve events related to painting"
        
        exhibition_thoughts = memory.retrieve_relevant_thoughts("Isabella Rodriguez", "thinks about", "her art exhibition")
        assert len(exhibition_thoughts) > 0, "Should retrieve thoughts related to exhibition"
        
        maria_chat = memory.get_last_chat("Maria Lopez")
        assert maria_chat is not False, "Should retrieve chat with Maria"
        
        print("✓ Keyword indexing works correctly")
        print(f"  Retrieved {len(painting_events)} events for painting")
        print(f"  Retrieved {len(exhibition_thoughts)} thoughts for exhibition")
        print(f"  Last chat with Maria found: {maria_chat is not False}")
        
        print("\n=== ADD FUNCTION TESTS COMPLETED SUCCESSFULLY ===")
        
        return memory
    
    run_add_function_tests()