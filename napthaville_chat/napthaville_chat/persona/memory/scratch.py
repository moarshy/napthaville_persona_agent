import datetime
import json
import os

class Scratch:
    """
    Short-term memory module for generative agents.
    
    Scratch memory stores the agent's current state, including:
    - Perception parameters
    - Identity information
    - Current activity and planning
    - Reflection mechanisms
    
    This serves as the working memory that coordinates between long-term
    memory and immediate actions.
    """
    
    def __init__(self, scratch_memory):
        """
        Initialize the scratch memory.
        
        Args:
            f_saved: Path to saved scratch memory file, or False if creating new
        """
        # PERSONA HYPERPARAMETERS
        # The number of tiles that the persona can see around them
        self.vision_r = 4
        # Attention bandwidth - number of events agent can perceive at once
        self.att_bandwidth = 3
        # Retention - number of recent events to remember without storing in memory
        self.retention = 5

        # WORLD INFORMATION
        # Perceived world time
        self.curr_time = None
        # Current x,y tile coordinate of the persona
        self.curr_tile = None
        # Perceived world daily requirement
        self.daily_plan_req = None
        
        # THE CORE IDENTITY OF THE PERSONA 
        # Base information about the persona
        self.name = None
        self.first_name = None
        self.last_name = None
        self.age = None
        # L0 permanent core traits
        self.innate = None
        # L1 stable traits
        self.learned = None
        # L2 external implementation
        self.currently = None
        self.lifestyle = None
        self.living_area = None

        # REFLECTION VARIABLES
        # Threshold for forgetting concepts
        self.concept_forget = 100
        # How much time to spend on daily reflection (minutes)
        self.daily_reflection_time = 60 * 3
        # Number of reflections to generate per day
        self.daily_reflection_size = 5
        # Threshold for reflecting on overlapping concepts
        self.overlap_reflect_th = 2
        # Thresholds for keyword strength to trigger reflection
        self.kw_strg_event_reflect_th = 4
        self.kw_strg_thought_reflect_th = 4

        # New reflection variables
        # Weights for memory retrieval scoring
        self.recency_w = 1
        self.relevance_w = 1
        self.importance_w = 1
        # Decay factor for recency
        self.recency_decay = 0.99
        # Maximum importance trigger threshold
        self.importance_trigger_max = 150
        # Current importance trigger (decreases with important events)
        self.importance_trigger_curr = self.importance_trigger_max
        # Count of important elements encountered
        self.importance_ele_n = 0 
        # Number of thoughts to generate during reflection
        self.thought_count = 5

        # PERSONA PLANNING 
        # Daily requirements - goals the persona aims to achieve today
        # e.g., ['Work on her paintings for her upcoming show', 
        #        'Take a break to watch some TV', ...]
        self.daily_req = []
        
        # Daily schedule - full decomposed plan
        # This contains both completed actions with their decompositions
        # and future actions at an hourly level
        # e.g., [['sleeping', 360], 
        #        ['wakes up and ... (wakes up and stretches ...)', 5], ...]
        self.f_daily_schedule = []
        
        # Hourly organization - original non-decomposed schedule
        # e.g., [['sleeping', 360], 
        #        ['wakes up and starts her morning routine', 120], ...]
        self.f_daily_schedule_hourly_org = []
        
        # CURRENT ACTION 
        # Address where action takes place: "{world}:{sector}:{arena}:{game_objects}"
        # e.g., "dolores double studio:double studio:bedroom 1:bed"
        self.act_address = None
        # When the action started (datetime)
        self.act_start_time = None
        # Duration in minutes
        self.act_duration = None
        # Text description of action
        self.act_description = None
        # Emoji representation
        self.act_pronunciatio = None
        # Event triple (subject, predicate, object)
        self.act_event = (self.name, None, None)

        # OBJECT ACTION (for the object being used in the action)
        # Text description of object state
        self.act_obj_description = None
        # Emoji representation of object state
        self.act_obj_pronunciatio = None
        # Event triple for object
        self.act_obj_event = (self.name, None, None)

        # CHAT STATE
        # Name of persona currently chatting with
        self.chatting_with = None
        # Conversation history as list of [speaker, text] pairs
        self.chat = None
        # Buffer for tracking nearby personas available to chat
        self.chatting_with_buffer = dict()
        # When chat is scheduled to end
        self.chatting_end_time = None

        # MOVEMENT/PATHFINDING
        # Whether path to destination has been calculated
        self.act_path_set = False
        # List of (x,y) coordinates for movement
        self.planned_path = []

        # Load saved state if provided
        if scratch_memory:
            # Load from bootstrap file
            scratch_load = scratch_memory
            self.vision_r = scratch_load["vision_r"]
            self.att_bandwidth = scratch_load["att_bandwidth"]
            self.retention = scratch_load["retention"]

            if scratch_load["curr_time"]:
                self.curr_time = datetime.datetime.strptime(scratch_load["curr_time"],
                                                          "%B %d, %Y, %H:%M:%S")
            else:
                self.curr_time = None
            self.curr_tile = scratch_load["curr_tile"]
            self.daily_plan_req = scratch_load["daily_plan_req"]

            self.name = scratch_load["name"]
            self.first_name = scratch_load["first_name"]
            self.last_name = scratch_load["last_name"]
            self.age = scratch_load["age"]
            self.innate = scratch_load["innate"]
            self.learned = scratch_load["learned"]
            self.currently = scratch_load["currently"]
            self.lifestyle = scratch_load["lifestyle"]
            self.living_area = scratch_load["living_area"]

            self.concept_forget = scratch_load["concept_forget"]
            self.daily_reflection_time = scratch_load["daily_reflection_time"]
            self.daily_reflection_size = scratch_load["daily_reflection_size"]
            self.overlap_reflect_th = scratch_load["overlap_reflect_th"]
            self.kw_strg_event_reflect_th = scratch_load["kw_strg_event_reflect_th"]
            self.kw_strg_thought_reflect_th = scratch_load["kw_strg_thought_reflect_th"]

            self.recency_w = scratch_load["recency_w"]
            self.relevance_w = scratch_load["relevance_w"]
            self.importance_w = scratch_load["importance_w"]
            self.recency_decay = scratch_load["recency_decay"]
            self.importance_trigger_max = scratch_load["importance_trigger_max"]
            self.importance_trigger_curr = scratch_load["importance_trigger_curr"]
            self.importance_ele_n = scratch_load["importance_ele_n"]
            self.thought_count = scratch_load["thought_count"]

            self.daily_req = scratch_load["daily_req"]
            self.f_daily_schedule = scratch_load["f_daily_schedule"]
            self.f_daily_schedule_hourly_org = scratch_load["f_daily_schedule_hourly_org"]

            self.act_address = scratch_load["act_address"]
            if scratch_load["act_start_time"]:
                self.act_start_time = datetime.datetime.strptime(
                                                  scratch_load["act_start_time"],
                                                  "%B %d, %Y, %H:%M:%S")
            else:
                self.act_start_time = None
            self.act_duration = scratch_load["act_duration"]
            self.act_description = scratch_load["act_description"]
            self.act_pronunciatio = scratch_load["act_pronunciatio"]
            self.act_event = tuple(scratch_load["act_event"])

            self.act_obj_description = scratch_load["act_obj_description"]
            self.act_obj_pronunciatio = scratch_load["act_obj_pronunciatio"]
            self.act_obj_event = tuple(scratch_load["act_obj_event"])

            self.chatting_with = scratch_load["chatting_with"]
            self.chat = scratch_load["chat"]
            self.chatting_with_buffer = scratch_load["chatting_with_buffer"]
            if scratch_load["chatting_end_time"]:
                self.chatting_end_time = datetime.datetime.strptime(
                                                scratch_load["chatting_end_time"],
                                                "%B %d, %Y, %H:%M:%S")
            else:
                self.chatting_end_time = None

            self.act_path_set = scratch_load["act_path_set"]
            self.planned_path = scratch_load["planned_path"]

    def to_dict(self):
        """
        Save persona's scratch memory to disk.

        Args:
            out_json: File path where memory will be saved
        """
        scratch = dict()
        scratch["vision_r"] = self.vision_r
        scratch["att_bandwidth"] = self.att_bandwidth
        scratch["retention"] = self.retention

        scratch["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
        scratch["curr_tile"] = self.curr_tile
        scratch["daily_plan_req"] = self.daily_plan_req

        scratch["name"] = self.name
        scratch["first_name"] = self.first_name
        scratch["last_name"] = self.last_name
        scratch["age"] = self.age
        scratch["innate"] = self.innate
        scratch["learned"] = self.learned
        scratch["currently"] = self.currently
        scratch["lifestyle"] = self.lifestyle
        scratch["living_area"] = self.living_area

        scratch["concept_forget"] = self.concept_forget
        scratch["daily_reflection_time"] = self.daily_reflection_time
        scratch["daily_reflection_size"] = self.daily_reflection_size
        scratch["overlap_reflect_th"] = self.overlap_reflect_th
        scratch["kw_strg_event_reflect_th"] = self.kw_strg_event_reflect_th
        scratch["kw_strg_thought_reflect_th"] = self.kw_strg_thought_reflect_th

        scratch["recency_w"] = self.recency_w
        scratch["relevance_w"] = self.relevance_w
        scratch["importance_w"] = self.importance_w
        scratch["recency_decay"] = self.recency_decay
        scratch["importance_trigger_max"] = self.importance_trigger_max
        scratch["importance_trigger_curr"] = self.importance_trigger_curr
        scratch["importance_ele_n"] = self.importance_ele_n
        scratch["thought_count"] = self.thought_count

        scratch["daily_req"] = self.daily_req
        scratch["f_daily_schedule"] = self.f_daily_schedule
        scratch["f_daily_schedule_hourly_org"] = self.f_daily_schedule_hourly_org

        scratch["act_address"] = self.act_address
        scratch["act_start_time"] = (self.act_start_time
                                         .strftime("%B %d, %Y, %H:%M:%S"))
        scratch["act_duration"] = self.act_duration
        scratch["act_description"] = self.act_description
        scratch["act_pronunciatio"] = self.act_pronunciatio
        scratch["act_event"] = self.act_event

        scratch["act_obj_description"] = self.act_obj_description
        scratch["act_obj_pronunciatio"] = self.act_obj_pronunciatio
        scratch["act_obj_event"] = self.act_obj_event

        scratch["chatting_with"] = self.chatting_with
        scratch["chat"] = self.chat
        scratch["chatting_with_buffer"] = self.chatting_with_buffer
        if self.chatting_end_time:
            scratch["chatting_end_time"] = (self.chatting_end_time
                                            .strftime("%B %d, %Y, %H:%M:%S"))
        else:
            scratch["chatting_end_time"] = None

        scratch["act_path_set"] = self.act_path_set
        scratch["planned_path"] = self.planned_path

        return scratch

    def get_f_daily_schedule_index(self, advance=0):
        """
        Get the current index in the daily schedule based on elapsed time.
        
        Finds the position in the decomposed daily schedule (f_daily_schedule)
        based on how much time has elapsed in the day so far.
        
        Args:
            advance: Minutes to look ahead (default 0)
            
        Returns:
            Integer index into f_daily_schedule
        """
        # Calculate minutes elapsed in the day
        today_min_elapsed = self.curr_time.hour * 60 + self.curr_time.minute + advance

        # For debugging
        x = 0
        for task, duration in self.f_daily_schedule:
            x += duration
        x = 0
        for task, duration in self.f_daily_schedule_hourly_org:
            x += duration

        # Find the current schedule entry based on elapsed time
        curr_index = 0
        elapsed = 0
        for task, duration in self.f_daily_schedule:
            elapsed += duration
            if elapsed > today_min_elapsed:
                return curr_index
            curr_index += 1

        return curr_index

    def get_f_daily_schedule_hourly_org_index(self, advance=0):
        """
        Get the current index in the hourly schedule based on elapsed time.
        
        Similar to get_f_daily_schedule_index but works with the original
        hourly schedule format (before decomposition).
        
        Args:
            advance: Minutes to look ahead (default 0)
            
        Returns:
            Integer index into f_daily_schedule_hourly_org
        """
        # Calculate minutes elapsed in the day
        today_min_elapsed = self.curr_time.hour * 60 + self.curr_time.minute + advance
        
        # Find the current schedule entry based on elapsed time
        curr_index = 0
        elapsed = 0
        for task, duration in self.f_daily_schedule_hourly_org:
            elapsed += duration
            if elapsed > today_min_elapsed:
                return curr_index
            curr_index += 1
        return curr_index

    def get_str_iss(self):
        """
        Get the Identity Stable Set (ISS) string representation.
        
        ISS is the core identity information that defines the persona,
        used in many prompts that need persona context.
        
        Returns:
            String containing key persona details
        """
        commonset = ""
        commonset += f"Name: {self.name}\n"
        commonset += f"Age: {self.age}\n"
        commonset += f"Innate traits: {self.innate}\n"
        commonset += f"Learned traits: {self.learned}\n"
        commonset += f"Currently: {self.currently}\n"
        commonset += f"Lifestyle: {self.lifestyle}\n"
        commonset += f"Daily plan requirement: {self.daily_plan_req}\n"
        commonset += f"Current Date: {self.curr_time.strftime('%A %B %d')}\n"
        return commonset

    def get_str_name(self):
        """Get the full name as a string."""
        return self.name

    def get_str_firstname(self):
        """Get the first name as a string."""
        return self.first_name

    def get_str_lastname(self):
        """Get the last name as a string."""
        return self.last_name

    def get_str_age(self):
        """Get the age as a string."""
        return str(self.age)

    def get_str_innate(self):
        """Get the innate traits as a string."""
        return self.innate

    def get_str_learned(self):
        """Get the learned traits as a string."""
        return self.learned

    def get_str_currently(self):
        """Get the current status as a string."""
        return self.currently

    def get_str_lifestyle(self):
        """Get the lifestyle description as a string."""
        return self.lifestyle

    def get_str_daily_plan_req(self):
        """Get the daily plan requirements as a string."""
        return self.daily_plan_req

    def get_str_curr_date_str(self):
        """Get the current date as a formatted string."""
        return self.curr_time.strftime("%A %B %d")

    def get_curr_event(self):
        """
        Get the current event triple.
        
        Returns:
            Tuple of (subject, predicate, object)
        """
        if not self.act_address:
            return (self.name, None, None)
        else:
            return self.act_event

    def get_curr_event_and_desc(self):
        """
        Get the current event triple with description.
        
        Returns:
            Tuple of (subject, predicate, object, description)
        """
        if not self.act_address:
            return (self.name, None, None, None)
        else:
            return (self.act_event[0],
                  self.act_event[1],
                  self.act_event[2],
                  self.act_description)

    def get_curr_obj_event_and_desc(self):
        """
        Get the current object event triple with description.
        
        Returns:
            Tuple of (subject, predicate, object, description)
        """
        if not self.act_address:
            return ("", None, None, None)
        else:
            return (self.act_address,
                  self.act_obj_event[1],
                  self.act_obj_event[2],
                  self.act_obj_description)

    def add_new_action(self,
                     action_address,
                     action_duration,
                     action_description,
                     action_pronunciatio,
                     action_event,
                     chatting_with,
                     chat,
                     chatting_with_buffer,
                     chatting_end_time,
                     act_obj_description,
                     act_obj_pronunciatio,
                     act_obj_event,
                     act_start_time=None):
        """
        Set a new current action for the persona.
        
        Args:
            action_address: Location string
            action_duration: Duration in minutes
            action_description: Text description
            action_pronunciatio: Emoji representation
            action_event: Triple of (subject, predicate, object)
            chatting_with: Name of chat partner (or None)
            chat: List of conversation turns
            chatting_with_buffer: Dict of nearby personas
            chatting_end_time: When chat should end
            act_obj_description: Description of object state
            act_obj_pronunciatio: Emoji for object state
            act_obj_event: Triple for object state
            act_start_time: Start time (defaults to current time)
        """
        self.act_address = action_address
        self.act_duration = action_duration
        self.act_description = action_description
        self.act_pronunciatio = action_pronunciatio
        self.act_event = action_event

        self.chatting_with = chatting_with
        self.chat = chat
        if chatting_with_buffer:
            self.chatting_with_buffer.update(chatting_with_buffer)
        self.chatting_end_time = chatting_end_time

        self.act_obj_description = act_obj_description
        self.act_obj_pronunciatio = act_obj_pronunciatio
        self.act_obj_event = act_obj_event
        
        self.act_start_time = self.curr_time
        
        self.act_path_set = False

    def act_time_str(self):
        """
        Get a string representation of the action start time.
        
        Returns:
            Time string like "14:05 P.M."
        """
        return self.act_start_time.strftime("%H:%M %p")

    def act_check_finished(self):
        """
        Check if the current action has finished based on time.
        
        Returns:
            True if action has finished, False if still ongoing
        """
        if not self.act_address:
            return True
            
        if self.chatting_with:
            end_time = self.chatting_end_time
        else:
            x = self.act_start_time
            if x.second != 0:
                x = x.replace(second=0)
                x = (x + datetime.timedelta(minutes=1))
            end_time = (x + datetime.timedelta(minutes=self.act_duration))

        return end_time.strftime("%H:%M:%S") == self.curr_time.strftime("%H:%M:%S")

    def act_summarize(self):
        """
        Summarize the current action as a dictionary.
        
        Returns:
            Dict with action details
        """
        exp = dict()
        exp["persona"] = self.name
        exp["address"] = self.act_address
        exp["start_datetime"] = self.act_start_time
        exp["duration"] = self.act_duration
        exp["description"] = self.act_description
        exp["pronunciatio"] = self.act_pronunciatio
        return exp

    def act_summary_str(self):
        """
        Get a human-readable string summary of the current action.
        
        Returns:
            Formatted string describing the action
        """
        start_datetime_str = self.act_start_time.strftime("%A %B %d -- %H:%M %p")
        ret = f"[{start_datetime_str}]\n"
        ret += f"Activity: {self.name} is {self.act_description}\n"
        ret += f"Address: {self.act_address}\n"
        ret += f"Duration in minutes (e.g., x min): {str(self.act_duration)} min\n"
        return ret

    def get_str_daily_schedule_summary(self):
        """
        Get a string summary of the decomposed daily schedule.
        
        Returns:
            Formatted string with time and activity
        """
        ret = ""
        curr_min_sum = 0
        for row in self.f_daily_schedule:
            curr_min_sum += row[1]
            hour = int(curr_min_sum/60)
            minute = curr_min_sum%60
            ret += f"{hour:02}:{minute:02} || {row[0]}\n"
        return ret

    def get_str_daily_schedule_hourly_org_summary(self):
        """
        Get a string summary of the original hourly schedule.
        
        Returns:
            Formatted string with time and activity
        """
        ret = ""
        curr_min_sum = 0
        for row in self.f_daily_schedule_hourly_org:
            curr_min_sum += row[1]
            hour = int(curr_min_sum/60)
            minute = curr_min_sum%60
            ret += f"{hour:02}:{minute:02} || {row[0]}\n"
        return ret
    

if __name__ == "__main__":
    from pathlib import Path
    repo_path = Path(__file__).parent.parent.parent.parent / "test_data" / "July1_the_ville_isabella_maria_klaus-step-3-19" / "personas" / "Isabella Rodriguez" / "bootstrap_memory" / "scratch.json"

    # make a scratch from the file
    scratch = Scratch(repo_path)
    print(scratch.get_str_daily_schedule_summary())

    def test_scratch_creation():
        """Test creating a new Scratch instance and saving it"""
        # Create a new Scratch without loading from file
        scratch = Scratch("nonexistent_file.json")
        
        # Set basic properties
        scratch.name = "Test Agent"
        scratch.first_name = "Test"
        scratch.last_name = "Agent"
        scratch.age = 30
        scratch.curr_time = datetime.datetime.now()
        scratch.curr_tile = (10, 10)
        scratch.daily_plan_req = "Complete daily tasks and rest"
        scratch.innate = "Curious, Creative"
        scratch.learned = "Programming, Writing"
        scratch.currently = "Testing the Scratch class"
        scratch.lifestyle = "Tech worker who enjoys learning"
        scratch.living_area = "Small apartment in the city"
        
        # Set up a simple daily schedule
        scratch.f_daily_schedule = [
            ["sleeping", 360],
            ["waking up and getting ready", 60],
            ["having breakfast", 30],
            ["working", 240],
            ["lunch break", 60],
            ["working", 240],
            ["exercise", 60],
            ["dinner", 60],
            ["relaxing", 120],
            ["preparing for bed", 60],
            ["sleeping", 120]
        ]
        
        scratch.f_daily_schedule_hourly_org = [
            ["sleeping", 360],
            ["morning routine", 90],
            ["working", 240],
            ["lunch", 60],
            ["working", 240],
            ["evening activities", 240],
            ["sleeping", 120]
        ]
        
        # Set current action
        scratch.act_address = "home:living_room:desk"
        scratch.act_start_time = scratch.curr_time
        scratch.act_duration = 60
        scratch.act_description = "testing the Scratch class implementation"
        scratch.act_pronunciatio = "🧪"
        scratch.act_event = (scratch.name, "is", "testing code")
        
        # Set object action
        scratch.act_obj_description = "laptop running tests"
        scratch.act_obj_pronunciatio = "💻"
        scratch.act_obj_event = (scratch.name, "using", "laptop")
        
        # Save to temporary file
        temp_file = "temp_scratch.json"
        scratch.save(temp_file)
        print(f"Saved scratch to {temp_file}")
        
        return temp_file

    def test_scratch_loading(file_path):
        """Test loading a Scratch instance from file"""
        # Load the scratch from file
        loaded_scratch = Scratch(file_path)
        
        # Print some key information to verify loading
        print("\nLoaded Scratch Info:")
        print(f"Name: {loaded_scratch.name}")
        print(f"Current Time: {loaded_scratch.curr_time}")
        print(f"Current Action: {loaded_scratch.act_description}")
        
        # Test some of the utility methods
        print("\nIdentity Stable Set:")
        print(loaded_scratch.get_str_iss())
        
        print("\nDaily Schedule Summary:")
        print(loaded_scratch.get_str_daily_schedule_summary())
        
        print("\nHourly Schedule Summary:")
        print(loaded_scratch.get_str_daily_schedule_hourly_org_summary())
        
        print("\nAction Summary:")
        print(loaded_scratch.act_summary_str())
        
        # Test the schedule index functions
        current_index = loaded_scratch.get_f_daily_schedule_index()
        hourly_index = loaded_scratch.get_f_daily_schedule_hourly_org_index()
        print(f"\nCurrent schedule index: {current_index}")
        print(f"Current hourly schedule index: {hourly_index}")
        
        # Test if current action is finished
        is_finished = loaded_scratch.act_check_finished()
        print(f"\nIs current action finished? {is_finished}")
        
        return loaded_scratch

    def test_new_action(scratch):
        """Test adding a new action"""
        # Add a new action
        scratch.add_new_action(
            action_address="home:kitchen:stove",
            action_duration=30,
            action_description="cooking dinner",
            action_pronunciatio="🍳",
            action_event=(scratch.name, "is", "cooking"),
            chatting_with=None,
            chat=None,
            chatting_with_buffer=None,
            chatting_end_time=None,
            act_obj_description="stove with pot boiling",
            act_obj_pronunciatio="🔥",
            act_obj_event=(scratch.name, "using", "stove")
        )
        
        print("\nNew Action Added:")
        print(scratch.act_summary_str())
        
        # Advance time to check if action is finished
        scratch.curr_time = scratch.curr_time + datetime.timedelta(minutes=30)
        is_finished = scratch.act_check_finished()
        print(f"Is new action finished after advancing time by 30 min? {is_finished}")
        
        return scratch

    def cleanup(file_path):
        """Clean up temporary files"""
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"\nRemoved temporary file: {file_path}")

    def main():
        """Main test function"""
        print("Testing Scratch class functionality...")
        temp_file = test_scratch_creation()
        loaded_scratch = test_scratch_loading(temp_file)
        updated_scratch = test_new_action(loaded_scratch)
        cleanup(temp_file)
        print("\nAll tests completed!")

    main()