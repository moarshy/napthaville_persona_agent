################################### wake_up_template ###################################
wake_up_template = """
<INSTRUCTIONS>
- only respond with the wake up hour in AM format (e.g. '7' or '15')
- only respond with the wake up hour, no other text
</INSTRUCTIONS>

{identity_set}

In general, {lifestyle}

{names}'s wake up hour:
"""

################################### daily_planning_template ###################################
daily_planning_template = """
<INSTRUCTIONS>
- only respond with the daily plan in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm)
- only respond with the daily plan, no other text
- please respond with the daily plan in the JSON format
- example output:
```json
[
    "wake up and complete the morning routine at 6:00 am",
    "eat breakfast at 7:00 am",
    "read a book from 8:00 am to 12:00 pm"
]
</INSTRUCTIONS>

{commonset}\n\nIn general, {lifestyle}\nToday is {datetime}. Here is {names}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm): 1) wake up and complete the morning routine at {wake_up_hour}, 2)"
"""
################################### hourly_schedule_template ###################################
hourly_schedule_template = """Hourly schedule format: 
{schedule_format}
===
{commonset}
{prior_schedule}{intermission_str}{intermission2}
{prompt_ending}"""

################################### task_decomp_template ###################################
task_decomp_template = """Describe subtasks in 5 min increments. 
---
{commonset}
{schedule_desc}
In 5 min increments, list the subtasks {name} does when {name_repeat} is {task} from {time_range} (total duration in minutes {duration}): 
1) {persona_name} is"""

################################### action_sector_template ###################################
action_sector_template = """Jane Anderson lives in Oak Hill College Student Dormitory.
Jane Anderson is currently in Oak Hill College that has classroom, library.
Area options: Oak Hill College Student Dormitory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the Area options, verbatim.
For eating dinner, Jane Anderson should go to the following area:
Answer: Hobbs Cafe
---
Sam Kim lives in Sam Kim's house.
Sam Kim is currently in Sam Kim's house that has bedroom, kitchen, bathroom.
Area options: Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the Area options, verbatim.
For taking a walk, Sam Kim should go to the following area:
Answer: Johnson Park
---

{name} lives in {living_sector}.
{name_repeat} is currently in {current_sector} that has {current_areas}.
Area options: {available_sectors}
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the Area options, verbatim.
For {action}, {name2} should go to the following area:
Answer: """

################################### action_arena_template ###################################
action_arena_template = """Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: kitchen, bedroom, bathroom
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: kitchen
---
Tom Watson is in common room in Tom Watson's apartment. 
Tom Watson is going to Hobbs Cafe that has the following areas: cafe
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: cafe
---

{name} is in {current_arena} in {target_sector}
{name2} is going to {target_sector2} that has the following areas: {available_areas}
* Stay in the current area if the activity can be done there. 
* NEVER go into other people's rooms unless necessary.
For {action}, {name3} should go to the following area in {target_sector3}:
Answer: """

################################### action_game_object_template ###################################
action_game_object_template = """Current activity: sleep in bed
Objects available: bed, easel, closet, painting
Pick ONE most relevant object from the objects available:
Answer: bed
---
Current activity: painting
Objects available: easel, closet, sink, microwave
Pick ONE most relevant object from the objects available:
Answer: easel
---
Current activity: cooking
Objects available: stove, sink, fridge, counter
Pick ONE most relevant object from the objects available:
Answer: stove
---
Current activity: watch TV
Objects available: couch, TV, remote, coffee table
Pick ONE most relevant object from the objects available:
Answer: TV
---
Current activity: study
Objects available: desk, computer, chair, bookshelf
Pick ONE most relevant object from the objects available:
Answer: desk
---
Current activity: talk on the phone
Objects available: phone, charger, bed, nightstand
Pick ONE most relevant object from the objects available:
Answer: phone
---
Current activity: {action}
Objects available: {available_objects}
Pick ONE most relevant object from the objects available:
Answer: """

################################### pronunciation_template ###################################
pronunciation_template = """Convert an action description to an emoji (important: use two or less emojis).

Examples:
Action description: sleeping in bed
Emoji: 😴

Action description: eating breakfast
Emoji: 🍳

Action description: taking a shower
Emoji: 🚿

Action description: working on computer
Emoji: 💻

Action description: reading a book
Emoji: 📚

Action description: {action}
Emoji: """

################################### event_triple_template ###################################
event_triple_template = """Task: Turn the input into (subject, predicate, object). 

Input: Sam Johnson is eating breakfast. 
Output: (Sam Johnson, eat, breakfast) 
--- 
Input: Joon Park is brewing coffee.
Output: (Joon Park, brew, coffee)
---
Input: Jane Cook is sleeping. 
Output: (Jane Cook, sleep, None)
---
Input: Michael Bernstein is writing email on a computer. 
Output: (Michael Bernstein, write, email)
---
Input: Percy Liang is teaching students in a classroom. 
Output: (Percy Liang, teach, students)
---
Input: Merrie Morris is running on a treadmill. 
Output: (Merrie Morris, run, treadmill)
---
Input: {name} is {action}. 
Output: ({name2},"""

################################### act_obj_desc_template ###################################
act_obj_desc_template = """Task: We want to understand the state of an object that is being used by someone. 

Example 1:
We want to know about bed's state.
Step 1. Jane Smith is sleeping in the bed.
Step 2. Describe the bed's state: bed is being used for sleeping

Example 2:
We want to know about computer's state.
Step 1. John Doe is working on computer.
Step 2. Describe the computer's state: computer is being operated

Example 3:
We want to know about stove's state.
Step 1. Mary Johnson is cooking on stove.
Step 2. Describe the stove's state: stove is heating up

Let's think step by step.
We want to know about {object_name}'s state.
Step 1. {persona_name} is {action_desc}.
Step 2. Describe the {object_name2}'s state: {object_name3} is"""

################################### act_obj_event_triple_template ###################################
act_obj_event_triple_template = """Task: Turn the input into (subject, predicate, object). 

Input: desk is being used for writing.
Output: (desk, used, writing)
---
Input: bed is supporting a sleeping person.
Output: (bed, support, person)
---
Input: computer is running a program.
Output: (computer, run, program)
---
Input: stove is heating up food.
Output: (stove, heat, food)
---
Input: TV is displaying a show.
Output: (TV, display, show)
---
Input: {object_name} is {object_state}.
Output: ({object_name2},"""

################################### new_decomp_schedule_template ###################################
new_decomp_schedule_template = """Here was {name}'s originally planned schedule from {start_time} to {end_time}. 
{original_plan}

But {name2} unexpectedly ended up {new_event} for {new_event_duration} minutes. Revise {name3}'s schedule from {start_time2} to {end_time2} accordingly (it has to end by {end_time3}). 
The revised schedule:
{new_schedule_init}"""

################################### decide_to_talk_template ###################################
decide_to_talk_template = """Task -- given context, determine whether the subject will initiate a conversation with another.

Context: {context}
Right now, it is {current_time}. {persona_name} and {target_name} last chatted at {last_chat_time} about {last_chat_topic}.
{persona_status}
{target_status}

Question: Would {persona_name2} initiate a conversation with {target_name2}?

Reasoning: Let's think step by step.

Answer in yes or no: """

################################### decide_to_react_template ###################################
decide_to_react_template = '''Task -- given context and options, determine whether the subject should wait or continue with their action.

Context: {context}
Right now, it is {current_time}.
{init_desc}
{target_desc}

Question: Let's think step by step. Of the following options, what should {init_name} do?
Option 1: Wait on {init_act} until {target_name} is done {target_act}
Option 2: Continue on to {init_act} now

Reasoning: Let's think step by step.

Answer in Option 1 or 2: '''

################################### create_conversation_template ###################################
create_conversation_template = '''We have two characters.

Character 1:
{init_iss}

Character 2:
{target_iss}
---
Context:
Here is what {init_name} thinks about {target_name}:
{init_thoughts}

Here is what {target_name} thinks about {init_name}:
{target_thoughts}

Currently, it is {current_time}
-- {init_desc}
-- {target_desc}
{prev_convo}

{init_name} and {target_name} are in {location}. What would they talk about now?

{init_name}: "'''

################################### summarize_conversation_template ###################################
summarize_conversation_template = '''Here is a conversation:
{conversation}

Summarize the conversation above in one concise sentence that captures the main topic and participants. 
Start with "conversing about".

Example summaries:
- conversing about lunch plans for the afternoon between Alice and Bob
- conversing about recent travel experiences and future trip planning
- conversing about a shared work project and upcoming deadlines

Your summary:
conversing about'''
################################### extract_keywords_template ###################################
extract_keywords_template = """Given a text description of an event or a conversation, output CSV (comma-separated values) of factually descriptive and emotive keywords about the description.
Below is the format of the output. 
Description of an event or a conversation: [Provided]
Factually descriptive keywords: [Fill in]
Emotive keywords: [Fill in]
===
Description of an event or a conversation: {description}
Factually descriptive keywords:"""

################################### keyword_to_thoughts_template ###################################
keyword_to_thoughts_template = """The following events/thoughts happened about "{keyword}". 
{events_thoughts}

Here is what {persona_name} thinks about these events/thoughts in one sentence. The sentence needs to be in third person:"""

################################### convo_to_thoughts_template ###################################
convo_to_thoughts_template = """Here is the conversation that happened between {init_persona_name} and {target_persona_name}. 

{convo_str}

Summarize what {init_persona_name_repeat} thought about {fin_target} in one short sentence. The sentence needs to be in third person:"""

################################### event_poignancy_template ###################################
event_poignancy_template = """Here is a brief description of {persona_name}. 
{persona_iss}

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for {persona_name_repeat}.

Event: {event_description}
Rate (return a number between 1 to 10):"""

################################### thought_poignancy_template ###################################
thought_poignancy_template = """Here is a brief description of {persona_name}. 
{persona_iss}

On the scale of 1 to 10, where 1 is purely mundane (e.g., I need to do the dishes, I need to walk the dog) and 10 is extremely significant (e.g., I wish to become a professor, I love Elie), rate the likely significance of the following thought for {persona_name_repeat}.

Thought: {thought_description}
Rate (return a number between 1 to 10):"""

################################### chat_poignancy_template ###################################
chat_poignancy_template = """Here is a brief description of {persona_name}. 
{persona_iss}

On the scale of 1 to 10, where 1 is purely mundane (e.g., routine morning greetings) and 10 is extremely poignant (e.g., a conversation about breaking up, a fight), rate the likely poignancy of the following conversation for {persona_name_repeat}.

Conversation: 
{conversation_description}

Rate (return a number between 1 to 10):"""

################################### focal_pt_template ###################################
focal_pt_template = """{statements}

Given only the information above, what are {n} most salient high-level questions we can answer about the subjects grounded in the statements?
1)"""

################################### insight_and_guidance_template ###################################
insight_and_guidance_template = """Input:
{statements}

What {n} high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))
1."""

################################### summarize_chat_ideas_template ###################################
summarize_chat_ideas_template = """Current Date: {curr_date}

{curr_context}

Currently: {currently}

{statements}
Summarize the most relevant statements above that can inform {persona_name} in his conversation with {target_name}.
"""

################################### summarize_chat_relationship_template ###################################
summarize_chat_relationship_template = """[Statements]
{statements}

Based on the statements above, summarize {persona_name} and {target_name}'s relationship. What do they feel or know about each other?
"""

################################### agent_chat_template ###################################
agent_chat_template = """Character 1: {init_persona_currently}
Character 2: {target_persona_currently}

Past Context: {prior_convo_summary}

Current Context: {curr_context}
Current Location: {curr_location}

(This is what is in {init_persona_name}'s head: {init_summ_idea} Beyond this, {init_persona_name} doesn't necessarily know anything more about {target_persona_name}) 

(This is what is in {target_persona_name}'s head: {target_summ_idea} Beyond this, {target_persona_name} doesn't necessarily know anything more about {init_persona_name}) 

Here is their conversation. 

{init_persona_name}: \""""

################################### summarize_ideas_template ###################################
summarize_ideas_template = """Statements: 
{statements}

An interviewer said to {persona_name}: 
"{question}"

Summarize the Statements that are most relevant to the interviewer's line:
"""

################################### generate_next_convo_line_template ###################################
generate_next_convo_line_template = """Here is some basic information about {persona_name}.
{persona_iss}

=== 
Following is a conversation between {persona_name} and {interlocutor_desc}. 

{prev_convo}

(Note -- This is the only information that {persona_name} has: {retrieved_summary})

{persona_name}: \""""

################################### whisper_inner_thought_template ###################################
whisper_inner_thought_template = """Translate the following thought into a statement about {persona_name}. 

Thought: "{whisper}"
Statement: \""""

################################### planning_thought_on_convo_template ###################################
planning_thought_on_convo_template = """[Conversation]
{all_utt}

Write down if there is anything from the conversation that {persona_name} need to remember for her planning, from {persona_name}'s perspective, in a full sentence.

"{persona_name}"""

################################### memo_on_convo_template ###################################
memo_on_convo_template = """[Conversation]
{all_utt}

Write down if there is anything from the conversation that {persona_name} might have found interesting from {persona_name}'s perspective, in a full sentence. 

"{persona_name}"""

################################### iterative_convo_template ###################################
iterative_convo_template = """Context for the task: 

PART 1. 
{init_iss}

Here is the memory that is in {init_persona_name}'s head: 
{retrieved_str}

PART 2. 
Past Context: 
{prev_convo_ins ert}

Current Location: {curr_location}

Current Context: 
{curr_context}

{init_persona_name} and {target_persona_name} are chatting. Here is their conversation so far: 
{convo_str}

---
Task: Given the above, what should {init_persona_name} say to {target_persona_name} next in the conversation? And did it end the conversation?

Output format: Output a json of the following format: 
{{
"{init_persona_name}": "<{init_persona_name}'s utterance>",
"Did the conversation end with {init_persona_name}'s utterance?": "<json Boolean>"
}}"""

################################### summarize_chat_relationship_template ###################################
summarize_chat_relationship_template = """[Statements]
{statements}

Based on the statements above, summarize {persona_name} and {target_name}'s relationship. What do they feel or know about each other?
"""