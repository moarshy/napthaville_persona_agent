from napthaville_chat.persona.persona import Persona
from napthaville_chat.schemas import InputSchema
from napthaville_chat.cognitive_modules.plan import _chat_react, _wait_react

async def run(module_run):
    input_schema = InputSchema(**module_run["inputs"])
    init_persona = Persona(input_schema.init_persona_name)
    target_persona = Persona(input_schema.target_persona_name)
    maze_data = input_schema.maze_data
    reaction_mode = input_schema.reaction_mode

    # Load memories
    await init_persona.load_memory()
    await target_persona.load_memory()
    
    # react
    if reaction_mode == "chat with":
        await _chat_react(maze_data, reaction_mode, init_persona, target_persona)
    elif reaction_mode == "wait":
        await _wait_react(init_persona, reaction_mode)

    # save memories
    await init_persona.save_memory()
    await target_persona.save_memory()

    # return
    return {
        "init_persona": init_persona.scratch.to_dict(),
        "target_persona": target_persona.scratch.to_dict(),
    }