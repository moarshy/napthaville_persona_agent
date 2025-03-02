from pydantic import BaseModel
from typing import Dict

class InputSchema(BaseModel):
    init_persona_name: str
    target_persona_name: str
    reaction_mode: str
    maze_data: Dict