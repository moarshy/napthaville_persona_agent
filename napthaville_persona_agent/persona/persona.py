import math
import sys
import datetime
import random
import json
import logging
from typing import List
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from napthaville_persona_agent.persona.memory.spatial import MemoryTree
from napthaville_persona_agent.persona.memory.associative_memory import AssociativeMemory
from napthaville_persona_agent.persona.memory.scratch import Scratch

from napthaville_persona_agent.persona.cognitive_modules.perceive import perceive
from napthaville_persona_agent.persona.cognitive_modules.retrieve import retrieve
from napthaville_persona_agent.persona.cognitive_modules.plan import plan
from napthaville_persona_agent.persona.cognitive_modules.reflect import reflect
from napthaville_persona_agent.persona.cognitive_modules.execute import execute
# from napthaville_persona_agent.persona.cognitive_modules.converse import open_convo_session

from napthaville_memory.run import run as memory_run
from napthaville_environment.run import run as maze_run


class Persona:
    def __init__(self, name, memory_deployment, maze_deployment):
        self.name = name
        self.memory_deployment = memory_deployment
        self.maze_deployment = maze_deployment
        self.s_mem = None
        self.a_mem = None
        self.scratch = None
        self.perceive_maze_data = None

    async def load_memory(self):
        """
        Load memory from the deployment.
        """
        # get spatial memory
        spatial_get = {
            "inputs": {
                "function_name": "get_memory",
                "function_input_data": {"memory_type": "spatial", "persona_name": self.name},
            },
            "deployment": self.memory_deployment
        }
        spatial_memory = await memory_run(spatial_get)
        spatial_memory = json.loads(json.loads(spatial_memory['data'][0]['memory_data'])[0]['data'])
        self.s_mem = MemoryTree(spatial_memory)

        # get scratch memory
        scratch_get = {
            "inputs": {
                "function_name": "get_memory",
                "function_input_data": {"memory_type": "scratch", "persona_name": self.name},
            },
            "deployment": self.memory_deployment
        }
        scratch_memory = await memory_run(scratch_get)
        scratch_memory = json.loads(json.loads(scratch_memory['data'][0]['memory_data'])[0]['data'])
        self.scratch = Scratch(scratch_memory)

        # get associative memory
        _a_mem = {}
        for subtype in ["embeddings", "nodes", "kw_strength"]:
            get = {
                "inputs": {
                    "function_name": "get_memory",
                    "function_input_data": {"memory_type": "associative", "subtype": subtype, "persona_name": self.name},
                },
                "deployment": self.memory_deployment
            }
            a_mem = await memory_run(get)
            a_mem = json.loads(json.loads(a_mem['data'][0]['memory_data'])[0]['data'])
            _a_mem[subtype] = a_mem
        self.a_mem = AssociativeMemory(_a_mem)

    async def save_memory(self):
        """
        Save memory to the deployment.
        """
        try:
            # save spatial memory
            spatial_update = {
                "inputs": {
                    "function_name": "set_memory",
                    "function_input_data": {
                        "memory_type": "spatial",
                        "operation": "update",
                        "data": json.dumps(self.s_mem.tree),
                        "persona_name": self.name
                    },
                },
                "deployment": self.memory_deployment
            }
            await memory_run(spatial_update)
        except Exception as e:
            logger.error(f"Error saving spatial memory: {e}")
            raise e

        try:
            # save scratch memory
            scratch_update = {
                "inputs": {
                "function_name": "set_memory",
                "function_input_data": {
                        "memory_type": "scratch",
                        "operation": "update",
                        "data": json.dumps(self.scratch.to_dict()),
                        "persona_name": self.name
                    },
                },
                "deployment": self.memory_deployment
            }
            await memory_run(scratch_update)
        except Exception as e:
            logger.error(f"Error saving scratch memory: {e}")
            raise e

        try:
            # save associative memory
            all_mems = self.a_mem.to_dict()
            for mem_type in ["nodes", "kw_strength", "embeddings"]:
                mem_data = all_mems[mem_type]
                associative_update = {
                    "inputs": {
                        "function_name": "set_memory",
                        "function_input_data": {
                            "memory_type": "associative",
                            "operation": "update",
                            "data": json.dumps(mem_data),
                            "persona_name": self.name
                            },
                    },
                    "deployment": self.memory_deployment
                }
                await memory_run(associative_update)
        except Exception as e:
            logger.error(f"Error saving associative memory: {e}")
            raise e

    async def perceive_maze(self):
        """
        Maze data for perceive. 
        """
        perceive_maze = {
            "inputs": {
                "function_name": "perceive",
                "function_input_data": {
                    "curr_tile": self.scratch.curr_tile,
                    "vision_r": self.scratch.vision_r
                },
            },
            "deployment": self.maze_deployment
        }
        maze_data = await maze_run(perceive_maze)
        return maze_data

    async def perceive(self):
        maze_data = await self.perceive_maze()
        self.perceive_maze_data = maze_data
        return perceive(self, maze_data)

    def retrieve(self, perceived):
        return retrieve(self, perceived)

    async def plan(self, personas, new_day, retrieved):
        return await plan(self, self.perceive_maze_data, personas, new_day, retrieved)

    async def execute(self, persona_names: List[str], plan: str):
        # get maze data for execute
        execute_maze = {
            "inputs": {
                "function_name": "execute",
                "function_input_data": {}
            },
            "deployment": self.maze_deployment
        }
        execute_maze_data = await maze_run(execute_maze)

        # get personas details for execute curr_tile
        personas = {}
        for persona_name in persona_names:
            get_scratch = {
                "inputs": {
                    "function_name": "get_memory",
                    "function_input_data": {
                        "memory_type": "scratch",
                        "persona_name": persona_name
                    }
                },
                "deployment": self.memory_deployment
            }
            scratch_memory = await memory_run(get_scratch)
            scratch_memory = json.loads(json.loads(scratch_memory['data'][0]['memory_data'])[0]['data'])
            personas[persona_name] = scratch_memory
        return execute(self, execute_maze_data, personas, plan)

    def reflect(self):
        """
        Review the persona's memory and create new thoughts based on it.
        """
        reflect(self)

    async def move(self, persona_names, curr_tile, curr_time):
        self.scratch.curr_tile = curr_tile

        new_day = False
        if not self.scratch.curr_time:
            new_day = "First day"
        elif (self.scratch.curr_time.strftime('%A %B %d') != 
              curr_time.strftime('%A %B %d')):
            new_day = "New day"
        self.scratch.curr_time = curr_time

        perceived = await self.perceive()
        retrieved = self.retrieve(perceived)
        plan = await self.plan(persona_names, new_day, retrieved)
        print(f"Plan: {plan}")
        self.reflect()
        return await self.execute(persona_names, plan)