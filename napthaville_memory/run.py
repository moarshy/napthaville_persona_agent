import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Union, Optional

from naptha_sdk.storage.schemas import (
    StorageType,
    CreateStorageRequest,
    ReadStorageRequest
)
from naptha_sdk.storage.storage_client import StorageClient
from naptha_sdk.schemas import NodeConfigUser

from napthaville_memory.memory.spatial import Spatial
from napthaville_memory.memory.associative import Associative
from napthaville_memory.memory.scratch import Scratch
from napthaville_memory.schemas import (
    InputSchema, OutputSchema, MemoryData, AssociativeMemoryData,
    MemoryType, AssociativeSubType, OperationType,
    GetMemoryInput, SetMemoryInput, 
)

logger = logging.getLogger(__name__)

class NapthavilleMemory:
    @classmethod
    async def create(cls, deployment: Dict) -> "NapthavilleMemory":
        """Create a new NapthavilleMemory instance asynchronously"""
        memory = cls.__new__(cls)
        await memory.__ainit__(deployment)
        return memory
    
    async def __ainit__(self, deployment: Dict):
        """Async initialization method"""
        self.deployment = deployment
        
        node_config = NodeConfigUser(
            ip=deployment["node"]["ip"],
            user_communication_port=deployment.get("user_communication_port", 7001),
            user_communication_protocol=deployment.get("user_communication_protocol", "http")
        )
            
        self.storage_client = StorageClient(node_config)
        self.spatial = Spatial(self.deployment["config"]["storage_config"], self.storage_client)
        self.associative = Associative(self.deployment["config"]["storage_config"], self.storage_client)
        self.scratch = Scratch(self.deployment["config"]["storage_config"], self.storage_client)

    def __init__(self):
        """Prevent direct instantiation, use create() instead"""
        raise RuntimeError("Please use NapthavilleMemory.create() to instantiate")

    async def init(self, inputs: Optional[Dict] = None) -> OutputSchema:
        """Initialize database tables"""
        try:
            storage_schema = self.deployment["config"]["storage_config"]["storage_schema"]
            table_name = "napthaville_memory"
            schema_def = storage_schema["napthaville_memory"]

            create_table_request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=table_name,
                data={"schema": schema_def}
            )

            await self.storage_client.execute(create_table_request)
            return OutputSchema(success=True)
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return OutputSchema(success=False, error=str(e))

    async def get_memory(self, inputs: Union[Dict, GetMemoryInput]) -> OutputSchema:
        """Get memory by type and optional persona"""
        try:
            if isinstance(inputs, Dict):
                inputs = GetMemoryInput(**inputs)
            
            memories = []
            persona_name = inputs.persona_name if hasattr(inputs, 'persona_name') else None

            if inputs.memory_type in [MemoryType.ALL, MemoryType.SPATIAL]:
                spatial_result = await self.spatial.get(persona_name)
                if spatial_result["success"] and spatial_result["data"]:
                    memories.append(MemoryData(
                        memory_type=MemoryType.SPATIAL,
                        memory_data=json.dumps(spatial_result["data"]),
                        persona_name=persona_name
                    ))

            if inputs.memory_type in [MemoryType.ALL, MemoryType.SCRATCH]:
                scratch_result = await self.scratch.get(persona_name)
                if scratch_result["success"] and scratch_result["data"]:
                    memories.append(MemoryData(
                        memory_type=MemoryType.SCRATCH,
                        memory_data=json.dumps(scratch_result["data"]),
                        persona_name=persona_name
                    ))

            if inputs.memory_type in [MemoryType.ALL, MemoryType.ASSOCIATIVE]:
                try:
                    embeddings = await self.associative.get_embeddings(persona_name)
                    nodes = await self.associative.get_nodes(persona_name)
                    kw_strength = await self.associative.get_kw_strength(persona_name)
                    
                    if all(r["success"] for r in [embeddings, nodes, kw_strength]):
                        if inputs.subtype:
                            if inputs.subtype == AssociativeSubType.EMBEDDINGS:
                                data = embeddings["data"] if embeddings["data"] else []
                            elif inputs.subtype == AssociativeSubType.NODES:
                                data = nodes["data"] if nodes["data"] else []
                            elif inputs.subtype == AssociativeSubType.KW_STRENGTH:
                                data = kw_strength["data"] if kw_strength["data"] else []
                            
                            memories.append(MemoryData(
                                memory_type=MemoryType.ASSOCIATIVE,
                                memory_data=json.dumps(data),
                                persona_name=persona_name
                            ))
                        else:
                            # Get all associative memory types
                            data = {
                                "embeddings": embeddings["data"] if embeddings["data"] else [],
                                "nodes": nodes["data"] if nodes["data"] else [],
                                "kw_strength": kw_strength["data"] if kw_strength["data"] else []
                            }
                            memories.append(MemoryData(
                                memory_type=MemoryType.ASSOCIATIVE,
                                memory_data=json.dumps(data),
                                persona_name=persona_name
                            ))
                except Exception as e:
                    logger.error(f"Error processing associative memory: {str(e)}")
                    raise

            return OutputSchema(success=True, data=memories)

        except Exception as e:
            logger.error(f"Error getting memory: {str(e)}")
            return OutputSchema(success=False, error=str(e))

    async def set_memory(self, inputs: Union[Dict, SetMemoryInput]) -> OutputSchema:
        """Set memory by type and optional persona"""
        try:
            if isinstance(inputs, Dict):
                inputs = SetMemoryInput(**inputs)

            persona_name = inputs.persona_name if hasattr(inputs, 'persona_name') else None

            if inputs.memory_type == MemoryType.SPATIAL:
                if inputs.operation == OperationType.ADD:
                    result = await self.spatial.add(inputs.data, persona_name)
                else:
                    result = await self.spatial.update(inputs.data, persona_name)

            elif inputs.memory_type == MemoryType.SCRATCH:
                if inputs.operation == OperationType.ADD:
                    result = await self.scratch.add(inputs.data, persona_name)
                else:
                    result = await self.scratch.update(inputs.data, persona_name)

            elif inputs.memory_type == MemoryType.ASSOCIATIVE:
                if not inputs.subtype:
                    return OutputSchema(success=False, error="Missing subtype for associative memory")

                if inputs.subtype == AssociativeSubType.EMBEDDINGS:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_embedding(inputs.data, persona_name)
                    else:
                        result = await self.associative.update_embedding(inputs.data, persona_name)
                
                elif inputs.subtype == AssociativeSubType.NODES:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_node(inputs.data, persona_name)
                    else:
                        result = await self.associative.update_node(inputs.data, persona_name)
                
                elif inputs.subtype == AssociativeSubType.KW_STRENGTH:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_kw_strength(inputs.data, persona_name)
                    else:
                        result = await self.associative.update_kw_strength(inputs.data, persona_name)
                else:
                    return OutputSchema(success=False, error="Invalid associative memory subtype")
            else:
                return OutputSchema(success=False, error="Invalid memory type")

            return OutputSchema(
                success=result["success"],
                error=result.get("error"),
                data=result.get("data")
            )

        except Exception as e:
            logger.error(f"Error setting memory: {str(e)}")
            return OutputSchema(success=False, error=str(e))
            
    async def get_personas(self) -> OutputSchema:
        """Get a list of all personas with memories"""
        try:
            # Read all records and extract unique persona names
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.deployment["config"]["storage_config"]["path"]
            )
            result = await self.storage_client.execute(request)
            
            if not result.data:
                return OutputSchema(success=True, data=[])
                
            personas = set()
            for record in result.data:
                # Extract persona name from type field (format: persona_name_memory_type)
                type_value = record.get("type", "")
                parts = type_value.split("_", 1)
                if len(parts) > 1:
                    personas.add(parts[0])
            
            return OutputSchema(success=True, data=list(personas))
            
        except Exception as e:
            logger.error(f"Error getting personas: {str(e)}")
            return OutputSchema(success=False, error=str(e))

async def run(module_run: Dict, *args, **kwargs) -> Dict:
    """Main entry point for the memory module"""
    try:
        module_run_input = InputSchema(**module_run["inputs"])
        memory = await NapthavilleMemory.create(module_run["deployment"])
        
        method = getattr(memory, module_run_input.function_name, None)
        if not method:
            return OutputSchema(
                success=False,
                error=f"Unknown function: {module_run_input.function_name}"
            ).model_dump()

        if asyncio.iscoroutinefunction(method):
            result = await method(module_run_input.function_input_data)
        else:
            result = method(module_run_input.function_input_data)
            
        return result.model_dump()

    except Exception as e:
        logger.error(f"Error in run: {str(e)}")
        return OutputSchema(success=False, error=str(e)).model_dump()

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    file_dir = Path(__file__).parent
    repo_dir = file_dir.parent
    deployment = json.load(open(file_dir / "configs" / "deployment.json"))
    deployment = deployment[0]
    
    async def test():
        """Run test cases using test data"""
        test_data_dir = repo_dir / "test_data/July1_the_ville_isabella_maria_klaus-step-3-19/personas/Isabella Rodriguez/bootstrap_memory"
        
        logger.info("Starting tests...")
        
        # Initialize memory
        init_run = {
            "inputs": {
                "function_name": "init",
                "function_input_data": {}
            },
            "deployment": deployment
        }
        init_result = await run(init_run)
        logger.info(f"Initialization result: {init_result}")
        
        # Test spatial memory for different personas
        logger.info("\nTesting spatial memory for different personas...")
        personas = ["Isabella", "Maria", "Klaus"]
        
        # try:
        #     with open(test_data_dir / "spatial_memory.json") as f:
        #         spatial_data = json.load(f)
            
        #     # Add spatial memory for each persona
        #     for persona in personas:
        #         spatial_add = {
        #             "inputs": {
        #                 "function_name": "set_memory",
        #                 "function_input_data": {
        #                     "memory_type": "spatial",
        #                     "operation": "add",
        #                     "data": json.dumps(spatial_data),
        #                     "persona_name": persona
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(spatial_add)
        #         logger.info(f"Add spatial memory for {persona} result: {result}")

        #     # Get spatial memory for each persona
        #     for persona in personas:
        #         spatial_get = {
        #             "inputs": {
        #                 "function_name": "get_memory",
        #                 "function_input_data": {
        #                     "memory_type": "spatial",
        #                     "persona_name": persona
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(spatial_get)
        #         logger.info(f"Get spatial memory for {persona} result: {result}")
                
        #     # Get all personas with memories
        #     personas_get = {
        #         "inputs": {
        #             "function_name": "get_personas",
        #             "function_input_data": {}
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(personas_get)
        #     logger.info(f"Get all personas result: {result}")
        # except Exception as e:
        #     logger.error(f"Error in spatial memory test: {e}")

        # # Test scratch memory for different personas
        # logger.info("\nTesting scratch memory for different personas...")
        # try:
        #     scratch_data = {"notes": ["Initial scratch note for testing"]}
            
        #     # Add scratch memory for each persona
        #     for persona in personas:
        #         scratch_add = {
        #             "inputs": {
        #                 "function_name": "set_memory",
        #                 "function_input_data": {
        #                     "memory_type": "scratch",
        #                     "operation": "add",
        #                     "data": json.dumps(scratch_data),
        #                     "persona_name": persona
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(scratch_add)
        #         logger.info(f"Add scratch memory for {persona} result: {result}")
                
        #     # Update scratch memory for first persona
        #     updated_scratch_data = {"notes": ["Updated scratch note", "Another note"]}
        #     scratch_update = {
        #         "inputs": {
        #             "function_name": "set_memory",
        #             "function_input_data": {
        #                 "memory_type": "scratch",
        #                 "operation": "update",
        #                 "data": json.dumps(updated_scratch_data),
        #                 "persona_name": personas[0]
        #             }
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(scratch_update)
        #     logger.info(f"Update scratch memory for {personas[0]} result: {result}")
                
        #     # Get scratch memory for each persona
        #     for persona in personas:
        #         scratch_get = {
        #             "inputs": {
        #                 "function_name": "get_memory",
        #                 "function_input_data": {
        #                     "memory_type": "scratch",
        #                     "persona_name": persona
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(scratch_get)
        #         logger.info(f"Get scratch memory for {persona} result: {result}")
        # except Exception as e:
        #     logger.error(f"Error in scratch memory test: {e}")

        # Test associative memory for different personas
        logger.info("\nTesting associative memory for different personas...")
        for subtype in ["embeddings", "nodes", "kw_strength"]:
            try:
                assoc_data = {"sample": f"Test {subtype} data"}
                
                # Add associative memory for each persona
                for persona in personas:
                    assoc_add = {
                        "inputs": {
                            "function_name": "set_memory",
                            "function_input_data": {
                                "memory_type": "associative",
                                "subtype": subtype,
                                "operation": "add",
                                "data": json.dumps(assoc_data),
                                "persona_name": persona
                            }
                        },
                        "deployment": deployment
                    }
                    result = await run(assoc_add)
                    logger.info(f"Add associative memory ({subtype}) for {persona} result: {result}")

                # Get associative memory for each persona
                for persona in personas:
                    assoc_get = {
                        "inputs": {
                            "function_name": "get_memory",
                            "function_input_data": {
                                "memory_type": "associative",
                                "subtype": subtype,
                                "persona_name": persona
                            }
                        },
                        "deployment": deployment
                    }
                    result = await run(assoc_get)
                    logger.info(f"Get associative memory ({subtype}) for {persona} result: {result}")
            except Exception as e:
                logger.error(f"Error in associative memory ({subtype}) test: {e}")

        # Test getting all memory types for a specific persona
        logger.info("\nTesting get all memory types for a specific persona...")
        try:
            all_get = {
                "inputs": {
                    "function_name": "get_memory",
                    "function_input_data": {
                        "memory_type": "all",
                        "persona_name": personas[0]
                    }
                },
                "deployment": deployment
            }
            result = await run(all_get)
            logger.info(f"Get all memory types for {personas[0]} result: {result}")
        except Exception as e:
            logger.error(f"Error in get all memory test: {e}")

        logger.info("\nTests completed.")
    
    asyncio.run(test())