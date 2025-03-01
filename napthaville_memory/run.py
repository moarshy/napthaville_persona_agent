import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Union, Optional

from naptha_sdk.storage.schemas import (
    StorageType,
    CreateStorageRequest
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
        """Get memory by type"""
        try:
            if isinstance(inputs, Dict):
                inputs = GetMemoryInput(**inputs)
            
            memories = []

            if inputs.memory_type in [MemoryType.ALL, MemoryType.SPATIAL]:
                spatial_result = await self.spatial.get()
                if spatial_result["success"] and spatial_result["data"]:
                    memories.append(MemoryData(
                        memory_type=MemoryType.SPATIAL,
                        memory_data=json.dumps(spatial_result["data"])
                    ))

            if inputs.memory_type in [MemoryType.ALL, MemoryType.SCRATCH]:
                scratch_result = await self.scratch.get()
                if scratch_result["success"] and scratch_result["data"]:
                    memories.append(MemoryData(
                        memory_type=MemoryType.SCRATCH,
                        memory_data=json.dumps(scratch_result["data"])
                    ))

            if inputs.memory_type in [MemoryType.ALL, MemoryType.ASSOCIATIVE]:
                try:
                    embeddings = await self.associative.get_embeddings()
                    nodes = await self.associative.get_nodes()
                    kw_strength = await self.associative.get_kw_strength()
                    
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
                                memory_data=json.dumps(data)
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
                                memory_data=json.dumps(data)
                            ))
                except Exception as e:
                    logger.error(f"Error processing associative memory: {str(e)}")
                    raise

            return OutputSchema(success=True, data=memories)

        except Exception as e:
            logger.error(f"Error getting memory: {str(e)}")
            return OutputSchema(success=False, error=str(e))

    async def set_memory(self, inputs: Union[Dict, SetMemoryInput]) -> OutputSchema:
        """Set memory by type"""
        try:
            if isinstance(inputs, Dict):
                inputs = SetMemoryInput(**inputs)

            if inputs.memory_type == MemoryType.SPATIAL:
                if inputs.operation == OperationType.ADD:
                    result = await self.spatial.add(inputs.data)
                else:
                    result = await self.spatial.update(inputs.data)

            elif inputs.memory_type == MemoryType.SCRATCH:
                if inputs.operation == OperationType.ADD:
                    result = await self.scratch.add(inputs.data)
                else:
                    result = await self.scratch.update(inputs.data)

            elif inputs.memory_type == MemoryType.ASSOCIATIVE:
                if not inputs.subtype:
                    return OutputSchema(success=False, error="Missing subtype for associative memory")

                if inputs.subtype == AssociativeSubType.EMBEDDINGS:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_embedding(inputs.data)
                    else:
                        result = await self.associative.update_embedding(inputs.data)
                
                elif inputs.subtype == AssociativeSubType.NODES:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_node(inputs.data)
                    else:
                        result = await self.associative.update_node(inputs.data)
                
                elif inputs.subtype == AssociativeSubType.KW_STRENGTH:
                    if inputs.operation == OperationType.ADD:
                        result = await self.associative.add_kw_strength(inputs.data)
                    else:
                        result = await self.associative.update_kw_strength(inputs.data)
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
        
        # Test spatial memory
        logger.info("\nTesting spatial memory...")
        try:
            with open(test_data_dir / "spatial_memory.json") as f:
                spatial_data = json.load(f)
            
            # Add spatial memory
            spatial_add = {
                "inputs": {
                    "function_name": "set_memory",
                    "function_input_data": {
                        "memory_type": "spatial",
                        "operation": "add",
                        "data": json.dumps(spatial_data)
                    }
                },
                "deployment": deployment
            }
            result = await run(spatial_add)
            logger.info(f"Add spatial memory result: {result}")

            # Get spatial memory
            spatial_get = {
                "inputs": {
                    "function_name": "get_memory",
                    "function_input_data": {
                        "memory_type": "spatial"
                    }
                },
                "deployment": deployment
            }
            result = await run(spatial_get)
            logger.info(f"Get spatial memory result: {result}")
        except Exception as e:
            logger.error(f"Error in spatial memory test: {e}")

        # # Test scratch memory
        # time.sleep(5)
        # logger.info("\nTesting scratch memory...")
        # try:
        #     with open(test_data_dir / "scratch.json") as f:
        #         scratch_data = json.load(f)
            
        #     # Add scratch memory
        #     scratch_add = {
        #         "inputs": {
        #             "function_name": "set_memory",
        #             "function_input_data": {
        #                 "memory_type": "scratch",
        #                 "operation": "add",
        #                 "data": json.dumps(scratch_data)
        #             }
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(scratch_add)
        #     logger.info(f"Add scratch memory result: {result}")

        #     # Get scratch memory
        #     scratch_get = {
        #         "inputs": {
        #             "function_name": "get_memory",
        #             "function_input_data": {
        #                 "memory_type": "scratch"
        #             }
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(scratch_get)
        #     logger.info(f"Get scratch memory result: {result}")
        # except Exception as e:
        #     logger.error(f"Error in scratch memory test: {e}")

        # # Test associative memory
        # time.sleep(5)
        # logger.info("\nTesting associative memory...")
        # for subtype in ["embeddings", "nodes", "kw_strength"]:
        #     try:
        #         with open(test_data_dir / "associative_memory" / f"{subtype}.json") as f:
        #             assoc_data = json.load(f)
                
        #         # Add associative memory
        #         assoc_add = {
        #             "inputs": {
        #                 "function_name": "set_memory",
        #                 "function_input_data": {
        #                     "memory_type": "associative",
        #                     "subtype": subtype,
        #                     "operation": "add",
        #                     "data": json.dumps(assoc_data)
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(assoc_add)
        #         logger.info(f"Add associative memory ({subtype}) result: {result}")

        #         # Get associative memory
        #         assoc_get = {
        #             "inputs": {
        #                 "function_name": "get_memory",
        #                 "function_input_data": {
        #                     "memory_type": "associative",
        #                     "subtype": subtype
        #                 }
        #             },
        #             "deployment": deployment
        #         }
        #         result = await run(assoc_get)
        #         logger.info(f"Get associative memory ({subtype}) result: {result}")
        #     except Exception as e:
        #         logger.error(f"Error in associative memory ({subtype}) test: {e}")

        # # Test getting all memory
        # time.sleep(5)
        # logger.info("\nTesting get all memory...")
        # try:
        #     all_get = {
        #         "inputs": {
        #             "function_name": "get_memory",
        #             "function_input_data": {
        #                 "memory_type": "all"
        #             }
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(all_get)
        #     logger.info(f"Get all memory result: {result}")
        # except Exception as e:
        #     logger.error(f"Error in get all memory test: {e}")

        # # Final test: Update all memory types and verify
        # logger.info("\nTesting update all memories with {'updated': 'yes'}...")
        # time.sleep(5)
        # update_data = json.dumps({"updated": "yes"})

        # # Update spatial memory
        # spatial_update = {
        #     "inputs": {
        #         "function_name": "set_memory",
        #         "function_input_data": {
        #             "memory_type": "spatial",
        #             "operation": "update",
        #             "data": update_data
        #         }
        #     },
        #     "deployment": deployment
        # }
        # result = await run(spatial_update)
        # logger.info(f"Update spatial memory result: {result}")

        # # Update scratch memory
        # scratch_update = {
        #     "inputs": {
        #         "function_name": "set_memory",
        #         "function_input_data": {
        #             "memory_type": "scratch",
        #             "operation": "update",
        #             "data": update_data
        #         }
        #     },
        #     "deployment": deployment
        # }
        # result = await run(scratch_update)
        # logger.info(f"Update scratch memory result: {result}")

        # # Update associative memory components
        # for subtype in ["embeddings", "nodes", "kw_strength"]:
        #     assoc_update = {
        #         "inputs": {
        #             "function_name": "set_memory",
        #             "function_input_data": {
        #                 "memory_type": "associative",
        #                 "subtype": subtype,
        #                 "operation": "update",
        #                 "data": update_data
        #             }
        #         },
        #         "deployment": deployment
        #     }
        #     result = await run(assoc_update)
        #     logger.info(f"Update associative memory ({subtype}) result: {result}")

        # # Verify updates by getting all memory
        # logger.info("\nVerifying updates...")
        # verify_get = {
        #     "inputs": {
        #         "function_name": "get_memory",
        #         "function_input_data": {
        #             "memory_type": "all"
        #         }
        #     },
        #     "deployment": deployment
        # }
        # result = await run(verify_get)
        
        # # Pretty print the verification results
        # if result["success"]:
        #     logger.info("Update verification results:")
        #     for memory in result["data"]:
        #         memory_type = memory["memory_type"]
        #         memory_data = json.loads(memory["memory_data"])
        #         logger.info(f"\n{memory_type} memory data:")
        #         logger.info(json.dumps(memory_data, indent=2))
        # else:
        #     logger.error(f"Failed to verify updates: {result['error']}")

        logger.info("\nTests completed.")
    asyncio.run(test())