import json
import logging
from typing import Dict
from naptha_sdk.storage.schemas import (
    StorageType,
    CreateStorageRequest,
    ReadStorageRequest,
    UpdateStorageRequest
)
from naptha_sdk.storage.storage_client import StorageClient

logger = logging.getLogger(__name__)

class Scratch:
    def __init__(self, storage_config: Dict, storage_client: StorageClient):
        self.storage_client = storage_client
        self.table_name = storage_config["path"]
        self.base_memory_type = "scratch"

    async def add(self, data: str, persona_name: str = None) -> Dict:
        """Add scratch memory for a specific persona"""
        try:
            memory_type = self._get_memory_type(persona_name)
            request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "type": memory_type,
                    "data": data
                }}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error adding scratch memory: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get(self, persona_name: str = None) -> Dict:
        """Get scratch memories for a specific persona"""
        try:
            memory_type = self._get_memory_type(persona_name)
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting scratch memories: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update(self, data: str, persona_name: str = None) -> Dict:
        """Update scratch memory for a specific persona"""
        try:
            memory_type = self._get_memory_type(persona_name)
            request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "data": data
                }},
                options={"condition": {"type": memory_type}}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating scratch memory: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _get_memory_type(self, persona_name: str = None) -> str:
        """Generate the memory type with optional persona prefix"""
        if persona_name:
            return f"{persona_name}_{self.base_memory_type}"
        return self.base_memory_type
