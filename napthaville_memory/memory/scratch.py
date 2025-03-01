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
        self.memory_type = "scratch"

    async def add(self, data: str) -> Dict:
        """Add scratch memory"""
        try:
            request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={ "data":{
                    "type": self.memory_type,
                    "data": data
                }
            })
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error adding scratch memory: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get(self) -> Dict:
        """Get scratch memories"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": self.memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting scratch memories: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update(self, data: str) -> Dict:
        """Update scratch memory"""
        try:
            request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "data": data
                }},
                options={"condition": {"type": self.memory_type}}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating scratch memory: {str(e)}")
            return {"success": False, "error": str(e)}