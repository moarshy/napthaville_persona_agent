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

class Spatial:
    def __init__(self, storage_config: Dict, storage_client: StorageClient):
        self.storage_client = storage_client
        self.table_name = storage_config["path"]
        self.memory_type = "spatial"

    async def add(self, data: str) -> Dict:
        """Add spatial memory"""
        try:
            logger.info(f"Adding spatial memory: {data}")
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
            logger.error(f"Error adding spatial memory: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get(self) -> Dict:
        """Get spatial memories"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": self.memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting spatial memories: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update(self, data: str) -> Dict:
        """Update spatial memory"""
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
            logger.error(f"Error updating spatial memory: {str(e)}")
            return {"success": False, "error": str(e)}