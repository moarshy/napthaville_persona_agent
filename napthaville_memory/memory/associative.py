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

class Associative:
    def __init__(self, storage_config: Dict, storage_client: StorageClient):
        self.storage_client = storage_client
        self.table_name = storage_config["path"]
        self.memory_type = "associative"
        
    async def add_embedding(self, data: str) -> Dict:
        """Add associative memory embedding"""
        try:
            request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={ "data":{
                    "type": "associative-embeddings",
                    "data": data
                }
            })
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error adding associative embedding: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_node(self, data: str) -> Dict:
        """Add associative memory node"""
        try:
            request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={ "data":{
                    "type": "associative-nodes",
                    "data": data
                }
            })
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error adding associative node: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_kw_strength(self, data: str) -> Dict:
        """Add keyword strength"""
        try:
            request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={ "data":{
                    "type": "associative-kw-strength",
                    "data": data
                }
            })
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error adding keyword strength: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_embeddings(self) -> Dict:
        """Get associative memory embeddings"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": "associative-embeddings"}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting associative embeddings: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_nodes(self) -> Dict:
        """Get associative memory nodes"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": "associative-nodes"}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting associative nodes: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_kw_strength(self) -> Dict:
        """Get keyword strengths"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": "associative-kw-strength"}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting keyword strengths: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_embedding(self, data: str) -> Dict:
        """Update associative memory embedding"""
        try:
            request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "data": data
                }},
                options={"condition": {"type": "associative-embeddings"}}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating associative embedding: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_node(self, data: str) -> Dict:
        """Update associative memory node"""
        try:
            request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "data": data
                }},
                options={"condition": {"type": "associative-nodes"}}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating associative node: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_kw_strength(self, data: str) -> Dict:
        """Update keyword strength"""
        try:
            request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={"data": {
                    "data": data
                }},
                options={"condition": {"type": "associative-kw-strength"}}
            )
            await self.storage_client.execute(request)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error updating keyword strength: {str(e)}")
            return {"success": False, "error": str(e)}