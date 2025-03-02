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
        self.base_memory_type = "associative"
        
    async def add_embedding(self, data: str, persona_name: str = None) -> Dict:
        """Add associative memory embedding for a specific persona"""
        try:
            memory_type = self._get_memory_type("embeddings", persona_name)
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
            logger.error(f"Error adding associative embedding: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_node(self, data: str, persona_name: str = None) -> Dict:
        """Add associative memory node for a specific persona"""
        try:
            memory_type = self._get_memory_type("nodes", persona_name)
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
            logger.error(f"Error adding associative node: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_kw_strength(self, data: str, persona_name: str = None) -> Dict:
        """Add keyword strength for a specific persona"""
        try:
            memory_type = self._get_memory_type("kw-strength", persona_name)
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
            logger.error(f"Error adding keyword strength: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_embeddings(self, persona_name: str = None) -> Dict:
        """Get associative memory embeddings for a specific persona"""
        try:
            memory_type = self._get_memory_type("embeddings", persona_name)
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting associative embeddings: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_nodes(self, persona_name: str = None) -> Dict:
        """Get associative memory nodes for a specific persona"""
        try:
            memory_type = self._get_memory_type("nodes", persona_name)
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting associative nodes: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_kw_strength(self, persona_name: str = None) -> Dict:
        """Get keyword strengths for a specific persona"""
        try:
            memory_type = self._get_memory_type("kw-strength", persona_name)
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                options={"conditions": [{"type": memory_type}]}
            )
            result = await self.storage_client.execute(request)
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error getting keyword strengths: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_embedding(self, data: str, persona_name: str = None) -> Dict:
        """Update associative memory embedding for a specific persona"""
        try:
            memory_type = self._get_memory_type("embeddings", persona_name)
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
            logger.error(f"Error updating associative embedding: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_node(self, data: str, persona_name: str = None) -> Dict:
        """Update associative memory node for a specific persona"""
        try:
            memory_type = self._get_memory_type("nodes", persona_name)
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
            logger.error(f"Error updating associative node: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_kw_strength(self, data: str, persona_name: str = None) -> Dict:
        """Update keyword strength for a specific persona"""
        try:
            memory_type = self._get_memory_type("kw-strength", persona_name)
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
            logger.error(f"Error updating keyword strength: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _get_memory_type(self, subtype: str, persona_name: str = None) -> str:
        """Generate the memory type with optional persona prefix"""
        base = f"{self.base_memory_type}-{subtype}"
        if persona_name:
            return f"{persona_name}_{base}"
        return base
