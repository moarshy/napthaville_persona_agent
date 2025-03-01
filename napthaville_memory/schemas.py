from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class MemoryType(str, Enum):
    """Enum for memory type"""
    SPATIAL = "spatial"
    ASSOCIATIVE = "associative"
    SCRATCH = "scratch"
    ALL = "all"

class AssociativeSubType(str, Enum):
    """Enum for associative memory subtypes"""
    EMBEDDINGS = "embeddings"
    NODES = "nodes"
    KW_STRENGTH = "kw_strength"

class OperationType(str, Enum):
    """Enum for operation type"""
    ADD = "add"
    UPDATE = "update"

class AvailableFunctions(str, Enum):
    """Enum for available functions"""
    INIT = "init"
    GET_MEMORY = "get_memory"
    SET_MEMORY = "set_memory"

class GetMemoryInput(BaseModel):
    """Schema for get memory input"""
    memory_type: MemoryType = MemoryType.ALL
    subtype: Optional[AssociativeSubType] = None

class SetMemoryInput(BaseModel):
    """Schema for set memory input"""
    memory_type: MemoryType
    operation: OperationType = OperationType.ADD
    data: str
    subtype: Optional[AssociativeSubType] = None

class InputSchema(BaseModel):
    """Schema for function input"""
    function_name: AvailableFunctions
    function_input_data: Optional[Union[GetMemoryInput, SetMemoryInput, Dict]] = None

class MemoryData(BaseModel):
    """Schema for memory data"""
    memory_type: MemoryType
    memory_data: str

class AssociativeMemoryData(BaseModel):
    """Schema for associative memory data"""
    embeddings: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    kw_strength: List[Dict[str, Any]] = Field(default_factory=list)

class OutputSchema(BaseModel):
    """Schema for output"""
    success: bool
    data: Optional[Union[List[MemoryData], Dict]] = None
    error: Optional[str] = None