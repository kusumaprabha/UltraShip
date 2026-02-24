from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    chunks: int
    message: str

class AskRequest(BaseModel):
    file_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    confidence: float
    source_text: str
    source_chunk_index: int
    
class ExtractionResponse(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[float] = None
    carrier_name: Optional[str] = None
    
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None