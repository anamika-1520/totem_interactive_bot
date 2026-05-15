from pydantic import BaseModel, Field
from typing import Literal, Optional

class VoiceInput(BaseModel):
    audio_data: str  # base64 encoded
    
class TextInput(BaseModel):
    text: str
    
class IntentSchema(BaseModel):
    """Structured output for intent extraction"""
    intent: str = Field(description="Primary user intent")
    task: str = Field(description="Specific task to accomplish")
    domain: str = Field(description="Domain/category")
    constraints: list[str] = Field(default_factory=list)
    output_format: str = Field(description="Expected output format")
    audience: str = Field(description="Target audience")
    language_detected: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class ConfirmationRequest(BaseModel):
    session_id: str
    intent: IntentSchema
    original_input: str
    
class ConfirmationResponse(BaseModel):
    session_id: str
    confirmed: bool
    modifications: Optional[str] = None

class ClarificationResponse(BaseModel):
    session_id: str
    selected_task: str

class OptimizedPrompt(BaseModel):
    original_prompt: str
    optimized_prompt: str
    token_reduction_pct: float
    optimization_steps: list[str]
    
class WorkflowResponse(BaseModel):
    session_id: str
    status: Literal["pending_confirmation", "processing", "completed", "error"]
    current_step: str
    data: dict
