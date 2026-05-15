# graph_workflow.py - THE CORE OF YOUR PROJECT

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator

# State Schema
class WorkflowState(TypedDict):
    raw_input: str
    language: str
    confidence: float
    extracted_intent: dict
    user_confirmed: bool
    optimized_prompt: str
    token_reduction: float
    errors: list
    step_history: Annotated[list, operator.add]

# Node Functions
def process_input(state: WorkflowState) -> WorkflowState:
    """Detect language and calculate confidence"""
    # Implementation here
    
def extract_intent(state: WorkflowState) -> WorkflowState:
    """Use GPT-4o-mini with structured output"""
    # Implementation here
    
def require_confirmation(state: WorkflowState) -> WorkflowState:
    """Pause workflow for user confirmation"""
    # Implementation here
    
def optimize_prompt(state: WorkflowState) -> WorkflowState:
    """Apply token reduction strategies"""
    # Implementation here
    
def validate_output(state: WorkflowState) -> WorkflowState:
    """Check quality and token count"""
    # Implementation here

# Routing Logic
def should_confirm(state: WorkflowState) -> Literal["confirm", "reprocess"]:
    if state["confidence"] < 0.7:
        return "reprocess"
    return "confirm"

def check_confirmation(state: WorkflowState) -> Literal["optimize", "reextract"]:
    if state["user_confirmed"]:
        return "optimize"
    return "reextract"

# Build Graph
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("process_input", process_input)
workflow.add_node("extract_intent", extract_intent)
workflow.add_node("require_confirmation", require_confirmation)
workflow.add_node("optimize_prompt", optimize_prompt)
workflow.add_node("validate_output", validate_output)

# Add edges
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "extract_intent")
workflow.add_conditional_edges(
    "extract_intent",
    should_confirm,
    {
        "confirm": "require_confirmation",
        "reprocess": "process_input"
    }
)
workflow.add_conditional_edges(
    "require_confirmation",
    check_confirmation,
    {
        "optimize": "optimize_prompt",
        "reextract": "extract_intent"
    }
)
workflow.add_edge("optimize_prompt", "validate_output")
workflow.add_edge("validate_output", END)

# Compile
app = workflow.compile()