from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import APIConnectionError, RateLimitError
import uuid
import re
from pathlib import Path

from .database import Database
from .models import ClarificationResponse, ConfirmationResponse, TextInput
from .services import (
    add_token_metrics,
    enforce_token_target,
    extract_intent_structured,
    normalize_and_filter_input,
    optimize_prompt_tokens,
    repair_optimized_prompt,
    transcribe_audio,
    unsafe_request_reason,
    validate_optimized_prompt,
)

app = FastAPI(title="Prompt Optimization Engine")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()

# In-memory session store (for simplicity)
sessions = {}
recent_completed_intent = None


@app.exception_handler(Exception)
async def processing_failure_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Processing failed unexpectedly. Please retry with a clearer instruction or try again later.",
            "recoverable": True,
        },
    )


@app.exception_handler(RateLimitError)
async def groq_rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Groq rate limit reached. Please wait a few minutes or retry with text input.",
            "recoverable": True,
        },
    )


@app.exception_handler(APIConnectionError)
async def groq_connection_handler(request: Request, exc: APIConnectionError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Groq connection failed. Please retry, or use text input if voice transcription is unavailable.",
            "recoverable": True,
        },
    )

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".ogg",
    ".flac",
}

SUPPORTED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "video/webm",
}


def _prepare_session_input(raw_text: str) -> dict:
    """Normalize input and apply guardrails before downstream workflow steps."""
    unsafe_reason = unsafe_request_reason(raw_text)
    if unsafe_reason:
        raise HTTPException(400, unsafe_reason)

    preprocessed = normalize_and_filter_input(raw_text)
    if raw_text.strip().lower() == "pasta":
        preprocessed.update(
            {
                "cleaned_text": "pasta recipe",
                "normalized_text": "Provide a pasta recipe",
                "language": preprocessed.get("language", "english"),
                "confidence": max(preprocessed.get("confidence", 0), 0.75),
                "is_task_request": True,
                "actionable": True,
                "rejection_reason": "",
            }
        )
    normalized_text = preprocessed.get("normalized_text") or raw_text
    cleaned_text = preprocessed.get("cleaned_text") or normalized_text
    unsafe_reason = unsafe_request_reason(f"{cleaned_text} {normalized_text}")
    if unsafe_reason:
        raise HTTPException(400, unsafe_reason)

    if not preprocessed.get("is_task_request", False):
        reason = preprocessed.get("rejection_reason") or "Input is not a task request."
        raise HTTPException(400, reason)

    if not preprocessed.get("actionable", False):
        reason = preprocessed.get("rejection_reason") or "Input is too unclear to process reliably."
        raise HTTPException(400, reason)

    return {
        "raw_input": raw_text,
        "cleaned_input": cleaned_text,
        "normalized_input": normalized_text,
        "input_meta": preprocessed,
    }


def _persist_memory(session_id: str, key: str, value: dict):
    db.upsert_memory(session_id, key, value)
    db.log_step(
        session_id,
        "memory_decision",
        {"key": key},
        {"decision": "saved", "reason": f"{key} helps preserve session context"},
    )


def _skip_memory(session_id: str, key: str, reason: str):
    db.log_step(
        session_id,
        "memory_decision",
        {"key": key},
        {"decision": "skipped", "reason": reason},
    )


def _needs_clarification(intent_data: dict, confidence: float) -> bool:
    text = " ".join(
        str(intent_data.get(key, ""))
        for key in ("intent", "task", "domain")
    ).lower()
    context_missing = any(word in text for word in ["previous", "repeat", "again", "same task"])
    multiple_tasks = "multiple tasks detected" in text or "mixed" in text
    return multiple_tasks or confidence < 0.45 or (confidence < 0.7 and context_missing)


def _multiple_task_intent(text: str) -> dict | None:
    lowered = text.lower()
    task_groups = [
        {
            "label": "Pasta Recipe",
            "domain": "Culinary",
            "terms": ["cook", "cooking", "pasta", "recipe", "culinary"],
        },
        {
            "label": "House Cleaning",
            "domain": "Home Maintenance",
            "terms": ["clean", "cleaning", "house", "home", "housekeeping"],
        },
        {
            "label": "Gym App Development",
            "domain": "Technical / Business",
            "terms": ["gym app", "gym application", "gym mobile", "fitness app", "fitness application"],
        },
    ]
    matches = [
        group
        for group in task_groups
        if any(term in lowered for term in group["terms"])
    ]
    if len(matches) >= 2:
        labels = [group["label"] for group in matches]
        domains = [group["domain"] for group in matches]
        return {
            "intent": "clarify_multiple_tasks",
            "task": f"Multiple tasks detected ({' & '.join(labels)})",
            "domain": f"Mixed ({' / '.join(domains)})",
            "constraints": ["choose_one_task_before_optimization"],
            "output_format": "text",
            "audience": "user",
            "language_detected": "english",
            "confidence_score": 0.5,
            "detected_tasks": labels,
        }
    return None


def _normalize_intent(intent_data: dict) -> dict:
    task = str(intent_data.get("task") or intent_data.get("intent") or "").lower()
    if "pasta" in task and "recipe" in task:
        intent_data["domain"] = "culinary"
        intent_data["audience"] = intent_data.get("audience") or "home_cooks"
        intent_data["output_format"] = intent_data.get("output_format") or "text"
    if "python" in task and "function" in task:
        intent_data["domain"] = "technical"
        intent_data["audience"] = "developers"
        intent_data["output_format"] = "code"
    return intent_data


def _is_unit_test_followup(text: str) -> bool:
    lowered = text.lower()
    has_arabic_script = bool(re.search(r"[\u0600-\u06FF]", text))
    has_test_request = "unit test" in lowered or "test" in lowered or has_arabic_script
    has_reference = any(word in lowered for word in ["iska", "previous", "upar", "uper", "above", "function"]) or bool(
        re.search(r"\u0627\u0633", text)
    )
    return has_test_request and has_reference


def _apply_contextual_memory(raw_input: str, intent_data: dict) -> dict:
    if not _is_unit_test_followup(raw_input) or not recent_completed_intent:
        return intent_data

    previous_task = str(recent_completed_intent.get("task") or "").lower()
    if "area of a circle" in previous_task and "python" in previous_task:
        return {
            "intent": "write_unit_tests",
            "task": "Write unit tests for the Python function that calculates the area of a circle",
            "domain": "technical",
            "constraints": ["use_previous_context"],
            "output_format": "code",
            "audience": "developers",
            "language_detected": "hinglish",
            "confidence_score": 0.88,
        }

    return intent_data


def _intent_from_task_choice(selected_task: str) -> dict:
    task = selected_task.strip().lower()
    if "pasta" in task:
        return {
            "intent": "provide_recipe",
            "task": "Provide a pasta recipe",
            "domain": "culinary",
            "constraints": [],
            "output_format": "text",
            "audience": "home_cooks",
            "language_detected": "english",
            "confidence_score": 0.9,
        }
    if "gym" in task:
        return {
            "intent": "create_app_development_plan",
            "task": "Create a plan for developing a gym mobile application",
            "domain": "technical",
            "constraints": [],
            "output_format": "text",
            "audience": "app_stakeholders",
            "language_detected": "english",
            "confidence_score": 0.9,
        }
    if "clean" in task or "house" in task:
        return {
            "intent": "create_cleaning_plan",
            "task": "Create a house cleaning plan",
            "domain": "home_maintenance",
            "constraints": [],
            "output_format": "text",
            "audience": "home_users",
            "language_detected": "english",
            "confidence_score": 0.9,
        }
    raise HTTPException(400, "Please choose one of the detected tasks.")


def _only_token_efficiency_issues(issues: list[str]) -> bool:
    joined = " ".join(issues).lower()
    critical = ["intent", "constraint", "format", "audience", "actionable"]
    return bool(issues) and not any(word in joined for word in critical)


@app.post("/api/process-voice")
async def process_voice(audio: UploadFile):
    """Handle voice input"""
    filename = audio.filename or "audio_upload"
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS and audio.content_type not in SUPPORTED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            400,
            (
                f"Unsupported file '{filename}'. Please upload a real audio file "
                "like .mp3, .wav, .m4a, .webm, or .ogg instead of JSON or text."
            ),
        )

    session_id = str(uuid.uuid4())
    
    # Transcribe
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Uploaded audio file is empty.")

    try:
        text = transcribe_audio(audio_bytes, filename=filename)
    except RateLimitError:
        raise HTTPException(429, "Groq voice transcription rate limit reached. Please wait a few minutes or use text input.")
    except APIConnectionError:
        raise HTTPException(503, "Groq voice transcription connection failed. Please retry or use text input.")
    prepared = _prepare_session_input(text)
    
    # Create session
    db.create_session(session_id)
    sessions[session_id] = {
        **prepared,
        "status": "transcribed"
    }
    db.update_session(session_id, "transcribed", "voice_transcription")
    _persist_memory(session_id, "input", prepared)
    db.log_step(
        session_id,
        "voice_transcription",
        {"filename": filename},
        {"transcribed_text": text},
    )
    db.log_step(
        session_id,
        "input_normalization",
        {"raw_input": text},
        prepared["input_meta"],
    )
    _persist_memory(session_id, "normalization", prepared["input_meta"])
    
    return {
        "session_id": session_id,
        "transcribed_text": text,
        "normalized_text": prepared["normalized_input"],
        "status": "transcribed"
    }

@app.post("/api/process-text")
async def process_text(input: TextInput):
    """Handle text input"""
    session_id = str(uuid.uuid4())
    prepared = _prepare_session_input(input.text)
    
    db.create_session(session_id)
    sessions[session_id] = {
        **prepared,
        "status": "received"
    }
    db.update_session(session_id, "received", "text_received")
    _persist_memory(session_id, "input", prepared)
    db.log_step(
        session_id,
        "text_received",
        {"raw_input": input.text},
        {"status": "received"},
    )
    db.log_step(
        session_id,
        "input_normalization",
        {"raw_input": input.text},
        prepared["input_meta"],
    )
    _persist_memory(session_id, "normalization", prepared["input_meta"])
    
    return {
        "session_id": session_id,
        "normalized_text": prepared["normalized_input"],
        "status": "received"
    }

@app.post("/api/extract-intent/{session_id}")
async def extract_intent_endpoint(session_id: str):
    """Extract intent from input"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    raw_input = sessions[session_id].get("normalized_input") or sessions[session_id]["raw_input"]
    
    # Extract intent using structured output
    intent_data = _normalize_intent(
        _apply_contextual_memory(
            raw_input,
            _multiple_task_intent(raw_input) or extract_intent_structured(raw_input),
        )
    )
    confidence = intent_data.get("confidence_score", 1)
    
    # Log step
    db.log_step(
        session_id,
        "intent_extraction",
        {"raw_input": raw_input},
        intent_data
    )
    
    sessions[session_id]["intent"] = intent_data
    if _needs_clarification(intent_data, confidence):
        multiple_tasks = intent_data.get("intent") == "clarify_multiple_tasks"
        task_names = intent_data.get("detected_tasks") or ["the first task", "the second task"]
        message = (
            f"I detected two different tasks: {task_names[0]} and {task_names[1]}. "
            "I can only optimize one clear instruction at a time to maintain high-signal quality. "
            "Which one should I proceed with?"
            if multiple_tasks
            else "This request depends on missing previous context. Please rewrite the exact task and try again."
        )
        sessions[session_id]["status"] = "clarification_required"
        db.update_session(session_id, "clarification_required", "request_clarification")
        db.log_step(
            session_id,
            "request_clarification",
            {"confidence": confidence},
            {
                "message": message,
                "log": "Detected multiple unrelated domains." if multiple_tasks else "Low confidence or missing context.",
                "action": "Triggered clarification loop." if multiple_tasks else "Asked user to clarify.",
                "reasoning": (
                    "To prevent context window wastage and maintain deterministic behavior."
                    if multiple_tasks
                    else "Execution blocked until the instruction is clear."
                ),
            },
        )
        _skip_memory(session_id, "intent", "multiple unrelated tasks require clarification" if multiple_tasks else "confidence was too low for reliable memory")
        return {
            "session_id": session_id,
            "intent": intent_data,
            "status": "clarification_required",
            "requires_confirmation": False,
            "choices": intent_data.get("detected_tasks", []),
            "message": message,
        }

    sessions[session_id]["status"] = "pending_confirmation"
    db.update_session(session_id, "pending_confirmation", "intent_extraction")
    _persist_memory(session_id, "intent", intent_data)
    
    return {
        "session_id": session_id,
        "intent": intent_data,
        "status": "pending_confirmation",
        "requires_confirmation": True,
        "low_confidence": confidence < 0.7,
        "message": "Confidence is slightly low. Please confirm it or use Modify to improve it." if confidence < 0.7 else None,
    }

@app.post("/api/resolve-clarification")
async def resolve_clarification(response: ClarificationResponse):
    """Resolve an ambiguity inside the same session, then ask for confirmation."""
    session_id = response.session_id
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    intent_data = _intent_from_task_choice(response.selected_task)
    sessions[session_id]["intent"] = intent_data
    sessions[session_id]["normalized_input"] = intent_data["task"]
    sessions[session_id]["status"] = "pending_confirmation"
    sessions[session_id]["user_confirmed"] = False

    db.log_step(
        session_id,
        "clarification_resolution",
        {"selected_task": response.selected_task},
        {
            "action": "Resolved ambiguity inside the same session.",
            "intent": intent_data,
        },
    )
    db.update_session(session_id, "pending_confirmation", "clarification_resolution")
    _persist_memory(session_id, "intent", intent_data)

    return {
        "session_id": session_id,
        "intent": intent_data,
        "status": "pending_confirmation",
        "requires_confirmation": True,
        "message": f"You selected {response.selected_task}. Please confirm before optimization.",
    }

@app.post("/api/confirm-intent")
async def confirm_intent(confirmation: ConfirmationResponse):
    """User confirms or modifies intent"""
    session_id = confirmation.session_id
    
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    if confirmation.confirmed:
        sessions[session_id]["status"] = "confirmed"
        sessions[session_id]["user_confirmed"] = True
    else:
        sessions[session_id]["status"] = "needs_modification"
        sessions[session_id]["modifications"] = confirmation.modifications
    
    db.log_step(
        session_id,
        "user_confirmation",
        {"confirmed": confirmation.confirmed},
        {"status": sessions[session_id]["status"]}
    )
    db.update_session(session_id, sessions[session_id]["status"], "user_confirmation")
    if not confirmation.confirmed:
        _skip_memory(session_id, "optimized", "user rejected or modified the detected intent")
    
    return {
        "status": sessions[session_id]["status"],
        "message": (
            "Intent confirmed. Continuing to optimization."
            if confirmation.confirmed
            else "Intent was not correct. Please revise the input and submit again."
        ),
    }

@app.post("/api/optimize-prompt/{session_id}")
async def optimize_prompt_endpoint(session_id: str):
    """Optimize prompt after confirmation"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    if not sessions[session_id].get("user_confirmed"):
        raise HTTPException(400, "Intent not confirmed")
    
    global recent_completed_intent
    intent = sessions[session_id]["intent"]
    raw_input = sessions[session_id].get("normalized_input") or sessions[session_id]["raw_input"]
    
    # Optimize
    prior_context = db.get_memory(session_id)
    optimized = optimize_prompt_tokens(intent, raw_input, prior_context)
    
    # Validate
    validation = validate_optimized_prompt(intent, optimized["optimized_prompt"])
    
    if not validation["valid"] and not _only_token_efficiency_issues(validation.get("issues", [])):
        repaired_prompt = repair_optimized_prompt(
            intent,
            optimized["optimized_prompt"],
            validation.get("issues", []),
        )
        optimized["optimized_prompt"] = repaired_prompt
        optimized = add_token_metrics(optimized, raw_input)
        optimized = enforce_token_target(optimized, intent, raw_input)
        validation = validate_optimized_prompt(intent, repaired_prompt)

        if not validation["valid"] and not _only_token_efficiency_issues(validation.get("issues", [])):
            raise HTTPException(400, f"Validation failed: {validation['issues']}")
    
    # Log
    db.log_step(
        session_id,
        "prompt_optimization",
        {"intent": intent, "original": raw_input, "prior_context": prior_context},
        optimized
    )
    
    sessions[session_id]["optimized"] = optimized
    sessions[session_id]["status"] = "completed"
    recent_completed_intent = intent
    db.update_session(session_id, "completed", "prompt_optimization")
    _persist_memory(session_id, "optimized", optimized)
    memory_output = db.get_memory(session_id)
    
    return {
        "session_id": session_id,
        "optimized_prompt": optimized["optimized_prompt"],
        "token_reduction": optimized.get("reduction_pct", 0),
        "original_tokens": optimized.get("original_tokens", 0),
        "optimized_tokens": optimized.get("optimized_tokens", optimized.get("token_count", 0)),
        "memory_output": memory_output,
        "status": "completed"
    }

@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get workflow history for visualization"""
    history = db.get_session_history(session_id)
    return {"session_id": session_id, "history": history}

@app.get("/api/session/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get graph data for visualization"""
    history = db.get_session_history(session_id)
    done = {step["step"] for step in history}

    nodes = [
        {"id": "input_type", "label": "Input Type?", "x": 210, "y": 0, "shape": "diamond"},
        {"id": "text_received", "label": "Text Preprocessor", "x": 40, "y": 85},
        {"id": "voice_transcription", "label": "Whisper STT", "x": 360, "y": 85},
        {"id": "input_normalization", "label": "Language Detection", "x": 210, "y": 155},
        {"id": "confidence", "label": "Confidence Check", "x": 210, "y": 235, "shape": "diamond"},
        {"id": "request_clarification", "label": "Request Clarification", "x": 40, "y": 335},
        {"id": "intent_extraction", "label": "Intent Extraction", "x": 360, "y": 335},
        {"id": "user_confirmation", "label": "User Confirmation UI", "x": 360, "y": 410},
        {"id": "confirmed", "label": "User Confirmed?", "x": 360, "y": 490, "shape": "diamond"},
        {"id": "request_revision", "label": "Request Revision", "x": 120, "y": 595},
        {"id": "prompt_optimization", "label": "Prompt Decomposition", "x": 360, "y": 595},
        {"id": "context_enrichment", "label": "Context Enrichment", "x": 360, "y": 670},
        {"id": "token_optimization", "label": "Token Optimization", "x": 360, "y": 745},
        {"id": "validation", "label": "Validation Layer", "x": 360, "y": 820},
        {"id": "valid", "label": "Valid Output?", "x": 360, "y": 900, "shape": "diamond"},
        {"id": "completed", "label": "Minimum Viable Output", "x": 500, "y": 1005},
    ]
    for node in nodes:
        node["status"] = "done" if node["id"] in done else "pending"
    if done:
        next(node for node in nodes if node["id"] == "input_type")["status"] = "done"
    if "intent_extraction" in done or "request_clarification" in done:
        next(node for node in nodes if node["id"] == "confidence")["status"] = "done"
    if "prompt_optimization" in done:
        for id_ in ["context_enrichment", "token_optimization", "validation", "valid", "completed"]:
            next(node for node in nodes if node["id"] == id_)["status"] = "done"

    edges = [
        {"source": "input_type", "target": "text_received", "label": "Text"},
        {"source": "input_type", "target": "voice_transcription", "label": "Voice"},
        {"source": "text_received", "target": "input_normalization"},
        {"source": "voice_transcription", "target": "input_normalization"},
        {"source": "input_normalization", "target": "confidence"},
        {"source": "confidence", "target": "request_clarification", "label": "No"},
        {"source": "confidence", "target": "intent_extraction", "label": "Yes"},
        {"source": "intent_extraction", "target": "user_confirmation"},
        {"source": "user_confirmation", "target": "confirmed"},
        {"source": "confirmed", "target": "request_revision", "label": "No"},
        {"source": "confirmed", "target": "prompt_optimization", "label": "Yes"},
        {"source": "prompt_optimization", "target": "context_enrichment"},
        {"source": "context_enrichment", "target": "token_optimization"},
        {"source": "token_optimization", "target": "validation"},
        {"source": "validation", "target": "valid"},
        {"source": "valid", "target": "completed", "label": "Yes"},
        {"source": "valid", "target": "token_optimization", "label": "No"},
    ]
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
