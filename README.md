# Voice-Driven Deterministic Prompt Optimization Engine

A full-stack chat workflow that converts messy text or voice input into a clean, minimal, high-signal prompt. The system is designed for multilingual, noisy, real-world instructions where the model must first understand intent, ask for confirmation, then optimize only after the user approves.

## What This Solves

Users often give prompts that are vague, mixed-language, too verbose, or contain multiple unrelated tasks. This project turns that raw input into a deterministic workflow:

1. Capture text or voice input.
2. Normalize English, Hindi, Hinglish, and similar mixed input.
3. Extract structured intent with confidence.
4. Detect unsafe, unclear, or multi-task requests.
5. Ask for confirmation before optimization.
6. Produce a compact CAVEMAN MODE prompt with token metrics.
7. Save workflow logs, memory decisions, and graph data.

## Key Features

- React chat interface with text and voice input
- Groq Whisper transcription for voice input
- Multilingual normalization and noise removal
- Structured intent extraction with confidence score
- Mandatory user confirmation before optimization
- No confirmation, no execution
- Clarification loop for multiple unrelated tasks
- Unsafe or unauthorized request rejection
- CAVEMAN MODE token optimization with 30-50% target
- Validation layer for intent alignment, format, and efficiency
- SQLite persistence for sessions, workflow logs, and memory output
- Live workflow graph visualization using React Flow
- Pytest coverage for workflow rules and deterministic examples

## Architecture

The Mermaid workflow diagram is available in [ARCHITECTURE.md](ARCHITECTURE.md).

```text
User Input / Voice
  -> Normalization + Guardrails
  -> Intent Extraction
  -> Confidence / Ambiguity Check
  -> User Confirmation
  -> Prompt Optimization
  -> Validation
  -> Memory + Logs + Graph
```

## Project Structure

```text
backend/
  main.py              FastAPI routes and workflow control
  services.py          STT, normalization, intent, optimization, validation
  database.py          SQLite persistence and workflow logs
  models.py            Pydantic request/response schemas
  prompts.py           Deterministic LLM prompts
  graph_workflow.py    LangGraph workflow reference
frontend/
  src/                 React chat UI and workflow graph
tests/                 Workflow and optimization tests
init_db.sql            Database initialization script
requirements.txt       Backend dependencies
.env.example           Environment variable template
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
copy .env.example .env
```

Add your Groq credentials:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
GROQ_CHAT_MODEL_FALLBACKS=llama-3.1-8b-instant
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Run Locally

Start backend from the repository root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start frontend:

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## Test

```bash
pytest
```

Current test coverage checks:

- multiple-task clarification
- same-session ambiguity resolution
- deterministic prompt examples
- contextual memory follow-up
- unsafe request rejection
- token reduction target examples

## Demo Flow

Use this flow during evaluation:

1. Send a normal prompt:

```text
Ek marketing plan bana do for gym app.
```

Expected optimized output:

```text
Create a 3-step marketing plan for a gym app.
Output: bullet points
Constraint: under 100 words
```

2. Send a voice or text prompt in mixed language:

```text
Meko pasta ki recipe chahiye beginner ke liye.
```

Expected compact output:

```text
Beginner pasta | Text
```

3. Send multiple unrelated tasks:

```text
Create a plan for developing a gym mobile application and a pasta recipe
```

Expected behavior:

```text
The app asks which task to proceed with before optimization.
```

4. Try optimizing without confirmation:

```text
The backend blocks optimization until the intent is confirmed.
```

5. Check the graph panel:

```text
Workflow nodes update as the session moves through normalization, intent extraction, confirmation, optimization, validation, and memory.
```

## API Examples

Create a text session:

```bash
curl -X POST http://localhost:8000/api/process-text ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Ek marketing plan bana do for gym app.\"}"
```

Extract intent:

```bash
curl -X POST http://localhost:8000/api/extract-intent/{session_id}
```

Confirm intent:

```bash
curl -X POST http://localhost:8000/api/confirm-intent ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"{session_id}\",\"confirmed\":true}"
```

Optimize prompt:

```bash
curl -X POST http://localhost:8000/api/optimize-prompt/{session_id}
```

Get workflow graph:

```bash
curl http://localhost:8000/api/session/{session_id}/graph
```

Get decision logs:

```bash
curl http://localhost:8000/api/session/{session_id}/history
```

## Design Decisions

- Deterministic workflow: each stage logs its decision and blocks unsafe transitions.
- Confirmation-first execution: optimization is impossible until the extracted intent is confirmed.
- CAVEMAN MODE optimization: prompts are compressed into high-signal instructions while preserving constraints.
- Memory transparency: the final response includes saved memory so the evaluator can inspect what the system retained.
- Prototype-safe guardrails: harmful or unauthorized requests are rejected before intent confirmation.

## Notes

- This is an assignment-ready prototype, not a hardened production deployment.
- Do not commit `.env`, `node_modules`, or build artifacts.
- `prompt_optimizer.db` is created automatically by the backend if missing.
- Production hardening would add persistent server-side sessions, restricted CORS, auth, rate limiting, monitoring, and managed Postgres.
