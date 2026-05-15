# Voice-Driven Deterministic Prompt Optimization Engine

Full-stack assignment implementation for converting raw voice or unstructured multilingual input into a clean, minimal, high-signal prompt.

## Features

- Text and voice input support
- English, Hindi, and Hinglish normalization
- Intent extraction with confidence score
- User confirmation before optimization
- No confirmation, no execution
- Multiple unrelated task detection with clarification loop
- Prompt enhancement and minimum viable prompt generation
- Token metrics with clarified-vs-reduced reporting
- Validation layer for alignment, format, and efficiency
- SQLite sessions, workflow logs, and memory output
- React chat UI and workflow graph visualization

## Architecture

The Mermaid diagram is available in [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure

```text
backend/
  main.py              FastAPI routes and workflow control
  services.py          STT, normalization, intent, optimization, validation
  database.py          SQLite persistence and workflow logs
  models.py            Pydantic request/response schemas
  prompts.py           Deterministic LLM prompts
frontend/
  src/                 React chat and graph UI
init_db.sql            Database initialization script
requirements.txt       Backend dependencies
.env.example           Environment variable template
```

## Setup

1. Create and activate a Python environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install backend dependencies.

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`.

```bash
copy .env.example .env
```

Add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
GROQ_CHAT_MODEL_FALLBACKS=llama-3.1-8b-instant
```

4. Install frontend dependencies.

```bash
cd frontend
npm install
```

## Run

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

## Example API Calls

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

Get graph data:

```bash
curl http://localhost:8000/api/session/{session_id}/graph
```

Get decision logs:

```bash
curl http://localhost:8000/api/session/{session_id}/history
```

## Expected Example

Input:

```text
Ek marketing plan bana do for gym app.
```

Optimized prompt:

```text
Create a 3-step marketing plan for a gym app.
Output: bullet points
Constraint: under 100 words
```

## Failure Handling Examples

Multiple unrelated tasks:

```text
Create a plan for developing a gym mobile application and a pasta recipe
```

System response:

```text
I detected two different tasks: Pasta Recipe and Gym App Development. I can only optimize one clear instruction at a time to maintain high-signal quality. Which one should I proceed with?
```

The UI shows choice buttons for the detected tasks. Selecting one resolves the ambiguity inside the same session, then the app asks for confirmation before optimization. This blocks optimization until the ambiguity is resolved and records:

```text
Log: Detected multiple unrelated domains.
Action: Triggered clarification loop.
Reasoning: To prevent context window wastage and maintain deterministic behavior.
```

## Demo Checklist

- Show text input
- Show voice input
- Show intent confirmation
- Show no-confirmation/no-execution behavior
- Show memory save and skip logs
- Show graph visualization
- Show decision logs from session history
- Show final memory output in UI
- Show token reduction or clarified token change

## Notes

- Do not commit `.env` or database files.
- Use `.env.example` for sharing required environment variables.
- `prompt_optimizer.db` is created automatically by the backend if missing.
