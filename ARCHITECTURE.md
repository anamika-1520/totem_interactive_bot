# Voice-Driven Deterministic Prompt Optimization Engine

```mermaid
flowchart TD
    A{Input Type?} -->|Text| B[Text Input API]
    A -->|Voice| C[Voice Upload API]
    C --> D[Speech-to-Text]
    B --> E[Input Normalization]
    D --> E
    E --> F{Actionable and Safe?}
    F -->|No| G[Reject / Ask for Clear Task]
    F -->|Yes| H[Intent Extraction]
    H --> I{Confidence / Ambiguity Check}
    I -->|Low confidence or multiple tasks| J[Clarification Loop]
    J -->|User selects one task| K
    I -->|Clear| K[User Confirmation UI]
    K --> L{Confirmed?}
    L -->|No| M[Return Draft for Modification]
    L -->|Yes| N[Prompt Enhancement and Decomposition]
    N --> O[Token Optimization / MVP Prompt]
    O --> P[Validation Layer]
    P --> Q{Valid?}
    Q -->|No| O
    Q -->|Yes| R[Save Memory + Decision Logs]
    R --> S[Final Optimized Prompt]

    subgraph Storage
        T[(SQLite Sessions)]
        U[(Workflow Steps)]
        V[(Memory)]
    end

    B --> T
    C --> T
    E --> U
    H --> U
    J --> U
    K --> U
    R --> U
    R --> V
```

## Processing Layers

- Input layer: text and voice upload endpoints.
- Normalization layer: removes noise, normalizes Hindi/Hinglish/English input to English, and rejects non-task input.
- Intent layer: extracts task, domain, constraints, output format, audience, and confidence.
- Confirmation layer: blocks execution until the user confirms the interpreted intent.
- Optimization layer: creates a minimum viable prompt and preserves required constraints.
- Validation layer: checks intent alignment, format correctness, and token efficiency.
- Memory/log layer: records save/skip decisions, workflow steps, and final memory output.
